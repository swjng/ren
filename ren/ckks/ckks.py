import math
import operator
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import cache, reduce
from operator import add
from typing import TYPE_CHECKING, Self, overload

import numpy as np
from flint import acb, arb
from ren.ckks.utils import NoiseStats

from ren.bigfloat import BigFloat, FloatLike
from ren.container import PolyContainer
from ren.poly import Poly
from ren.random import ChaChaRng
from ren.ring import Residue, RingSpec
from ren.ckks.errors import InvalidParameterError, ParameterMismatchError

from ._rlwe import RLWECiphertext, _fresh_encryption_noise, _reduce_keeping
from .context import context
from .keys import SwitchKey
from .metadata import CKKSNoiseInfo
from . import encoding
from .encoding import RawSlots, Scalar
from .utils import count_calls, steps_to_galois_elt


if TYPE_CHECKING:
    from .multiparty import PartialDecryption


@dataclass(frozen=True, kw_only=True)
class SlotArithmetic[Other](PolyContainer, ABC):
    """Arithmetic and level alignment for encoded or encrypted CKKS slots.

    Values have a polynomial representation and noise bounds in a common ring,
    with optional diagnostic slot values.
    """

    __slots__ = ()  # Concrete dataclasses allocate storage, avoiding conflicts with RLWE's slots.

    noise_info: CKKSNoiseInfo
    # Optional message-tracking field. ``shape == (slot_count,)``, dtype
    # ``object`` with python-flint ``acb`` entries.
    _values: np.ndarray | None = None

    if TYPE_CHECKING:

        @property
        def ring(self) -> RingSpec: ...

    def __tree_reduce__(self):
        # Noise bounds and tracked values stay out of tree traversal and jit plan keys.
        return _reduce_keeping(self, ("noise_info", "_values"))

    @property
    def values(self) -> np.ndarray:
        if self._values is None:
            raise RuntimeError("plaintext value tracking is not available")
        return np.array([complex(v) for v in self._values])

    def _map_values(self, fn: Callable[..., np.ndarray], *others: SlotArithmetic) -> np.ndarray | None:
        """Apply ``fn`` to tracked slot values, or return None if tracking is unavailable."""
        ctx = context(required=False)
        if ctx is None or not ctx.track_plaintext_values or self._values is None:
            return None
        values = [self._values]
        for other in others:
            if other._values is None:
                return None
            values.append(other._values)
        return encoding.apply(fn, *values, prec=ctx._slot_prec)

    @abstractmethod
    def _prepare(self, other: object) -> tuple[Self, Self] | None:
        """Align compatible encoded operands, in self/other order."""
        raise NotImplementedError

    def _coerce(self, other: object) -> object:
        if not isinstance(other, Scalar):
            return other
        return Plaintext.const(other, level=self.level)

    def _binary(
        self,
        other: object,
        *,
        polys: Callable[[Poly, Poly], Poly],
        noise_info: Callable[[CKKSNoiseInfo, CKKSNoiseInfo], CKKSNoiseInfo],
        values: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> Self:
        """Combine compatible operands, deferring unsupported types to reflected operators."""
        other = self._coerce(other)
        operands = self._prepare(other)
        if operands is None:
            return NotImplemented
        left, right = operands
        return left.replace(
            polys=map(polys, left.polys(), right.polys(), strict=True),
            noise_info=noise_info(left.noise_info, right.noise_info),
            _values=left._map_values(values, right),
        )

    @count_calls
    def __add__(self, other: Self | Other) -> Self:
        return self._binary(other, polys=operator.add, noise_info=operator.add, values=operator.add)

    @count_calls
    def __radd__(self, other: Self | Other) -> Self:
        return self._binary(other, polys=operator.add, noise_info=operator.add, values=operator.add)

    @count_calls
    def __sub__(self, other: Self | Other) -> Self:
        return self._binary(other, polys=operator.sub, noise_info=operator.add, values=operator.sub)

    @count_calls
    def __rsub__(self, other: Self | Other) -> Self:
        return self._binary(other, polys=lambda a, b: b - a, noise_info=operator.add, values=lambda a, b: b - a)

    def __neg__(self) -> Self:
        return self.map_polys(lambda poly: -poly).replace(_values=self._map_values(lambda values: -values))

    @classmethod
    @abstractmethod
    def const(cls, val: Scalar, *, msg_bound: float | None = None, level: int | None = None) -> Self:
        raise NotImplementedError

    @classmethod
    def zero(cls, *, level: int | None = None) -> Self:
        """Return an encoded or trivially encrypted zero."""
        return cls.const(0, level=level)

    @classmethod
    def one(cls, *, level: int | None = None) -> Self:
        """Return an encoded or trivially encrypted one."""
        return cls.const(1, level=level)

    @property
    def scale(self) -> BigFloat:
        return context().params.scale(self.ring)

    @property
    def level(self) -> int:
        return context().params.level_of_ring(self.ring)

    def to_level(self, level: int) -> Self:
        """Drop to the specified level, adjusting the scale accordingly.

        The parameter transition owns the working rings and rounding boundaries.
        Its arithmetic is shared by plaintext and ciphertext conversion.
        """
        if level < 0:
            raise ValueError(f"Level must be nonnegative, got {level}")
        params = context().params
        if self.level < level:
            raise ValueError(f"Cannot raise {type(self).__name__} from level {self.level} to {level}")
        if self.level == level:
            return self

        transition = params.transition(self.level, level)
        return self.map_polys(transition.apply).replace(noise_info=self.noise_info * transition.poly_ratio)

    def _align[Value: SlotArithmetic](self, other: Value) -> tuple[Self, Value]:
        """Drop supported operands to their common scheduled level."""
        if self.ring == other.ring:
            return self, other
        level = min(self.level, other.level)
        return self.to_level(level), other.to_level(level)

    def _rescale(self, target_moduli: tuple[int, ...]) -> Self:
        """Rescale the basis, discarding tracked slot values.

        Scaling coefficients by added_prod / removed_prod is not a pure message
        transformation: the represented value also depends on the destination
        ring's scale, including reserved bootstrap transitions. Callers must
        reattach correctly scaled values. Clear tracking in the reconstruction
        so overflow checks never see stale, mis-scaled values.
        """
        noise_info = self.noise_info._rescale(self.ring.moduli, target_moduli)
        return self.map_polys(lambda poly: poly.rescale(target_moduli)).replace(noise_info=noise_info, _values=None)


@dataclass(slots=True, frozen=True)
class Plaintext(SlotArithmetic[Scalar]):
    """Encoded CKKS slot vector backed by one polynomial.

    Plaintexts carry the encoded polynomial plus scale/noise metadata. Most
    operations that read ``scale`` or ``level`` need an active ``CKKSContext``.
    """

    m: Poly

    @property
    def ring(self) -> RingSpec:
        return self.m.ring

    @classmethod
    def const(cls, val: Scalar, *, msg_bound: float | None = None, level: int | None = None) -> Self:
        """Encode a slot-constant plaintext without constructing a dense length-n
        coefficient vector. In coefficient space this plaintext only has nonzero terms
        on 1 and, for complex values with nonzero imaginary part, X^(n/2).

        msg_bound is public metadata and defaults to abs(val).
        """
        if isinstance(val, np.integer):
            val = int(val)
        pt = encode(val, msg_bound=abs(val) if msg_bound is None else msg_bound, level=level)
        return cls(m=pt.m, noise_info=pt.noise_info, _values=pt._values)

    def __getstate__(self):
        # ``values`` contains python-flint ``acb`` objects, which are not
        # picklable. It is debug-only tracking data.
        return {
            "m": self.m,
            "noise_info": self.noise_info,
        }

    def __setstate__(self, state):
        object.__setattr__(self, "m", state["m"])
        object.__setattr__(self, "noise_info", state["noise_info"])
        object.__setattr__(self, "_values", None)

    def encrypt(self, rng: ChaChaRng | None = None) -> Ciphertext:
        """
        Encrypt using the current context.
        """
        return encrypt(self, rng=rng)

    def _prepare(self, other: object) -> tuple[Plaintext, Plaintext] | None:
        return self._align(other) if isinstance(other, Plaintext) else None

    @count_calls
    def __mul__(self, other: Plaintext | Scalar) -> Plaintext:
        if isinstance(other, int | np.integer):
            scalar = int(other)
            noise_info = self.noise_info * scalar
            values = self._map_values(lambda values: values * scalar)
            return self.map_polys(lambda poly: poly * scalar).replace(noise_info=noise_info, _values=values)
        operand = self._coerce(other)
        if not isinstance(operand, Plaintext):
            return NotImplemented
        # Plaintext products must be rescaled to the next scheduled level.
        self, right = self._align(operand)
        if self.level == 0:
            raise InvalidParameterError("cannot multiply plaintexts at level 0")
        target_ring = context().params.ring_at(self.level - 1)

        noise_info = self.noise_info * right.noise_info
        unscaled_product = self.map_polys(lambda poly: poly * right.m).replace(noise_info=noise_info, _values=None)
        output = unscaled_product._rescale(target_ring.moduli)
        return output.replace(_values=self._map_values(operator.mul, right))

    @overload
    def __rmul__(self, other: Plaintext | Scalar) -> Plaintext: ...
    @overload
    def __rmul__(self, other: RLWECiphertext) -> RLWECiphertext: ...
    @count_calls
    def __rmul__(self, other: Plaintext | Scalar | RLWECiphertext) -> Plaintext | RLWECiphertext:
        """``other * self``: for an RLWE ciphertext, the product of both of its
        components with this plaintext, in the same ring and not rescaled.
        (A :class:`Ciphertext` has its own ``*``, which rescales.)"""
        if not isinstance(other, RLWECiphertext):
            return self.__mul__(other)
        if other.ring != self.ring:
            raise ParameterMismatchError("plaintext ring does not match the ciphertext ring")
        return replace(other, a=other.a * self.m, b=other.b * self.m, noise_info=other.noise_info * self.noise_info)

    def to_ciphertext(self) -> Ciphertext:
        """
        Returns a trivial encryption of this plaintext.
        """
        return Ciphertext(a=Poly.zero(self.ring), b=self.m, noise_info=self.noise_info, _values=self._values)

    def rotate(self, steps: int) -> Plaintext:
        """
        Rotate plaintext by the given number of steps.

        Positive steps rotate left; negative steps rotate right.
        A value at slot i moves to slot (i - steps) % slot_count.
        """
        steps %= self.ring.n // 2
        galois_elt = steps_to_galois_elt(steps, self.ring.n)
        values = self._map_values(lambda values: np.roll(values, -steps))
        return self.map_polys(lambda poly: poly.automorphism(galois_elt)).replace(_values=values)

    def noise_stats(self, values: Sequence[complex | arb | acb] | np.ndarray | None = None) -> NoiseStats:
        if values is None:
            values = self._values
        if values is None:
            raise RuntimeError("Noise stats can only be computed if plaintext values are tracked or provided")

        prec = context()._slot_prec
        actual = encoding._decode(self.m, scale=self.scale, prec=prec)
        expected = encoding.slots(values, self.ring.n // 2, prec=prec)
        return encoding.apply(lambda: NoiseStats(actual - expected, prec=prec), prec=prec)


def encode(values: RawSlots, *, msg_bound: FloatLike, level: int | None = None, prec: int | None = None) -> Plaintext:
    """Encode Python values into a CKKS plaintext.

    Slot i is placed at evaluation point ζ^(5^i), ensuring that the Galois
    automorphism σ_5 (x -> x^5) corresponds to slot rotation by 1.

    Args:
        values: A scalar to broadcast to every slot, or a vector of slot values.
            Short vectors are padded with zeros.
        msg_bound: An upper bound on the magnitude of every slot value.
        level: CKKS level. Defaults to the context's ``default_level``.
            For generated parameters with bootstrapping enabled, use levels 0 through
            ``bootstrap_output_level`` for ordinary computation. A product that crosses
            from ``bootstrap_output_level + 1`` to ``bootstrap_output_level`` can decode
            at the wrong scale.
        prec: Precision in bits. Defaults to the context's ``encoding_precision_bits``.
            Positive values up to 53 use NumPy; higher values use FLINT.

    Returns:
        A plaintext at the selected level.

    Raises:
        InvalidParameterError: If the precision is invalid or FLINT cannot determine
            a unique rounded integer for each scaled coefficient.

    Warning: Public metadata
        ``msg_bound`` is public metadata. Do not compute it from secret values.
    """
    ctx = context()
    ring = ctx.params.ring_at(ctx.default_level if level is None else level)
    scale = ctx.params.scale(ring)
    prec = ctx.encoding_precision_bits if prec is None else prec
    poly, noise_info = encoding._encode(values, ring, scale=scale, msg_bound=msg_bound, prec=prec)
    tracked = encoding.slots(values, ring.n // 2, prec=ctx._slot_prec) if ctx.track_plaintext_values else None
    return Plaintext(m=poly, noise_info=noise_info, _values=tracked)


def decode(pt: Plaintext, length: int | None = None, *, prec: int | None = None) -> list[complex]:
    """Decode slots using ``prec`` or the context's ``encoding_precision_bits``.

    Results are always returned as Python ``complex`` values.
    """
    ctx = context()
    prec = ctx.encoding_precision_bits if prec is None else prec
    slots = encoding._decode(pt.m, scale=pt.scale, prec=prec)
    return slots[:length].astype(np.complex128, copy=False).tolist()


def encrypt(pt: Plaintext, rng: ChaChaRng | None = None) -> Ciphertext:
    """Encrypt a plaintext using the active CKKS context.

    Requires a public key or a single-party secret key. Multiparty contexts
    require the collective public key. Preserves the plaintext's tracked values.
    """
    ctx = context()

    if ctx.keyset.sk is not None and ctx.multiparty is None:
        # TODO: handle coeffs vs slots noise
        # TODO use heuristic noise
        noise_info = _fresh_encryption_noise(pt.ring.n, ctx.params.noise_dist) + pt.noise_info
        rlwe = RLWECiphertext._encrypt(pt.m, secret=ctx.keyset.sk, error_dist=ctx.params.noise_dist, output_noise_info=noise_info, rng=rng)
    elif ctx.keyset.pk is not None:
        rlwe = ctx.keyset.pk.encrypt(pt.m, ctx.params, noise_info=pt.noise_info, rng=rng)
    else:
        raise RuntimeError("encryption requires a public key or a single-party secret key")
    return Ciphertext(a=rlwe.a, b=rlwe.b, noise_info=rlwe.noise_info, _values=pt._values)


@overload
def decrypt(ct: Ciphertext, *, secret: Poly | None = None) -> Plaintext: ...


@overload
def decrypt(ct: Ciphertext, *, partials: Sequence[PartialDecryption]) -> Plaintext: ...


def decrypt(ct: Ciphertext, *, secret: Poly | None = None, partials: Sequence[PartialDecryption] | None = None) -> Plaintext:
    """Decrypt with a secret key or partial decryptions.

    Without an explicit ``secret`` or ``partials``, use the active single-party
    keyset. Supply one partial per participating party. All partials must
    correspond to the same ciphertext. Shamir partials must use the same quorum.
    Secret-key basis extension requires small coefficients.
    """
    if partials is not None and secret is not None:
        raise InvalidParameterError("secret and partials are mutually exclusive")

    if partials is not None:
        if not partials:
            raise ParameterMismatchError("at least one partial decryption is required")
        m = sum((partial.d for partial in partials), start=ct.b)
        # Count the ciphertext's noise once; shares contribute only flooding noise.
        noise_info = ct.noise_info + CKKSNoiseInfo.sum(partial.noise_info for partial in partials)
        return Plaintext(m=m, noise_info=noise_info, _values=ct._values)

    if secret is None:
        if (ctx := context()).multiparty is not None:
            raise RuntimeError("use partial_decrypt with a multiparty keyset")
        if (secret := ctx.keyset.sk) is None:
            raise RuntimeError("CKKSContext has no secret key configured")

    return Plaintext(m=ct._decrypt(secret), noise_info=ct.noise_info, _values=ct._values)


def _keyswitch(
    c: Poly,
    key: SwitchKey,
    *,
    residue: Residue = Residue.BALANCED,
) -> RLWECiphertext:
    """
    Method-II key-switching for both relinearization and Galois rotation.

    Key-switching transforms a ciphertext component c associated with secret
    s_old, producing (a, b) such that: a·s_new + b ≈ c·s_old.

    For relinearization: switch from s² to s.
    For Galois rotation (permute-last): switch from s to σ_{g⁻¹}(s).
    The caller then applies σ_g to the combined ciphertext, mapping its secret
    back to s.

    Method II flow:
      1. For each D segment i, extract c on segment primes and base-convert to Q̃
      2. External product: sum over segments of c_segment * key_digit
      3. Rescale from Q̃ to Q (removes prod(P) factor)

    Implemented as the composition of the two reusable halves:
    `_decompose_and_modraise` (step 1, hoistable) and `_keyswitch_hoisted`
    (steps 2-3, per-key).
    """
    return _keyswitch_hoisted(_decompose_and_modraise(c, key, residue=residue), key)


@dataclass(frozen=True)
class HoistedDigits:
    """Pre-decomposed, mod-raised digits of a single polynomial `c1`.

    Produced once by `_decompose_and_modraise`; reused across many key-switches
    (the "hoist"). Each digit is one RNS segment of `c1` base-converted to the
    extended ring Q̃, stored in COEFF domain (the form the external product and
    the final rescale both consume). The digit ranges identify their original
    positions in the target Q ring.
    """

    digits: tuple[Poly, ...]
    digit_ranges: tuple[range, ...]

    @property
    def ring_qt(self) -> RingSpec:
        return self.digits[0].ring

    @property
    def target_ring(self) -> RingSpec:
        q_prime_count = sum(len(digit_range) for digit_range in self.digit_ranges)
        return self.ring_qt[-q_prime_count:]


def _decompose_and_modraise(
    c: Poly,
    key: SwitchKey,
    *,
    residue: Residue = Residue.BALANCED,
) -> HoistedDigits:
    """Shared half of key-switching: digit-decompose `c` and mod-raise to Q̃.

    This is the expensive, rotation-independent work (one base conversion per
    RNS segment). It is hoisted out so a single call feeds many key-switches of
    the same `c` under different Galois keys.

    The switch key describes its own P basis and digit partition. A wider key is
    projected onto `c.ring`, allowing one key to cover multiple target levels.
    """
    projection = key._project_digits(c.ring)
    ring_qt = RingSpec(n=c.ring.n, moduli=key.p_primes + c.ring.moduli, dtype=c.ring.dtype)
    digit_ranges = tuple(digit_range for _, digit_range in projection)

    digits: list[Poly] = []
    for digit_range in digit_ranges:
        seg_moduli = c.ring.moduli[digit_range.start : digit_range.stop]

        ring_seg = RingSpec(n=c.ring.n, moduli=seg_moduli, dtype=c.ring.dtype)
        c_seg = c.to_quotient_ring(ring_seg)
        c_seg_qt = c_seg.lift_to_ring(ring_qt, residue=residue)
        digits.append(c_seg_qt)

    return HoistedDigits(digits=tuple(digits), digit_ranges=digit_ranges)


@cache
def _key_switch_norm_bound(seg_moduli: tuple[int, ...], n: int) -> BigFloat:
    """Heuristic coefficient bound of a hoisted digit over ``seg_moduli``.

    BigFloat: a single digit over a wide basis (e.g. KG+ master keys covering
    all of P ∪ Q) has a product beyond float range. Cached by segment, since
    the key-switch hot path asks for it once per digit per switch.
    """
    # TODO this is heuristic, is that OK?
    return BigFloat(math.prod(seg_moduli)) * math.sqrt(n) / 4


def _keyswitch_product(hoisted: HoistedDigits, key: SwitchKey) -> RLWECiphertext:
    """Per-key half of key-switching, without the final rescale.

    Consumes the pre-decomposed digits from `_decompose_and_modraise` and returns
    the summed external products in the extended ring Q̃, where the message is
    still scaled by the special-modulus product. Outputs for the same
    target Q ring share `ring_qt`, so a caller may add several of them and
    rescale the sum once.
    """
    target_ring = hoisted.target_ring
    n = hoisted.ring_qt.n
    projection = key._project_digits(target_ring)
    if tuple(digit_range for _, digit_range in projection) != hoisted.digit_ranges:
        raise ParameterMismatchError("switch key digit layout does not align with the hoisted digits")
    hoisted_p_prime_count = len(hoisted.ring_qt.moduli) - len(target_ring.moduli)
    if key.p_primes != hoisted.ring_qt.moduli[:hoisted_p_prime_count]:
        raise ParameterMismatchError("switch key uses a different P basis than the hoisted digits")

    ks_terms: list[RLWECiphertext] = []
    for (key_digit_index, digit_range), digit in zip(projection, hoisted.digits, strict=True):
        norm_bound = _key_switch_norm_bound(target_ring.moduli[digit_range.start : digit_range.stop], n)
        term = key.digits[key_digit_index].map_polys(lambda poly: poly.to_quotient_ring(hoisted.ring_qt))
        ks_terms.append(term._multiply_small_polynomial_coeffs(digit, norm_bound=norm_bound))

    return reduce(add, ks_terms)


def _keyswitch_hoisted(hoisted: HoistedDigits, key: SwitchKey) -> RLWECiphertext:
    """Per-key half of key-switching: external product with `key` then rescale.

    Consumes the pre-decomposed digits from `_decompose_and_modraise`, so it carries
    none of the base-conversion cost. Called once per target rotation/key.
    """
    ks = _keyswitch_product(hoisted, key)
    # rescale from Q̃ to Q
    return ks._rescale(target_moduli=hoisted.target_ring.moduli)


def _switch_secret(
    ct: Ciphertext,
    key: SwitchKey,
    *,
    hoisted: HoistedDigits | None = None,
) -> Ciphertext:
    """Re-encrypt `ct` under the secret `key` encrypts with.

    `key` holds encryptions of the ciphertext's current secret s_old under a
    new secret s_new (masked by prod(P) per segment, see
    `SwitchKey.generate`), so b + KS(a) decrypts under s_new. Used by
    bootstrap's sparse-secret encapsulation and by `_rotate_by_galois_elt`
    (rotation is a switch to the automorphed secret followed by the
    automorphism).

    The key's own P basis and digit layout determine how it applies to the
    ciphertext ring. An existing decomposition may be supplied when several
    key-switches share the same `a` component.
    """
    if hoisted is None:
        hoisted = _decompose_and_modraise(ct.a, key)
    elif hoisted.target_ring != ct.ring:
        raise ParameterMismatchError("hoisted digits do not match this ciphertext's ring")
    ks_ct = Ciphertext._from_rlwe(_keyswitch_hoisted(hoisted, key), msg_bound=0)
    return ks_ct + ct.replace(a=Poly.zero(ct.ring))


@dataclass(slots=True, frozen=True)
class Ciphertext(SlotArithmetic[Plaintext | Scalar], RLWECiphertext):
    """Encrypted CKKS value backed by two polynomial components.

    Ren names the components ``a`` and ``b``. Ciphertexts also carry scale,
    level, and noise metadata used by arithmetic, alignment, and diagnostics.
    """

    def __bool__(self) -> bool:
        raise TypeError("A ciphertext has no Python truth value")

    def __post_init__(self) -> None:
        RLWECiphertext.__post_init__(self)
        current = context(required=False)
        if self.noise_info.msg_bound != math.inf and self.noise_info.msg_bound * 2 > self.ring.q:
            if current is not None and current.strict_noise_bounds:
                raise OverflowError("message bound exceeds modulus q/2, so overflow may occur.")
            # ``frozen=True`` blocks normal attribute assignment; the dataclass
            # docs sanction ``object.__setattr__`` in ``__post_init__`` for
            # post-construction normalization like this.
            object.__setattr__(self, "noise_info", CKKSNoiseInfo.unbounded())
        # Slot overflow needs the active scale, but reconstruction can preserve
        # diagnostic values without a context.
        if self._values is not None and current is not None:
            max_abs = encoding.max_abs(self._values, prec=current._slot_prec)
            if max_abs * current.params.scale(self.ring) > self.ring.q // 2:
                raise OverflowError("Ciphertext overflow is occurring")

    def rerandomize(self, rng: ChaChaRng | None = None) -> Ciphertext:
        """Add a fresh encryption of zero at this ciphertext's level.

        Preserve any tracked slot values.
        """
        return (self + encrypt(Plaintext.zero(level=self.level), rng=rng)).replace(_values=self._values)

    @classmethod
    def _from_rlwe(cls, ct: RLWECiphertext, *, msg_bound: float | None = None) -> Self:
        noise_info = ct.noise_info if msg_bound is None else replace(ct.noise_info, msg_bound=msg_bound)
        return cls(a=ct.a, b=ct.b, noise_info=noise_info)

    @classmethod
    def const(cls, val: Scalar, *, msg_bound: float | None = None, level: int | None = None) -> Self:
        """Return a trivial encryption of a constant, with public msg_bound defaulting to abs(val)."""
        pt = Plaintext.const(val, msg_bound=msg_bound, level=level)
        return cls(a=Poly.zero(pt.ring), b=pt.m, noise_info=pt.noise_info, _values=pt._values)

    @staticmethod
    def sum(cts: Sequence[Ciphertext]) -> Ciphertext:
        """Sum a sequence of ciphertexts, possibly at mixed levels.

        See :func:`ren.ckks.arithmetic_ops.sum_ciphertexts`.
        """
        from ren.ckks.arithmetic_ops import sum_ciphertexts

        return sum_ciphertexts(cts)

    @staticmethod
    def multiply(cts: list[Ciphertext]) -> Ciphertext:
        """Compute the product of multiple ciphertexts.

        See :func:`ren.ckks.arithmetic_ops.multiply_ciphertexts`.
        """
        from ren.ckks.arithmetic_ops import multiply_ciphertexts

        return multiply_ciphertexts(cts)

    @staticmethod
    def dot(
        left: Sequence[RawSlots | Plaintext | Ciphertext],
        right: Sequence[RawSlots | Plaintext | Ciphertext],
    ) -> Ciphertext:
        """Heterogeneous dot product ``sum_i left[i] * right[i]``.

        See :func:`ren.ckks.arithmetic_ops.dot`.
        """
        from ren.ckks.arithmetic_ops import dot

        return dot(left, right)

    def bootstrap(self, log_bound: int) -> Ciphertext:
        """Refresh this ciphertext, assuming ``abs(message) <= 2**log_bound``."""
        from ren.ckks.bootstrap import bootstrap

        return bootstrap(self, log_bound=log_bound)

    def __getstate__(self):
        # Tracked values are non-essential debug metadata and contain
        # non-picklable python-flint ``acb`` objects.
        return {"a": self.a, "b": self.b, "noise_info": self.noise_info}

    def __setstate__(self, state):
        object.__setattr__(self, "a", state["a"])
        object.__setattr__(self, "b", state["b"])
        object.__setattr__(self, "noise_info", state["noise_info"])
        object.__setattr__(self, "_values", None)

    def _multiply_small_polynomial_coeffs(self, other: Poly, *, norm_bound: FloatLike) -> Ciphertext:
        """
        Multiply this ciphertext in coefficient space by a small polynomial with a norm bound provided in coefficient space.
        The ciphertext's message must be in coefficient space.
        """
        return super()._multiply_small_polynomial_coeffs(other, norm_bound=norm_bound).replace(_values=None)

    def _prepare(self, other: object) -> tuple[Ciphertext, Ciphertext] | None:
        if not isinstance(other, Ciphertext | Plaintext):
            return None
        aligned, operand = self._align(other)
        if isinstance(operand, Plaintext):
            operand = operand.to_ciphertext()
        return aligned, operand

    @count_calls
    def __mul__(self, other: Ciphertext | Plaintext | Scalar) -> Ciphertext:
        if isinstance(other, int | np.integer):
            scalar = int(other)
            return self.map_polys(lambda poly: poly * scalar).replace(
                noise_info=self.noise_info * scalar,
                _values=self._map_values(lambda values: values * scalar),
            )
        operand = self._coerce(other)
        if not isinstance(operand, Ciphertext | Plaintext):
            return NotImplemented
        self, right = self._align(operand)
        if self.level == 0:
            raise InvalidParameterError("cannot multiply ciphertext at level 0")
        ctx = context()
        target_ring = ctx.params.ring_at(self.level - 1)
        noise_info = self.noise_info * right.noise_info

        if isinstance(right, Ciphertext):
            rlk = ctx.keyset.rlk
            if rlk is None:
                raise RuntimeError("no relinearization key available")

            c0 = self.b * right.b
            c2 = self.a * right.a
            c1 = self.b * right.a + self.a * right.b  # naive: c1 = a0 * b1 + a1 * b0

            linear_ct_part = Ciphertext(a=c1, b=c0, noise_info=noise_info)
            linearized_deg_2_ct_part = Ciphertext._from_rlwe(_keyswitch(c2, rlk), msg_bound=0)

            # Add relinearization correction at this level, then rescale once.
            unscaled_product = linear_ct_part + linearized_deg_2_ct_part
        else:
            unscaled_product = self.map_polys(lambda poly: poly * right.m).replace(noise_info=noise_info, _values=None)

        # Rescale: drop one scheduled level with proper rounding (pure RNS).
        output = unscaled_product._rescale(target_ring.moduli)
        return self.replace(a=output.a, b=output.b, noise_info=output.noise_info, _values=self._map_values(operator.mul, right))

    @count_calls
    def __rmul__(self, other: Ciphertext | Plaintext | Scalar) -> Ciphertext:
        return self.__mul__(other)

    def __truediv__(self, other: complex) -> Ciphertext:
        return self * (1 / other)

    def __pow__(self, exponent: int) -> Ciphertext:
        exponent = operator.index(exponent)
        if exponent < 0:
            raise ValueError("Exponent must be nonnegative")
        if exponent == 0:
            return Ciphertext.one(level=self.level)
        if exponent == 1:
            return self
        half = self ** (exponent // 2)
        return half * half * self if exponent % 2 else half * half

    def _mod_raise(self, *, secret_l1_norm_bound: float) -> Ciphertext:
        ring = context().params.highest_ring
        if ring == self.ring:
            return self
        # b + a*s uses lifted coefficients. Near q/2 CUDA can round to the adjacent
        # representative, so bound each coefficient by q.
        # TODO: Convert the coefficient bound to a slot bound before storing it in msg_bound.
        noise_info = replace(self.noise_info, msg_bound=(1 + secret_l1_norm_bound) * self.ring.q)
        return self.map_polys(lambda poly: poly.lift_to_ring(ring)).replace(noise_info=noise_info)

    def _rotate_by_galois_elt(self, galois_elt: int, *, hoisted: HoistedDigits | None = None) -> Ciphertext:
        """
        Helper function for rotation and conjugation.

        When ``hoisted`` is provided it must be ``_decompose_and_modraise(self.a, ...)``
        for this same ciphertext: the key-switch reuses those pre-decomposed digits
        (the hoisted path) instead of re-decomposing ``self.a``. The result is
        bit-identical to the non-hoisted path; only the shared mod-raise is skipped.
        """

        ctx = context()
        gk = ctx.keyset.get_galois_key(galois_elt)

        # Permute-last rotation flow (see ``CKKSKeySet.generate``):
        # 1. Key-switch the unpermuted self.a against a key that encrypts s under
        #    σ_{g⁻¹}(s) and fold in self.b. With ks = KS(self.a):
        #    ks.a·σ_{g⁻¹}(s) + ks.b + self.b ≈ self.a·s + self.b.
        #    This switches the secret s -> σ_{g⁻¹}(s), so `_switch_secret` does it
        #    (threading the hoisted digits through unchanged).
        # 2. Apply σ_g to the combined ciphertext. σ_g maps σ_{g⁻¹}(s) back to s,
        #    so σ_g(self.a·s + self.b) ≈ σ_g(m) decrypts under s.
        #
        # Deferring σ to the output lets the decompose+mod-raise of self.a be hoisted
        # across many rotations (the un-permuted digits are shared).
        combined = _switch_secret(self, gk, hoisted=hoisted)
        return combined.map_polys(lambda poly: poly.automorphism(galois_elt))

    def _rotate_by_slot_counts(self, slot_counts: list[int], max_ks: int | None = None) -> list[Ciphertext]:
        """
        Rotate by multiple slot counts efficiently by reusing intermediate rotations.

        Uses BFS to find the minimum number of base rotations needed to reach each
        target from the set of already-computed rotations. Intermediate rotations
        are added to the reachable set to potentially help reach other targets.
        """
        ctx = context()

        n = self.ring.n
        half_n = n // 2
        galois_modulus = 2 * n
        base_galois_elts = list(ctx.keyset._rotation_galois_elts_at(self.level))

        normalized_slot_counts = [slot_count % half_n for slot_count in slot_counts]
        galois_elts_by_slot_count = {slot_count: steps_to_galois_elt(slot_count, n) for slot_count in normalized_slot_counts}
        targets = set(galois_elts_by_slot_count.values())
        targets.discard(1)

        total_rotations = 0

        rotated_ciphertexts: dict[int, Ciphertext] = {1: self}

        # Hoist phase: any target we hold a *direct* key for is reached in one hop,
        # so all such targets share a single decompose+mod-raise of `self.a`.
        # Reaching each target is bit-identical to the BFS path; we just skip the
        # repeated base conversion.
        # Hoisted results seed `rotated_ciphertexts`, so the BFS below can also
        # compose from them when reaching the non-direct targets.
        hoistable = targets & set(base_galois_elts)
        if len(hoistable) >= 2:
            hoisted = _decompose_and_modraise(self.a, ctx.keyset.get_galois_key(next(iter(hoistable))))
            for galois_elt in hoistable:
                rotated_ciphertexts[galois_elt] = self._rotate_by_galois_elt(galois_elt, hoisted=hoisted)
                total_rotations += 1
            targets -= hoistable

        while targets:
            # BFS to find the closest target from any reachable state
            # visited[g] = (source_g, path) where path is a list of Galois elements to apply
            visited: dict[int, tuple[int, list[int]]] = {}
            for galois_elt in rotated_ciphertexts:
                visited[galois_elt] = (galois_elt, [])

            queue = list(rotated_ciphertexts.keys())
            found_target: int | None = None
            found_source: int | None = None
            found_path: list[int] | None = None

            while queue and found_target is None:
                next_queue: list[int] = []
                for current_galois_elt in queue:
                    source, path = visited[current_galois_elt]
                    for base_galois_elt in base_galois_elts:
                        new_galois_elt = (current_galois_elt * base_galois_elt) % galois_modulus
                        if new_galois_elt not in visited:
                            new_path = path + [base_galois_elt]
                            visited[new_galois_elt] = (source, new_path)
                            if new_galois_elt in targets:
                                found_target = new_galois_elt
                                found_source = source
                                found_path = new_path
                                break
                            next_queue.append(new_galois_elt)
                    if found_target is not None:
                        break
                queue = next_queue

            if found_target is None:
                raise RuntimeError(f"Could not reach slot rotations {slot_counts} with Galois keys {base_galois_elts}")

            assert found_source is not None and found_path is not None

            current_ct = rotated_ciphertexts[found_source]
            current_galois_elt = found_source
            for base_galois_elt in found_path:
                next_ct = current_ct._rotate_by_galois_elt(base_galois_elt)
                total_rotations += 1

                current_galois_elt = (current_galois_elt * base_galois_elt) % galois_modulus
                current_ct = next_ct

                if current_galois_elt not in rotated_ciphertexts:
                    rotated_ciphertexts[current_galois_elt] = current_ct

            targets.remove(found_target)

        if max_ks is not None and total_rotations > max_ks:
            raise RuntimeError(f"Total rotations ({total_rotations}) exceeded max_ks ({max_ks}) when rotating by {slot_counts}")

        if self._values is not None:
            for slot_count, galois_elt in galois_elts_by_slot_count.items():
                ct = rotated_ciphertexts[galois_elt]
                rotated_ciphertexts[galois_elt] = ct.replace(_values=self._map_values(lambda values: np.roll(values, -slot_count)))

        return [rotated_ciphertexts[galois_elts_by_slot_count[slot_count]] for slot_count in normalized_slot_counts]

    @overload
    def rotate(self, step: int, *, max_ks: int | None = None) -> Ciphertext: ...

    @overload
    def rotate(self, step: list[int], *, max_ks: int | None = None) -> list[Ciphertext]: ...

    @count_calls
    def rotate(self, step: int | list[int], *, max_ks: int | None = None) -> Ciphertext | list[Ciphertext]:
        """
        Rotate ciphertext by the given number of steps, or by a list of step amounts.

        Positive steps rotate left; negative steps rotate right.
        For each step, a value at slot i moves to slot (i - step) % slot_count.
        """
        if isinstance(step, int):
            return self._rotate_by_slot_counts([step], max_ks=max_ks)[0]
        else:
            return self._rotate_by_slot_counts(step, max_ks=max_ks)

    @count_calls
    def conjugate(self) -> Ciphertext:
        """
        Returns a ciphertext encrypting the complex conjugate of what an input
        ciphertext was encrypting.

        The implementation is very similar to the rotation method implementation.
        """
        result = self._rotate_by_galois_elt(-1)

        return result.replace(_values=self._map_values(np.frompyfunc(lambda value: value.conjugate(exact=True), 1, 1)))

    @count_calls
    def times_i(self) -> Ciphertext:
        """
        Multiplies a ciphertext by i.
        """
        k = self.ring.n // 2
        return self.map_polys(lambda poly: poly.rotate(k)).replace(_values=self._map_values(lambda values: values * 1j))

    def clone(self, device: str | None = None) -> Self:
        """Return a value-equivalent ciphertext backed by fresh polynomial storage."""
        return self.map_polys(lambda poly: poly.clone(device))
