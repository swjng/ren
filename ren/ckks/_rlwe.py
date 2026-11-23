from collections.abc import Callable
from dataclasses import dataclass, fields
from functools import partial
from math import pi, sqrt
from operator import add
from typing import Any, Self

from ren.bigfloat import FloatLike
from ren.device import Device
from ren.container import PolyContainer
from ren.poly import Poly
from ren.polydist import PolyDistWithBounds
from ren.random import ChaChaRng
from ren.ring import RingSpec
from ren.ckks.errors import ParameterMismatchError

from .metadata import CKKSNoiseInfo


def _fresh_encryption_noise(n: int, error_dist: PolyDistWithBounds) -> CKKSNoiseInfo:
    bound = error_dist.coeff_inf_norm_bound()
    return CKKSNoiseInfo(
        noise_bound=2 * n * bound / pi,
        noise_heuristic_bound=sqrt(2 * n / 3) * bound,
        msg_bound=0,
    )


def _rebuild(cls: type, names: tuple[str, ...], carried: dict[str, Any], *values: Any) -> Any:
    result = object.__new__(cls)
    for name, value in (*zip(names, values, strict=True), *carried.items()):
        object.__setattr__(result, name, value)
    return result


def _reduce_keeping(value: Any, kept: tuple[str, ...]) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """Tree reduction over dataclass fields that carries the kept fields through rebuilds."""
    names = tuple(field.name for field in fields(value) if field.name not in kept)
    carried = {name: getattr(value, name) for name in kept}
    return partial(_rebuild, type(value), names, carried), tuple(getattr(value, name) for name in names)


@dataclass(slots=True, frozen=True)
class RLWECiphertext(PolyContainer):
    """Two-polynomial RLWE encryption of a ring element."""

    a: Poly
    b: Poly
    noise_info: CKKSNoiseInfo

    def __post_init__(self) -> None:
        if self.a.ring != self.b.ring:
            raise ParameterMismatchError("polynomial ring mismatch between RLWE ciphertext components")
        if self.a.device is not None and self.b.device is not None and self.a.device != self.b.device:
            raise ParameterMismatchError("polynomial device mismatch between RLWE ciphertext components")

    @property
    def ring(self) -> RingSpec:
        return self.a.ring

    @property
    def device(self) -> str | None:
        return self.a.device or self.b.device

    def __reduce__(self):
        self.realize()
        return super().__reduce__()

    def __tree_reduce__(self):
        # Noise bounds stay out of tree traversal and jit plan keys.
        return _reduce_keeping(self, ("noise_info",))

    def __add__(self, other) -> Self:
        if self.ring != other.ring:
            raise ParameterMismatchError("RLWE ciphertext ring mismatch")
        return self.replace(polys=map(add, self.polys(), other.polys(), strict=True), noise_info=self.noise_info + other.noise_info)

    def _multiply_small_polynomial_coeffs(self, other: Poly, *, norm_bound: FloatLike) -> Self:
        return self.map_polys(lambda poly: poly * other).replace(noise_info=(norm_bound * self.ring.n) * self.noise_info)

    def _rescale(self, target_moduli: tuple[int, ...]) -> Self:
        """Rescale the polynomial basis and noise."""
        noise_info = self.noise_info._rescale(self.ring.moduli, target_moduli)
        return self.map_polys(lambda poly: poly.rescale(target_moduli)).replace(noise_info=noise_info)

    @staticmethod
    def _encrypt(
        m: Poly,
        *,
        secret: Poly,
        error_dist: PolyDistWithBounds,
        output_noise_info: CKKSNoiseInfo | None = None,
        mask: Poly | None = None,
        rng: ChaChaRng | None = None,
    ) -> RLWECiphertext:
        """Encrypt a polynomial under a small secret without an active context.

        Default to fresh-error bounds with zero message bound. Store an explicit
        ``output_noise_info`` unchanged, including any payload bounds supplied by
        the caller. Use ``mask`` when participants must share the same mask.
        """
        device = m.device or (mask.device if mask is not None else None) or secret.device
        if mask is not None and (mask.ring != m.ring or (mask.device is not None and mask.device != device)):
            raise ParameterMismatchError("encryption mask must match the message ring and device")
        if rng is not None and device is not None and rng.device != device:
            raise ParameterMismatchError(f"encryption device is {device}, got RNG on {rng.device}")
        ring = m.ring
        if output_noise_info is None:
            output_noise_info = _fresh_encryption_noise(ring.n, error_dist)
        s = secret._to_ring_small_coeffs(ring)
        if rng is None:
            rng = Device[Device.canonicalize(device)].default_rng()
        e = Poly.rand(error_dist, ring, rng=rng)
        a = Poly.uniform(ring, rng=rng) if mask is None else mask
        return RLWECiphertext(a=a, b=-(a * s) + e + m, noise_info=output_noise_info)

    def _decrypt(self, secret: Poly) -> Poly:
        """Decrypt to a polynomial.

        Basis extension requires a secret with small coefficients.
        """
        s = secret._to_ring_small_coeffs(self.ring)
        return self.a * s + self.b
