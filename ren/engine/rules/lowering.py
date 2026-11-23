from ren.engine.pattern import RuleSet, pat, rules
from ren.ir import GroupOp, Op
from ren.ir.node import Node
from ren.ring import Domain
from ren.dtype import dtypes
from ren.tables import Tables


def _lower_rotate(n: Node, x: Node, device: str) -> Node:
    """Expand ROTATE to multiply by rotation factors."""
    ring = n.ring
    assert ring is not None, "lower_rotate expects ring"
    rf_buf = Tables(device).ring(ring).get_rotation_factors(n.arg)
    rf = Node.from_buffer(rf_buf).as_ring(ring, Domain.NTT)
    return x * rf


def _lower_automorphism_store(n: Node, target: Node, auto: Node, x: Node) -> Node | None:
    """Lower AUTOMORPHISM STORE to scatter write.

    Replaces STORE(buf, AUTOMORPHISM(src, k)) with a scatter write to
    INDEX(out_buf, computed_dst), reading the input at GIDX.

    COEFF domain: scatter index is (i*k) mod n with conditional negation when
    floor(i*k / n) is odd (from reduction mod X^n + 1).

    NTT domain: pure permutation using the inverse Galois element k_inv = k^{-1}
    mod 2n and bit-reversed NTT index mapping.
    """
    ring = auto.ring
    assert ring is not None, "AUTOMORPHISM must carry ring metadata"
    nn = ring.n
    domain = auto.domain

    out_buf = target
    idx = Node(Op.GIDX, dtypes.index)
    if x.base.op not in GroupOp.Elementwise:
        x = x.index(idx).load()
    i = idx & (nn - 1)
    p_base = idx - i

    if domain == Domain.COEFF:
        k = auto.arg
        index_raw = i * k
        new_idx = index_raw & (nn - 1)
        dst_idx = p_base + new_idx

        negate_flag = (index_raw // nn) & 1
        result = negate_flag.eq(1).where(-x, x)
        return n.replace(src=(out_buf.index(dst_idx), result.as_ring(ring, Domain.COEFF)))

    if domain == Domain.NTT:
        k_inv = pow(auto.arg, -1, 2 * nn)
        n_power = nn.bit_length() - 1

        i_rev = Node(Op.BITREVERSE, dtypes.index, src=(i,), arg=n_power)
        i_pow = i_rev * 2 + 1
        new_pow = (i_pow * k_inv) & (2 * nn - 1)
        half_pow = new_pow // 2
        dst_local = Node(Op.BITREVERSE, dtypes.index, src=(half_pow,), arg=n_power)
        dst_idx = p_base + dst_local
        return n.replace(src=(out_buf.index(dst_idx), x.as_ring(ring, Domain.NTT)))

    return None


def lower_ops(device: str) -> RuleSet:
    """Place buffer requests and rotation tables on the destination device.
    In NTT form, ROTATE(x, k) becomes x * rotation_factors[k]. AUTOMORPHISM writes to permuted indices.
    """
    return rules([
        (pat.b(Op.BUFFERIZE, pat.x), lambda b, x: x.bufferize(device) if b._device is None else None),
        (pat.n(Op.ROTATE, pat.x), lambda n, x: _lower_rotate(n, x, device)),
        (pat.n(Op.STORE, (pat.target, pat.auto(Op.AUTOMORPHISM, pat.x))), _lower_automorphism_store),
    ])


# KernelInfo carries the modulus once polynomial lowering has finished.
strip_ring = rules([
    (pat(Op.AS_RING, pat.x), lambda x: x),
    (pat.n(Op.MOD_REDUCE), lambda n: n.replace(arg=None)),
])
