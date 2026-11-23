from ren.engine.codegen import lower_kernel
import operator

import pytest

from ren import function, jit, realize
from ren.backends.python import PythonBackend, SUPPORTED_DTYPES
from ren.buffer import Buffer, BufferSpec
from ren.dtype import dtypes
from ren.engine.realize import make_exec_units, run_exec_units, call_buffers, call_op
from ren.poly import Poly
from ren.ir import Op, spec
from ren.ir.node import Node, ParamArg, RandomArg, ToRingArg, ToRnsArg
from ren.polydist import Ternary
from ren.ring import Domain, RingSpec
from ren.kernel import KernelInfo, KernelSpec, get_output_slots
from ren.engine.schedule import schedule
from ren.engine.pattern import rewrite
from ren.engine.rules import analyze_ring_lowering, indexify, lower_ring
from tests.utils import large_prime_ring


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda a: a % a, "uint128"),
        (lambda a: a >> a, "constant shift"),
        (lambda a: a.after(a), "uint128"),
        (lambda a: (Node.const(1 << 64) * 3).cast(dtypes.uint128), "uint128"),
    ],
)
def test_lower_wide_rejects_unsupported_ops(build, message):
    a = Node(Op.PARAM, dtypes.uint64, arg=ParamArg(1, 4, "python")).index(Node(Op.GIDX, dtypes.index)).load().cast(dtypes.uint128)
    out = Node(Op.PARAM, dtypes.uint64, arg=ParamArg(0, 4, "python")).index(Node(Op.GIDX, dtypes.index))
    info = KernelInfo("python", large_prime_ring(4)[:1], Domain.NTT)
    with pytest.raises(NotImplementedError, match=message):
        PythonBackend().make_kernel(Node.sink(out.store(build(a).cast(dtypes.uint64))).replace(arg=info))


def test_mul_inserts_rns_ntt_pipeline_python():
    ring = large_prime_ring(4)
    p1 = Poly([1, 2, 3, 4], ring, device="python")
    p2 = Poly([4, 3, 2, 1], ring, device="python")

    r = p1 * p2
    root = Node.sink(r.intt().node)
    linear, _ = schedule(root)

    ops: set[Op] = set()
    for call in linear.src:
        for n in call.src[0].toposort():
            ops.add(n.op)

    # Poly creation ensures coefficients are in canonical RNS already; planner
    # should insert NTT/INTT around ring elementwise chains, but should not
    # introduce FROM_RNS in this coeff-only view.
    assert Op.NTT in ops
    assert Op.INTT in ops
    assert Op.FROM_RNS not in ops


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize("domain", [Domain.COEFF, Domain.NTT])
@pytest.mark.parametrize("factor", [2, -3])
def test_scalar_mul_preserves_domain(dtype, domain, factor):
    ring = RingSpec(n=4, moduli=(17, 41, 73), dtype=dtype)
    x = Poly([1, 2, 3, 4], ring)
    x = (x.ntt() if domain is Domain.NTT else x).realize()
    result = x * factor

    linear, _ = schedule(Node.sink(result.node))

    assert all(call_op(call) not in {Op.NTT, Op.INTT} for call in linear.src)
    assert result.realize().domain is domain
    assert result.tolist() == [v * factor % ring.q for v in (1, 2, 3, 4)]


@pytest.mark.parametrize("conversion", [Poly.intt, Poly.ntt], ids=["intt", "ntt"])
@pytest.mark.parametrize("op", [operator.add, operator.sub], ids=["add", "sub"])
def test_scalar_arithmetic_cancels_transform_roundtrip(conversion, op):
    ring = RingSpec(n=4, moduli=(97,), dtype=dtypes.uint64)
    x = Poly([1, 2, 3, 4], ring).realize()
    y = Poly([4, 3, 2, 1], ring).realize()
    if conversion is Poly.ntt:
        x, y = x.ntt().realize(), y.ntt().realize()
    inverse = Poly.ntt if conversion is Poly.intt else Poly.intt
    result = conversion(op(inverse(x), inverse(y) * 3) * 7)

    linear, _ = schedule(Node.sink(result.node))

    assert result.tolist() == [op(a, b * 3) * 7 % ring.q for a, b in zip([1, 2, 3, 4], [4, 3, 2, 1])]
    assert all(call_op(call) not in {Op.NTT, Op.INTT} for call in linear.src)


@pytest.mark.parametrize("conversion", [Poly.intt, Poly.ntt], ids=["intt", "ntt"])
@pytest.mark.parametrize("op", [operator.add, operator.sub], ids=["add", "sub"])
def test_transform_of_sum_without_cancellation_converts_once(conversion, op):
    ring = RingSpec(n=4, moduli=(97,), dtype=dtypes.uint64)
    x, y = [Poly(values, ring).realize() for values in ([1, 2, 3, 4], [4, 3, 2, 1])]
    if conversion is Poly.intt:
        x, y = x.ntt().realize(), y.ntt().realize()
    result = conversion(op(x, y))

    linear, _ = schedule(Node.sink(result.node))

    assert sum(call_op(call) in {Op.NTT, Op.INTT} for call in linear.src) == 1
    assert result.tolist() == [op(a, b) % ring.q for a, b in zip([1, 2, 3, 4], [4, 3, 2, 1])]


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize("decorate", [lambda fn: fn, function], ids=["eager", "function"])
@pytest.mark.parametrize("conversion", [Poly.intt, Poly.ntt], ids=["intt", "ntt"])
def test_sum_after_domain_simplification(dtype, decorate, conversion):
    ring = RingSpec(n=4, moduli=(17, 41, 73), dtype=dtype)
    x = Poly([1, 2, 3, 4], ring)

    @decorate
    def double(x):
        return x + conversion(x)

    assert double(x).tolist() == [2, 4, 6, 8]


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize("product_first", [False, True], ids=["add-first", "mul-first"])
def test_scalar_mul_keeps_shared_constant_polynomial(dtype, product_first):
    ring = RingSpec(n=4, moduli=(17, 41, 73), dtype=dtype)
    x = Poly([1, 2, 3, 4], ring)
    constant = Poly.const(2, ring)
    product, summed = x * constant, x + constant

    outputs = (product, summed) if product_first else (summed, product)
    realize(*outputs, constant)

    assert product.tolist() == [2, 4, 6, 8]
    assert summed.tolist() == [3, 2, 3, 4]
    assert constant.tolist() == [2, 0, 0, 0]


@pytest.mark.parametrize("decorate", [lambda fn: fn, function], ids=["eager", "function"])
def test_contiguous_output_does_not_add_kernels(decorate):
    ring = large_prime_ring(4)
    x = Poly([1, 2, 3, 4], ring, device="python").realize()
    y = Poly([4, 3, 2, 1], ring, device="python").realize()

    @decorate
    def add(x, y):
        return (x + y).contiguous()

    value = x + y
    plain, _ = schedule(Node.sink(value.node))
    result = add(x, y)
    contiguous, _ = schedule(Node.sink(result.node))

    assert [call_op(call) for call in contiguous.src] == [call_op(call) for call in plain.src]
    assert result.tolist() == [5, 5, 5, 5]


def test_domain_injection_converts_fan_in_output():
    ring = large_prime_ring(4)
    x = Poly([1, 2, 3, 4], ring, device="python")
    y = Poly([4, 3, 2, 1], ring, device="python")
    z = Poly([1, 1, 1, 1], ring, device="python")

    linear, _ = schedule(Node.sink(((x + y) * z).node))

    assert sum(call_op(call) is Op.NTT for call in linear.src) == 2


@pytest.mark.parametrize("op", [operator.add, operator.sub], ids=["add", "sub"])
def test_intt_of_mixed_domain_sum_converts_once(op):
    ring = large_prime_ring(4)
    x = Poly([1, 2, 3, 4], ring, device="python")
    y = Poly([4, 3, 2, 1], ring, device="python").ntt().realize()
    out = op(x, y).intt()

    linear, _ = schedule(Node.sink(out.node))

    assert sum(call_op(call) is Op.NTT for call in linear.src) == 0
    assert out.tolist() == [op(a, b) % ring.q for a, b in zip([1, 2, 3, 4], [4, 3, 2, 1])]


@pytest.mark.parametrize("domain", [Domain.COEFF, Domain.NTT])
def test_automorphism_computes_elementwise_source(domain):
    ring = large_prime_ring(4)
    x = Poly([1, 2, 3, 4], ring, device="python")
    y = Poly([4, 3, 2, 1], ring, device="python")
    if domain is Domain.NTT:
        x, y = x.ntt(), y.ntt()
    x, y = x.realize(), y.realize()
    out = (x + y * 3).automorphism(3)

    linear, _ = schedule(Node.sink(out.node))

    assert len(linear.src) == 1
    assert out.tolist() == (x + y * 3).realize().automorphism(3).tolist()


def test_schedule_writes_conversion_output_directly():
    ring = large_prime_ring(4)
    source = Node.new_buffer("python", ring.storage_size, ring.dtype).as_ring(ring, Domain.COEFF)

    linear, _ = schedule(Node.sink(Poly(source).ntt().node))

    assert [call_op(call) for call in linear.src] == [Op.NTT]


def test_schedule_accepts_large_integer_constants():
    ring = large_prime_ring(4)
    big = 1 << 15839  # Exceeds Python's default integer-to-string digit limit.
    x = Poly([1, 2, 3, 4], ring)
    assert (x * big).tolist() == [v * big % ring.q for v in (1, 2, 3, 4)]


@pytest.mark.parametrize("dtype", [dtypes.uint32, dtypes.uint64])
@pytest.mark.parametrize("source", ["constant", "dependency", "annotated_dependency"])
def test_constant_output_preserves_materialized_consumer(dtype, source):
    ring = RingSpec(n=4, moduli=(17, 41), dtype=dtype)
    target = Poly([1, 2, 3, 4], ring).realize()
    constant = Poly.const(7, ring)
    write = target.node.store(Poly.const(9, ring).node)
    output = Poly(constant.node.after(write))
    value = constant.node if source == "constant" else constant.clone(target.device).realize().node.after(write)
    if source == "annotated_dependency":
        value = value.as_ring(ring, Domain.COEFF)
    coefficients = Poly(value.from_rns()).clone(device="python")

    realize(output, coefficients)

    assert target.tolist() == ([1, 2, 3, 4] if source == "constant" else [9, 0, 0, 0])
    assert output.tolist() == [7, 0, 0, 0]
    assert list(coefficients.node.base.buffer.as_typed_buffer()) == [7, 0, 0, 0]


def test_deferred_materialization_reuses_requested_output():
    ring = RingSpec(n=4, moduli=(17, 97))
    x = Poly([1, 2, 3, 4], ring).realize()
    rotated = Poly.const(7, ring).ntt().rotate(1)
    output = rotated.clone(x.device)
    consumer = rotated.intt() + x

    linear, outputs = schedule(Node.sink(output.node, consumer.node))
    assert len(linear.src) == 3
    intt = next(call for call in linear.src if call_op(call) is Op.INTT)
    assert intt.inputs[1].base is outputs[output.node].base
    realize(output, consumer)
    assert output.tolist() == [0, 7, 0, 0]
    assert consumer.tolist() == [1, 9, 3, 4]


@pytest.mark.parametrize("kind", ["computed", "constant", "view"])
def test_after_value_and_effect_are_preserved_across_outputs(kind):
    ring = RingSpec(n=4, moduli=(17, 41, 97))
    x = Poly([1, 2, 3, 4], ring).realize()
    side = Poly([8, 7, 6, 5], ring).realize()
    if kind == "computed":
        source, expected = x + 2, [3, 2, 3, 4]
    elif kind == "constant":
        source, expected = Poly.const(7, ring).clone(side.device), [7, 0, 0, 0]
    else:
        source, expected = x.to_quotient_ring(ring[1:]), [1, 2, 3, 4]
    effect = side.node.store((x + 8).node)
    value = Poly(source.node.after(effect))
    outputs = [value + offset for offset in (3, 4)]

    realize(*outputs)

    for output, offset in zip(outputs, (3, 4), strict=True):
        assert output.tolist() == [expected[0] + offset, *expected[1:]]
    assert side.tolist() == [9, 2, 3, 4]


def test_two_reads_of_one_buffer_keep_both_writes():
    ring = large_prime_ring(4)
    x, a, b = [Poly([i, 0, 0, 0], ring).realize() for i in (1, 2, 3)]
    lhs = Poly(x.node.after(a.node.store(Poly.const(5, ring).node)))
    rhs = Poly(x.node.after(b.node.store(Poly.const(6, ring).node)))
    assert (lhs + rhs + a + b).realize().tolist() == [13, 0, 0, 0]


def test_function_constant_argument_keeps_other_domain_inline():
    ring = RingSpec(n=4, moduli=(17, 97))
    x = Poly([1, 2, 3, 4], ring, device="python").realize()
    ntt = x.ntt().realize()
    output = function(lambda value: value)(ntt * 0 + 7)
    other = x + 7

    separate = [schedule(Node.sink(value.node))[0] for value in (output, other)]
    joint, _ = schedule(Node.sink(output.node, other.node))
    assert len(joint.src) <= sum(len(linear.src) for linear in separate)
    realize(output, other)
    assert output.tolist() == [7, 0, 0, 0]
    assert other.tolist() == [8, 2, 3, 4]


@pytest.mark.parametrize("decorate", [lambda fn: fn, function], ids=["eager", "function"])
def test_schedule_trace_records_cache_misses(schedule_cache, decorate):
    ring = large_prime_ring(4)
    p1 = Poly([1, 2, 3, 4], ring, device="python")
    p2 = Poly([4, 3, 2, 1], ring, device="python")
    realize(p1, p2)
    scale = 2
    pass_names: set[str | None] = set()

    def trace(pass_name, _from_node, _to_node):
        pass_names.add(pass_name)

    @decorate
    def multiply(x, y):
        return x * y * scale

    linear, becomes_map = schedule(Node.sink(multiply(p1, p2).node), trace=trace)

    assert linear.src
    assert becomes_map
    assert "bufferize" in pass_names
    assert "kernelize" in pass_names

    pass_names.clear()
    q1 = Poly([2, 3, 4, 5], ring, device="python")
    q2 = Poly([5, 4, 3, 2], ring, device="python")
    realize(q1, q2)
    schedule(Node.sink(multiply(q1, q2).node), trace=trace)

    assert "callify" in pass_names
    assert "bufferize" not in pass_names
    assert "kernelize" not in pass_names


def test_canonicalize_neg_plus_to_sub():
    """Test that NEG(a) + b gets canonicalized to b - a."""
    ring = large_prime_ring(4)
    a = Poly([1, 0, 0, 0], ring, device="python")
    b = Poly([2, 0, 0, 0], ring, device="python")

    expr = (-a) + b  # builds NEG(a) feeding an ADD, should become b - a
    root = Node.sink(expr.intt().node)
    linear, _ = schedule(root)

    has_sub = False
    has_neg = False
    for call in linear.src:
        for n in call.src[0].toposort():
            if n.op is Op.SUB:
                has_sub = True
            if n.op is Op.NEG:
                has_neg = True
    assert has_sub, "expected SUB after canonicalization"
    assert not has_neg, "NEG should have been rewritten to SUB"


def test_schedule_splits_at_copy_ir():
    ring = large_prime_ring(4)
    p = Poly([1, 2, 3, 4], ring, device="python")
    p2 = p.to("cuda:0")

    root = Node.sink(p2.intt().node)
    linear, _ = schedule(root)

    # There should be a COPY kernel targeting cuda:0 for this simple graph,
    # even though RNS conversions may run on the python device.
    copy_ops = [call for call in linear.src if call.src[0].op is Op.COPY]
    assert len(copy_ops) == 1
    assert copy_ops[0].src[0].device == "cuda:0"


def test_schedule_copy_and_add_on_cuda_ir():
    ring = large_prime_ring(4)
    # One poly starts on python and moves to cuda, the other is created directly on cuda.
    p = Poly([1, 2, 3, 4], ring, device="python")
    p_cuda = p.to("cuda:0")
    q_cuda = Poly([4, 3, 2, 1], ring, device="cuda:0")

    r = p_cuda + q_cuda
    root = Node.sink(r.intt().node)
    linear, _ = schedule(root)

    # Expect COPY kernels for cross-device transfers, followed by compute kernels.
    copy_ops = [call for call in linear.src if call.src[0].op is Op.COPY]
    assert len(copy_ops) >= 1
    assert all(call.src[0].device == "cuda:0" for call in copy_ops)

    # Compute calls have body=SINK(STORE(PARAM, computation), ...)
    # The ADD may be wrapped in MOD_REDUCE, so check recursively
    def contains_add(node):
        if node.op is Op.ADD:
            return True
        return any(contains_add(s) for s in node.src)

    def is_add_kernel(call):
        root = call.src[0]
        if root.op is not Op.SINK or not root.src:
            return False
        store = root.src[0]
        if store.op is not Op.STORE or len(store.src) < 2:
            return False
        return contains_add(store.src[1])

    add_kernels = [call for call in linear.src if is_add_kernel(call)]
    assert len(add_kernels) >= 1
    assert all(isinstance(call.src[0].arg, KernelInfo) and call.src[0].arg.device == "cuda:0" for call in add_kernels)


@pytest.mark.parametrize("input_domain", [Domain.COEFF, Domain.NTT])
def test_copy_preserves_domain_through_scheduling(input_domain):
    """Verify that copying a poly preserves its domain through scheduling."""
    ring = large_prime_ring(4)
    p = Poly([1, 2, 3, 4], ring, device="python")

    if input_domain == Domain.NTT:
        p = p.ntt()

    # Copy to another device
    p2 = p.to("cuda:0")
    assert p2.domain == input_domain

    # Schedule the copy operation
    root = Node.sink(p2.node)
    _, becomes_map = schedule(root)

    # The scheduled output should preserve the domain
    assert p2.node in becomes_map, "copied node should be in becomes_map"
    realized_node = becomes_map[p2.node]
    assert realized_node.domain == input_domain, f"expected {input_domain}, got {realized_node.domain}"


class TestToRingView:
    def test_to_ring_buffer_view(self, schedule_cache):
        # Create rings where smaller is a prefix of larger
        large_ring = large_prime_ring(4)
        small_ring = large_ring[:2]

        # Create poly in larger ring, convert to smaller
        p = Poly([1, 2, 3, 4], large_ring, device="python")
        p.realize()

        p_small = p.to_quotient_ring(small_ring)
        root = Node.sink(p_small.node)
        linear, outputs = schedule(root)

        # Viewing realized storage needs no executable call or new allocation.
        assert not linear.src
        view = outputs[p_small.node].base
        assert view.op is Op.BUFFER_VIEW
        assert view.arg == (small_ring.storage_size, 0)
        assert view.is_realized()
        assert view.buffer.base is p.node.base.buffer.base
        assert Poly(outputs[p_small.node]).tolist() == p.tolist()

        q = Poly([4, 3, 2, 1], large_ring, device="python").realize()
        q_small = q.to_quotient_ring(small_ring)
        replay, outputs = schedule(Node.sink(q_small.node))
        assert not replay.src
        assert outputs[q_small.node].base.buffer.base is q.node.base.buffer.base
        assert all(
            node.op is not Op.CALL or node.src[0].op is not Op.BUFFER_VIEW
            for linear, output_nodes in schedule_cache.values()
            for node in Node.sink(linear, *output_nodes).toposort(enter_calls=True)
        )

    @pytest.mark.parametrize("realized", [False, True], ids=["lazy", "realized"])
    @pytest.mark.parametrize("limbs", [slice(None, 1), slice(1, 3)], ids=["prefix", "offset"])
    def test_clone_of_view_uses_fresh_storage(self, realized, limbs):
        ring = RingSpec(n=8, moduli=(17, 97, 113))
        source = Poly(list(range(ring.n)), ring, device="python").ntt()
        if realized:
            source.realize()
        small_ring = ring[limbs]
        view = source.to_quotient_ring(small_ring)
        clone = view.clone().realize()

        assert clone.tolist() == view.tolist()
        assert clone.node.base.buffer.base.nbytes == small_ring.nbytes
        source.realize()
        assert clone.node.base.buffer.base is not source.node.base.buffer.base

    @pytest.mark.parametrize("consume", [lambda value: value, function(lambda value: value)], ids=["eager", "function"])
    def test_fused_view_followed_by_scaling(self, consume):
        ring = large_prime_ring(4, dtypes.uint32)
        x = Poly([1, 2, 3, 4], ring, device="python").ntt().realize()
        y = Poly([4, 3, 2, 1], ring, device="python").ntt().realize()
        result = consume((x + y).to_quotient_ring(ring[1:3]) * 3).realize()

        assert result.ring == ring[1:3]
        assert result.domain is Domain.NTT
        assert result.tolist() == [15, 15, 15, 15]

    def test_to_ring_fused_view(self):
        # Create rings where smaller is a prefix of larger
        large_ring = large_prime_ring(4)
        small_ring = large_ring[:2]

        # Create polys
        p_large = Poly([1, 2, 3, 4], large_ring, device="python")
        p_large.realize()

        p_small = Poly([4, 3, 2, 1], small_ring, device="python")
        p_small.realize()

        # TO_RING view followed by ADD - view should be fused into ADD kernel
        p_view = p_large.to_quotient_ring(small_ring)
        result = p_view + p_small

        root = Node.sink(result.intt().node)
        linear, _ = schedule(root)

        # Count kernel types
        to_ring_kernels = [call for call in linear.src if call_op(call) is Op.TO_RING]
        buffer_view_kernels = [call for call in linear.src if call.src[0].op is Op.BUFFER_VIEW]
        compute_kernels = [call for call in linear.src if call.src[0].op is Op.SINK]

        assert len(to_ring_kernels) == 0, "found independent to_ring kernel"

        # The TO_RING view should be fused into the ADD kernel, not a separate BUFFER_VIEW
        assert len(buffer_view_kernels) == 0, (
            f"Expected TO_RING view to be fused into elementwise kernel, but found {len(buffer_view_kernels)} BUFFER_VIEW kernel(s)"
        )

        # Verify we have compute kernels (for the ADD and any NTT/INTT)
        assert len(compute_kernels) >= 1, "Expected at least one compute kernel"

    def test_fused_realized_view_is_passed_as_input_subbuffer(self):
        """
        Regression: a realized quotient view feeding another compute kernel
        should become a sized input subbuffer, not a larger base buffer plus an
        in-kernel TO_RING.
        """
        large_ring = large_prime_ring(4)
        small_ring = large_ring[:2]
        small_size = small_ring.n * len(small_ring.moduli)

        p_large = Poly([1, 2, 3, 4], large_ring, device="python")
        p_large.realize()
        p_small = Poly([4, 3, 2, 1], small_ring, device="python")
        p_small.realize()

        result = p_large.to_quotient_ring(small_ring) + p_small
        linear, _ = schedule(Node.sink(result.node))

        compute_kernels = [call for call in linear.src if call.src[0].op is Op.SINK]
        assert compute_kernels, "Expected at least one compute kernel"
        final_kernel = compute_kernels[-1]

        assert all(buf.size == small_size for buf in call_buffers(final_kernel)), "expected fused kernel to consume sized view buffers"
        assert all(n.op is not Op.TO_RING for n in final_kernel.src[0].toposort()), (
            "expected view to become an input subbuffer, not an in-kernel TO_RING"
        )
        assert all(n.op is not Op.BUFFER_VIEW for n in final_kernel.src[0].toposort()), "BUFFER_VIEW should not remain inside compute kernel bodies"

    def test_to_ring_of_elementwise_chain_does_not_emit_buffer_view(self):
        """
        Regression case: (p + q).to_quotient_ring(view_ring) should stay in an elementwise
        compute kernel. A BUFFER_VIEW over ADD is invalid and crashes schedule.
        """
        large_ring = large_prime_ring(4)
        small_ring = large_ring[:2]

        p = Poly([1, 2, 3, 4], large_ring, device="python")
        q = Poly([4, 3, 2, 1], large_ring, device="python")
        result = (p + q).to_quotient_ring(small_ring)

        root = Node.sink(result.node)
        linear, _ = schedule(root)

        buffer_view_kernels = [call for call in linear.src if call.src[0].op is Op.BUFFER_VIEW]
        assert len(buffer_view_kernels) == 0, (
            f"Expected TO_RING view on elementwise chain to be indexified/fused, found {len(buffer_view_kernels)} BUFFER_VIEW kernel(s)"
        )

    def test_full_schedule_poly_pipeline_has_no_nested_buffer_view_in_conversion_chain(self):
        """
        In the full Poly scheduling pipeline (as used by Poly.realize), conversion
        chains should not leave BUFFER_VIEW nodes inside compute SINK kernels.
        """
        large_ring = large_prime_ring(8)
        small_ring = large_ring[1:3]

        p = Poly([1] * 8, large_ring, device="python")
        value = p.to_quotient_ring(small_ring).ntt()

        linear, _ = schedule(Node.sink(value.node))
        assert linear.src, "expected schedule to produce at least one call"

        sink_roots = [call.src[0] for call in linear.src if call.src[0].op is Op.SINK]
        assert sink_roots, "expected at least one SINK kernel root"
        for sink_root in sink_roots:
            assert all(n.op is not Op.BUFFER_VIEW for n in sink_root.toposort()), (
                "BUFFER_VIEW must not appear inside SINK kernel bodies; BUFFER_VIEW should be root-level materialization only"
            )

    def test_to_ring_then_ntt_runs_ntt_on_destination_ring(self):
        """The NTT only needs the primes kept by the quotient view."""
        large_ring = large_prime_ring(8)
        small_ring = large_ring[2:6]

        p = Poly([1] * large_ring.n, large_ring, device="python")
        p.realize()

        linear, _ = schedule(Node.sink(p.to_quotient_ring(small_ring).ntt().node))

        ntt_kernels = [call for call in linear.src if call_op(call) is Op.NTT]
        assert len(ntt_kernels) == 1, f"expected exactly one NTT kernel, found {len(ntt_kernels)}"

        ntt_info = ntt_kernels[0].src[0].arg
        assert isinstance(ntt_info, KernelInfo)
        assert ntt_info.ring == small_ring, "expected NTT to run on the destination ring, not the source ring"

    def test_fused_to_ring_view_after_realized_ntt_uses_offset_indexing(self):
        """
        Regression: when a quotient TO_RING view stays fused over a compute chain
        whose NTT inputs were already materialized, the final kernel must still
        index those inputs with the view offset instead of reading from plain GIDX.
        """
        large_ring = large_prime_ring(8)
        small_ring = large_ring[2:6]

        p1 = Poly([i + 1 for i in range(large_ring.n)], large_ring, device="python")
        p2 = Poly([2 * i + 3 for i in range(large_ring.n)], large_ring, device="python")
        p3 = Poly([5 * i + 7 for i in range(large_ring.n)], large_ring, device="python")
        p4 = Poly([11 * i + 13 for i in range(large_ring.n)], large_ring, device="python")

        result = (p1.ntt() - (p2.ntt() * p3.ntt()) + p4.ntt()).to_quotient_ring(small_ring)
        offset = result.node.view_offset
        assert offset is not None and offset > 0

        linear, _ = schedule(Node.sink(result.node))
        sink_roots = [call.src[0] for call in linear.src if call.src[0].op is Op.SINK]
        assert sink_roots, "expected at least one compute kernel"

        final_kernel = sink_roots[-1]
        has_offset_index = any(
            n.op is Op.INDEX
            and len(n.src) == 2
            and n.src[1].op is Op.ADD
            and any(s.op is Op.GIDX for s in n.src[1].src)
            and any(s.const_val == offset for s in n.src[1].src)
            for n in final_kernel.toposort()
        )
        assert has_offset_index, "expected fused quotient view to load realized inputs at GIDX + view_offset"

    def test_fused_to_ring_view_over_compute_reduces_before_store(self):
        """
        Regression: when a quotient TO_RING view stays fused over compute, the
        final STORE still needs a canonical reduced value.
        """
        large_ring = large_prime_ring(4)
        small_ring = large_ring[:2]

        p1 = Poly([i + 1 for i in range(large_ring.n)], large_ring, device="python")
        p2 = Poly([2 * i + 3 for i in range(large_ring.n)], large_ring, device="python")
        p3 = Poly([5 * i + 7 for i in range(large_ring.n)], large_ring, device="python")
        p4 = Poly([11 * i + 13 for i in range(large_ring.n)], large_ring, device="python")

        result = (p1.ntt() - (p2.ntt() * p3.ntt()) + p4.ntt()).to_quotient_ring(small_ring)
        linear, _ = schedule(Node.sink(result.node))

        final_kernel = [call.src[0] for call in linear.src if call.src[0].op is Op.SINK][-1]
        store = next(n for n in final_kernel.toposort() if n.op is Op.STORE)

        assert store.src[1].op is Op.MOD_REDUCE, "expected fused quotient view output to be reduced before STORE"
        assert all(n.ring is None for n in final_kernel.toposort())


@pytest.mark.parametrize("domain", [Domain.COEFF, Domain.NTT])
@pytest.mark.parametrize("decorate", [lambda fn: fn, function, jit], ids=["eager", "function", "jit"])
@pytest.mark.parametrize("materialized", [False, True], ids=["fused", "materialized"])
def test_shared_compute_source_can_feed_multiple_view_offsets(domain, decorate, materialized):
    """
    Regression: one computed source may feed multiple quotient views with
    different offsets in the same realization.
    """
    ring = RingSpec(n=8, moduli=(17, 97, 113, 193))
    xs, ys = [110 + i for i in range(ring.n)], [2 * i + 3 for i in range(ring.n)]
    x, y = Poly(xs, ring), Poly(ys, ring)
    if domain is Domain.NTT:
        x, y = x.ntt(), y.ntt()
    realize(x, y)

    @decorate
    def split(a, b):
        source = a + b
        if materialized:
            source = source.contiguous()
        return source.to_quotient_ring(ring[:2]), source.to_quotient_ring(ring[2:])

    left, right = split(x, y)
    realize(left, right)
    for result, expected_ring in ((left, ring[:2]), (right, ring[2:])):
        assert result.ring == expected_ring
        assert result.domain is domain
        assert result.tolist() == [a + b for a, b in zip(xs, ys, strict=True)]


@pytest.mark.parametrize("domain", [Domain.COEFF, Domain.NTT])
@pytest.mark.parametrize("decorate", [lambda fn: fn, function, jit], ids=["eager", "function", "jit"])
def test_nested_compute_views_use_relative_offsets(domain, decorate):
    ring = RingSpec(n=8, moduli=(17, 97, 113, 193))
    xs, ys, zs = [110 + i for i in range(ring.n)], [2 * i + 3 for i in range(ring.n)], [3 * i + 7 for i in range(ring.n)]
    x, y, z = Poly(xs, ring), Poly(ys, ring), Poly(zs, ring[1:])
    if domain is Domain.NTT:
        x, y, z = x.ntt(), y.ntt(), z.ntt()
    realize(x, y, z)

    @decorate
    def project(a, b, c):
        return ((a + b).to_quotient_ring(ring[1:]) + c).to_quotient_ring(ring[2:])

    result = project(x, y, z).realize()
    assert result.ring == ring[2:]
    assert result.domain is domain
    assert result.tolist() == [a + b + c for a, b, c in zip(xs, ys, zs, strict=True)]


def test_analyze_ring_lowering_tracks_growth_through_to_ring_views():
    """
    Regression: additive growth must propagate through TO_RING view nodes.

    Without this, overflow analysis underestimates ADD chains that repeatedly
    pass through view-level TO_RING, which can leave long unreduced ADD chains
    in MOD_REDUCE kernels.
    """
    ring = large_prime_ring(4)
    rings = [ring[:k] for k in range(len(ring.moduli), 9, -1)]  # 22 -> 10

    src = Node.const(7).as_ring(ring, Domain.COEFF)
    acc = src.to_ring(rings[1])
    for r in rings[2:]:
        # Force a view-level TO_RING boundary between additive steps.
        acc = acc.to_ring(r)
        term = src.to_ring(r)
        acc = Node(Op.ADD, dtype=src.dtype, src=(acc, term))

    analysis = analyze_ring_lowering(Node.sink(acc))

    final_add = acc.base
    assert final_add.op is Op.ADD
    child_through_view = final_add.src[0].base
    assert child_through_view.op is Op.TO_RING

    assert (final_add, child_through_view) in analysis.overflow_add_reduce_edges, (
        "Expected overflow analysis to mark ADD input flowing through TO_RING view."
    )


@pytest.mark.parametrize("dom", [Domain.NTT, Domain.COEFF])
def test_indexify_view_over_lowered_constant(dom):
    """Ring lowering keeps the constant's domain marker for quotient-view indexing."""
    ring = large_prime_ring(8)
    constant = Node.const(7).as_ring(ring, dom)
    lowered = rewrite(constant, lower_ring(analyze_ring_lowering(constant)), bottom_up=True)
    view = lowered.to_ring(ring[2:4])
    indexed = rewrite(view, indexify)
    assert indexed.ring == ring[2:4]
    assert indexed.domain is dom
    assert all(n.op is not Op.TO_RING for n in indexed.toposort())


@pytest.mark.parametrize("specialized", [False, True])
@pytest.mark.parametrize("case", ["empty", "non_store", "store_arity", "computed_target", "indexed_target", "index_arity"])
def test_kernel_rejects_invalid_store_structure(specialized, case):
    ring = large_prime_ring(8)[:1]
    output, source = [Node(Op.PARAM, ring.dtype, arg=ParamArg(i, ring.storage_size, "python")) for i in range(2)]
    idx = Node(Op.GIDX, dtypes.index)
    value = Node(Op.NTT, ring.dtype, src=(source,)) if specialized else source.index(idx).load()
    stores = {
        "empty": (),
        "non_store": (value,),
        "store_arity": (Node(Op.STORE, ring.dtype, src=(output,)),),
        "computed_target": (value.store(value),),
        "indexed_target": (value.index(idx).store(value),),
        "index_arity": (Node(Op.INDEX, ring.dtype, src=(output,)).store(value),),
    }
    ast = Node(Op.SINK, src=stores[case], arg=KernelInfo("python", ring, Domain.NTT))
    with pytest.raises(ValueError, match="Invalid IR node"):
        lower_kernel(ast, SUPPORTED_DTYPES)


@pytest.mark.parametrize(
    "build",
    [
        lambda x, ring: Node(Op.NTT, ring.dtype, src=(x, x)),
        lambda x, ring: Node(Op.INTT, dtypes.uint32, src=(x,)),
        lambda x, ring: Node(Op.NTT, ring.dtype, src=(x + x,)),
        lambda x, ring: Node(Op.RESCALE, ring.dtype, src=(x,), arg=ToRingArg(ring)),
        lambda x, ring: Node(Op.TO_RING, ring.dtype, src=(x,), arg=ring),
        lambda x, ring: Node(Op.TO_RNS, ring.dtype, src=(x,), arg=ToRnsArg(ring, 1)),
        lambda x, ring: Node(Op.TO_RNS, ring.dtype, src=(x.replace(dtype=dtypes.uint32),), arg=ToRnsArg(ring, 0)),
        lambda x, ring: Node(Op.FROM_RNS, ring.dtype, src=(x,)),
        lambda x, ring: Node(Op.RANDOM, ring.dtype, src=(x,), arg=RandomArg(Ternary(ring.n), 0)),
    ],
    ids=["ntt-arity", "intt-dtype", "ntt-expression", "rescale-arg", "to-ring-arg", "to-rns-input", "to-rns-width", "from-rns-output", "rng-key"],
)
def test_kernel_rejects_invalid_specialized_operation(build):
    ring = large_prime_ring(128, dtypes.uint64)[:1]
    source = Node(Op.PARAM, ring.dtype, arg=ParamArg(1, ring.storage_size, "python"))
    value = build(source, ring)
    output = Node(Op.PARAM, value.dtype, arg=ParamArg(0, ring.storage_size, "python"))
    ast = Node.sink(output.store(value)).replace(arg=KernelInfo("python", ring, Domain.NTT))
    with pytest.raises(ValueError, match="Invalid IR node"):
        lower_kernel(ast, SUPPORTED_DTYPES)


def test_generated_spec_rejects_sink_as_index():
    ring = large_prime_ring(8)[:1]
    output, source = [Node(Op.PARAM, ring.dtype, arg=ParamArg(i, ring.storage_size, "python")) for i in range(2)]
    idx = Node(Op.GIDX, dtypes.index)
    invalid_index = Node(Op.SINK, dtypes.index, src=(idx,))
    ast = Node.sink(output.index(invalid_index).store(source.index(idx).load())).replace(arg=KernelInfo("python", ring, Domain.NTT))
    kernel = KernelSpec(ast)
    with pytest.raises(ValueError, match="Invalid IR node"):
        spec.validate(kernel.program, spec.modular)


def test_kernel_rejects_conflicting_parameter_definitions():
    ring = large_prime_ring(128, dtypes.uint32)[:1]
    idx = Node(Op.GIDX, dtypes.index)
    output = Node(Op.PARAM, ring.dtype, arg=ParamArg(0, ring.storage_size, "python"))
    inputs = [Node(Op.PARAM, ring.dtype, arg=ParamArg(1, size, "python")).index(idx).load() for size in (ring.storage_size, ring.storage_size * 2)]
    ast = Node(Op.SINK, src=(output.index(idx).store(inputs[0] + inputs[1]),), arg=KernelInfo("python", ring, Domain.NTT))
    with pytest.raises(ValueError, match="one PARAM definition per slot"):
        lower_kernel(ast, SUPPORTED_DTYPES)


@pytest.mark.parametrize("device", ["python", pytest.param("cuda:0", marks=pytest.mark.requires_cuda)])
@pytest.mark.parametrize("op", [Op.SINK, Op.COPY])
def test_execution_uses_bound_outputs_without_return_values(device, op):
    ring = large_prime_ring(128, dtypes.uint32)[:1]
    buf_spec = BufferSpec(dtype=ring.dtype, size=ring.storage_size, device=device)
    source = Buffer(buf_spec)
    output = Buffer(buf_spec)
    target, operand = (Node.param(i, Node.from_buffer(buf)) for i, buf in enumerate((output, source)))
    if op is Op.SINK:
        idx = Node(Op.GIDX, dtypes.index)
        store = target.index(idx).store(operand.index(idx).load())
        ast = Node(Op.SINK, src=(store,), arg=KernelInfo(device, ring, Domain.NTT))
    else:
        ast = operand.copy_to_device(device)
    call = Node(Op.CALL, src=(ast, Node.from_buffer(output), Node.from_buffer(source)))
    (unit,) = make_exec_units(Node(Op.LINEAR, src=(call,)))
    units = [unit]
    calls = (lambda: unit.program([buf.ensure_allocated() for buf in unit.resolve()]), unit, lambda: run_exec_units(units))

    for iteration, execute in enumerate(calls, 1):
        payload = bytes((i + iteration) % 256 for i in range(source.nbytes))
        source.copyin(memoryview(payload))
        assert execute() is None
        offset = output.offset * ring.dtype.itemsize
        assert bytes(output.as_buffer()) == payload[offset : offset + output.nbytes]
        assert get_output_slots(unit.call.src[0]) == (0,)

    assert not units
    assert run_exec_units([]) is None


def test_backend_compilation_cache_precedes_lowering(monkeypatch):
    from ren.engine import codegen

    ring = RingSpec(n=4, moduli=(17,))
    source = Poly([1, 2, 3, 4], ring, device="python")
    calls, _ = schedule(Node.sink((source + 1).node))
    body = calls.src[-1].src[0]
    backend = PythonBackend()
    program = backend.make_kernel(body)

    def unexpected(*args):
        raise AssertionError("cached kernel was lowered again")

    monkeypatch.setattr(codegen, "lower_kernel", unexpected)
    assert backend.make_kernel(body) is program
