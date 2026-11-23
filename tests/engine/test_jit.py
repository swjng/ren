import gc
import importlib
import weakref
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, assert_type

import pytest

import ren
from ren.allocator import BufferSpec
from ren.buffer import Buffer
from ren.dtype import dtypes
from ren.engine.realize import ExecUnit
from ren.ir import Node, Op
from ren.poly import Poly
from ren.ring import Domain, RingSpec
from ren.ckks import Ciphertext
from ren.ckks.metadata import CKKSNoiseInfo
from ren.utils import disable_gc
from tests.utils import large_prime_ring, neg_convolve_naive


@pytest.mark.parametrize("invalid", [None, 1])
def test_jit_decorator_rejects_noncallables(invalid):
    with pytest.raises(TypeError, match="jit decorator expected a function"):
        ren.jit(prune=True)(invalid)


def test_jit_add_polys():
    ring = large_prime_ring(4)

    @ren.jit
    def add_polys(a: Poly, b: Poly) -> Poly:
        return a + b

    a1 = Poly([1, 2, 3, 4], ring)
    b1 = Poly([4, 3, 2, 1], ring)
    c1 = assert_type(add_polys(a1, b1), Poly)
    assert isinstance(c1, Poly)
    assert c1.ring == ring
    assert c1.tolist() == [5, 5, 5, 5]

    # Second call with different inputs should reuse the captured plan but
    # produce the correct new result.
    a2 = Poly([2, 2, 2, 2], ring)
    b2 = Poly([1, 1, 1, 1], ring)
    c2 = add_polys(a2, b2)
    assert c1 is not c2  # default fresh-output: each replay returns a new object
    assert c2.tolist() == [3, 3, 3, 3]


@pytest.mark.parametrize("source", ["shared", "closure", "stashed"])
@pytest.mark.parametrize("prune", [False, True])
def test_jit_capture_preserves_surviving_polys(source: str, prune: bool):
    ring = large_prime_ring(4)
    x = Poly([1, 2, 3, 4], ring).realize()
    external = (Poly([1, 2, 3, 4], ring).realize() if source == "closure" else x) * 2
    stashed = []

    @ren.jit(prune=prune)
    def double(x: Poly) -> Poly:
        result = external if source == "closure" else x * 2
        stashed.append(result)
        return result

    first = assert_type(double(x), Poly)
    surviving = stashed[0] if source == "stashed" else external
    assert first.tolist() == surviving.tolist() == [2, 4, 6, 8]
    second = double(Poly([4, 3, 2, 1], ring))
    assert second.tolist() == ([2, 4, 6, 8] if source == "closure" else [8, 6, 4, 2])
    assert first.tolist() == surviving.tolist() == [2, 4, 6, 8]


def test_jit_failed_plan_build_preserves_surviving_polys(monkeypatch: pytest.MonkeyPatch):
    jit_module = importlib.import_module("ren.engine.jit")
    ring = large_prime_ring(4)
    x = Poly([1, 2, 3, 4], ring).realize()
    external = x * 2

    def fail(*args, **kwargs):
        raise RuntimeError("plan build failed")

    monkeypatch.setattr(jit_module.CapturedPlan, "__post_init__", fail)
    with pytest.raises(RuntimeError, match="plan build failed"):
        ren.jit(lambda x: x * 2)(x)
    assert external.tolist() == [2, 4, 6, 8]


@pytest.mark.parametrize("prune", [False, True])
def test_jit_view_output_computes_producer_once(prune: bool):
    from ren.engine.realize import call_op

    ring = large_prime_ring(4)
    small = ring[:2]
    step = ren.jit(lambda x, y: (x * y).intt().to_quotient_ring(small), prune=prune)
    y = Poly([4, 3, 2, 1], ring).realize()
    for values in ([1, 2, 3, 4], [4, 3, 2, 1]):
        x = Poly(values, ring).realize()
        assert step(x, y).tolist() == (x * y).intt().to_quotient_ring(small).tolist()
    plan = next(iter(step._plans.values()))
    assert sum(call_op(unit.call) is Op.INTT for unit in plan.units) == 1


@pytest.mark.parametrize("record", [dataclass, lambda cls: cls], ids=["dataclass", "annotated-class"])
def test_jit_replays_polys_inside_ordinary_record_hierarchies(record):
    class Base:
        p: object

    @record
    class State(Base):
        label: str = "metadata"

    ring = large_prime_ring(4)
    state = object.__new__(State)
    captures = []

    @ren.jit
    def add(state, visible):
        captures.append(None)
        return state.p + visible

    visible = Poly([1, 1, 1, 1], ring)
    outputs = []
    for value in (2, 7):
        state.p = Poly([value] * 4, ring)
        outputs.append(add(state, visible))
        assert outputs[-1].tolist() == [value + 1] * 4
    assert outputs[0].tolist() == [3] * 4
    assert len(captures) == 1


@pytest.mark.parametrize("decorate", [dataclass, lambda cls: cls], ids=["dataclass", "annotated-subclass"])
def test_jit_replays_polys_added_by_metadata_subclasses(decorate):
    @dataclass
    class Metadata:
        label: str

    @decorate
    class WithPoly(Metadata):
        p: Poly

        def __init__(self, label, p):
            super().__init__(label)
            self.p = p

    @dataclass
    class Batch:
        p: Poly
        metadata: Metadata

    @ren.jit
    def add(batch):
        return batch.p + batch.metadata.p

    ring = large_prime_ring(4)
    for left, right in ((1, 10), (2, 20)):
        batch = Batch(Poly([left] * 4, ring), WithPoly("same", Poly([right] * 4, ring)))
        assert add(batch).tolist() == [left + right] * 4


def test_jit_replays_unannotated_attrs_fields():
    from attrs import field, make_class

    Record = make_class("Record", {"first": field(type=Poly), "second": field()})

    @ren.jit
    def add(record):
        return record.first + record.second

    ring = large_prime_ring(4)
    for left, right in ((1, 10), (2, 20)):
        record = Record(Poly([left] * 4, ring), Poly([right] * 4, ring))
        assert add(record).tolist() == [left + right] * 4


def test_jit_replays_nested_tree_reduce_hooks():
    class Queue:
        def __init__(self, items, label):
            self.items = deque(items)
            self.label = label

        def __tree_reduce__(self):
            return type(self), (tuple(self.items), self.label)

    ring = large_prime_ring(4)
    captures = []

    @ren.jit
    def double(inputs):
        captures.append(None)
        return [Queue([p + p for p in inputs["queue"].items], "doubled")]

    for values in ((1, 2), (5, 7)):
        (result,) = double({"queue": Queue([Poly([value] * 4, ring) for value in values], "input")})
        assert type(result) is Queue and type(result.items) is deque and result.label == "doubled"
        assert [p.tolist() for p in result.items] == [[2 * value] * 4 for value in values]
    assert len(captures) == 1


def test_jit_realizes_tuple_of_ciphertexts_and_rebuilds_outputs():
    ring = large_prime_ring(4)
    captures = []

    @ren.jit
    def negate(cts: tuple[Ciphertext, ...]) -> tuple[Ciphertext, ...]:
        captures.append(None)
        return tuple(-ct for ct in cts)

    def ciphertext(offset: int) -> Ciphertext:
        a = Poly([offset, 2, 3, 4], ring) + 1
        b = Poly([offset + 1, 4, 3, 2], ring) + 2
        return Ciphertext(a, b, noise_info=CKKSNoiseInfo.zero())

    outputs = []
    for offset in (1, 7):
        cts = ciphertext(offset), ciphertext(offset + 10)
        assert all(not poly.is_realized for ct in cts for poly in ct.polys())
        result = negate(cts)
        assert all(poly.is_realized for ct in cts for poly in ct.polys())
        assert isinstance(result, tuple)
        assert len(result) == len(cts)
        for source, output in zip(cts, result, strict=True):
            assert isinstance(output, Ciphertext)
            assert output is not source
            assert output.a.tolist() == [-x % ring.q for x in source.a.tolist()]
            assert output.b.tolist() == [-x % ring.q for x in source.b.tolist()]
        outputs.append(result)

    assert len(captures) == 1
    assert outputs[0][0] is not outputs[1][0]
    assert outputs[0][0].a.tolist() == [-2 % ring.q, -2 % ring.q, -3 % ring.q, -4 % ring.q]


def test_jit_kwargs_are_bound_by_name_not_insertion_order():
    ring = large_prime_ring(4)

    @ren.jit
    def subtract(a: Poly, b: Poly) -> Poly:
        return a - b

    first = subtract(a=Poly([5, 5, 5, 5], ring), b=Poly([1, 1, 1, 1], ring))
    second = subtract(b=Poly([2, 2, 2, 2], ring), a=Poly([7, 7, 7, 7], ring))

    assert first.tolist() == [4, 4, 4, 4]
    assert second.tolist() == [5, 5, 5, 5]


@disable_gc()
def test_jit_poly_container_output_preserves_aliases():
    ring = large_prime_ring(4)
    constant = Poly.const(7, ring)
    captured_refs: list[tuple[weakref.ReferenceType, weakref.ReferenceType]] = []

    @dataclass(eq=False)
    class Wrapped(ren.PolyContainer):
        values: dict[str, tuple[Poly, Poly]]

    @ren.jit
    def wrap(value: Poly) -> list:
        result = value + 1
        wrapped = Wrapped({"pair": (result, result), "constant": (constant, constant)})
        captured_refs.append((weakref.ref(wrapped), weakref.ref(result)))
        return [wrapped, wrapped, wrapped.values, result, constant]

    for values, expected in (([1, 2, 3, 4], [2, 2, 3, 4]), ([4, 3, 2, 1], [5, 3, 2, 1])):
        output = wrap(Poly(values, ring))
        assert output[0] is output[1]
        assert output[0].values is output[2]
        assert output[0].values["pair"][0] is output[0].values["pair"][1] is output[3]
        assert output[0].values["constant"][0] is output[0].values["constant"][1] is output[4] is constant
        assert output[0].values["pair"][0].tolist() == expected
        del output
        assert all(ref() is None for ref in captured_refs[0])
    assert len(captured_refs) == 1


def test_jit_replay_rebuilds_constant_output_structure():
    ring = large_prime_ring(4)

    @ren.jit
    def wrap(_value: Poly) -> list:
        return [Poly.const(7, ring), [1, 2, 3]]

    first = wrap(Poly([1, 2, 3, 4], ring))
    first[1].append(4)
    second = wrap(Poly([4, 3, 2, 1], ring))

    assert second is not first
    assert second[0].tolist() == first[0].tolist() == [7, 0, 0, 0]
    assert second[1] == [1, 2, 3]


def test_jit_fresh_output_double_use():
    """Calling a jitted fn twice within one unrealized computation must not
    overwrite the first still-live output. Default fresh-output makes this safe."""
    ring = large_prime_ring(4)

    @ren.jit
    def add_one(a: Poly) -> Poly:
        return a + 1

    a = Poly([0, 0, 0, 0], ring)
    b = Poly([1, 1, 1, 1], ring)
    c = Poly([2, 2, 2, 2], ring)
    # Poly + 1 bumps only the constant term: add_one(b)=[2,1,1,1], add_one(c)=[3,2,2,2].
    a = a + add_one(b)  # first output still live
    a = a + add_one(c)  # must not clobber the first (reuse would give a[0]=6)
    assert a.tolist() == [5, 3, 3, 3]


@pytest.mark.parametrize("calls", [1, 2], ids=["capture", "replay"])
def test_jit_releases_dead_output_buffers(calls):
    ring = large_prime_ring(4)

    @ren.jit
    def add_one(a: Poly) -> Poly:
        return a + 1

    for _ in range(calls):
        out = add_one(Poly([1, 1, 1, 1], ring))
    buffer_ref = weakref.ref(out.node.base.buffer)
    del out

    assert buffer_ref() is None


@pytest.mark.parametrize("view_kind", ["polynomial", "buffer"])
def test_jit_retained_output_view(view_kind):
    ring = RingSpec(n=4, moduli=(17, 97))

    @ren.jit
    def add_one(a: Poly) -> Poly:
        return a + 1

    add_one(Poly([9, 9, 9, 9], ring))
    out = add_one(Poly([1, 1, 1, 1], ring))
    buffer_ref = weakref.ref(out.node.base.buffer)
    small = ring[:1]
    if view_kind == "polynomial":
        view = out.to_quotient_ring(small)
    else:
        storage = out.node.base.buffer.view(small.storage_size, 0).ensure_allocated()
        view = Poly._from_buffer(storage, small, Domain.COEFF)
        del storage
    del out
    assert buffer_ref() is not None

    replacement = add_one(Poly([3, 3, 3, 3], ring))
    assert replacement.tolist() == [4, 3, 3, 3]
    assert view.tolist() == [2, 1, 1, 1]
    del view
    assert buffer_ref() is None


def test_jit_output_in_named_tuple():
    ring = large_prime_ring(4)

    class Wrapped(NamedTuple):
        p: Poly

    @ren.jit
    def add_one(a: Poly) -> Wrapped:
        return Wrapped(a + 1)

    out1 = add_one(Poly([1, 1, 1, 1], ring))
    out2 = add_one(Poly([2, 2, 2, 2], ring))
    assert isinstance(out2, Wrapped)
    assert out1.p.tolist() == [2, 1, 1, 1]
    assert out2.p.tolist() == [3, 2, 2, 2]


def test_jit_retained_buffer_survives_clear():
    ring = large_prime_ring(4)
    step = ren.jit(lambda x: x + 1)
    source = Poly([1, 2, 3, 4], ring).realize()
    step(source)
    output = step(source)
    storage, domain = output.node.base.buffer, output.domain
    storage_ref = weakref.ref(storage)
    del output

    assert step(Poly([5, 6, 7, 8], ring)).tolist() == [6, 6, 7, 8]
    step.clear_jit()
    assert Poly._from_buffer(storage, ring, domain).tolist() == [2, 2, 3, 4]
    del storage
    gc.collect()
    assert storage_ref() is None


@pytest.mark.parametrize("failure", ["allocation", "borrower"])
def test_jit_retries_failed_output_preparation(monkeypatch: pytest.MonkeyPatch, failure: str):
    ring = large_prime_ring(4)
    step = ren.jit(lambda x: (x + 1, x + 2))
    source = Poly([1, 2, 3, 4], ring).realize()
    step(source)
    target, method = (Buffer, "allocate") if failure == "allocation" else (importlib.import_module("ren.engine.jit"), "_OutputBuffer")
    original = getattr(target, method)
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise MemoryError("output preparation failed")
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(target, method, fail_second)
        with pytest.raises(MemoryError, match="output preparation failed"):
            step(source)

    first, second = step(source)
    assert first.tolist() == [2, 2, 3, 4]
    assert second.tolist() == [3, 2, 3, 4]


def test_jit_rejects_cyclic_output():
    ring = large_prime_ring(4)

    @ren.jit
    def add_one(a: Poly) -> list:
        out: list = [a + 1]
        out.append(out)
        return out

    with pytest.raises(ValueError, match="containers cannot contain cycles"):
        add_one(Poly([1, 1, 1, 1], ring))


def test_jit_output_in_poly_container_with_custom_init():
    """Rebuilding a dataclass must not assume its fields mirror __init__."""
    ring = large_prime_ring(4)

    @dataclass(init=False, eq=False)
    class Wrapped(ren.PolyContainer):
        tag: str
        p: Poly

        def __init__(self, data: Poly):
            self.tag = "wrapped"
            self.p = data

        def __reduce__(self):
            raise AssertionError("jit output rebuilding must not serialize dataclasses")

    @ren.jit
    def add_wrapped(a: Poly, b: Poly) -> Wrapped:
        return Wrapped(a + b)

    w1 = add_wrapped(Poly([1, 1, 1, 1], ring), Poly([1, 1, 1, 1], ring))
    assert w1.p.tolist() == [2, 2, 2, 2]

    w2 = add_wrapped(Poly([2, 2, 2, 2], ring), Poly([3, 3, 3, 3], ring))
    assert w2 is not w1
    assert w2.tag == "wrapped"
    assert w2.p.tolist() == [5, 5, 5, 5]
    assert w1.p.tolist() == [2, 2, 2, 2]


def test_jit_static_output_rewrites_in_place():
    ring = large_prime_ring(4)

    @ren.jit(static_output=True)
    def add_polys(a: Poly, b: Poly) -> Poly:
        return a + b

    out1 = add_polys(Poly([1, 1, 1, 1], ring), Poly([1, 1, 1, 1], ring))
    assert out1.tolist() == [2, 2, 2, 2]

    out2 = add_polys(Poly([2, 2, 2, 2], ring), Poly([3, 3, 3, 3], ring))
    assert out2 is out1
    assert out2.tolist() == [5, 5, 5, 5]


def test_jit_identity_output_gets_fresh_storage():
    """Identity-function replays copy the current input into independent output storage."""
    ring = large_prime_ring(4)

    @ren.jit
    def ident(a: Poly) -> Poly:
        return a

    a1 = Poly([1, 2, 3, 4], ring)
    y1 = ident(a1)
    assert y1 is not a1
    assert y1.tolist() == [1, 2, 3, 4]

    a2 = Poly([5, 6, 7, 8], ring)
    y2 = ident(a2)
    assert y2.tolist() == [5, 6, 7, 8]
    assert y1.tolist() == [1, 2, 3, 4]  # first output still intact


def test_jit_identity_output_from_full_buffer_view_gets_fresh_storage():
    ring = large_prime_ring(4)
    source = Poly([1, 2, 3, 4], ring).realize()
    view = source.node.base.buffer.view(ring.storage_size, 0).ensure_allocated()
    x = Poly._from_buffer(view, ring, Domain.COEFF)

    @ren.jit
    def ident(a: Poly) -> Poly:
        return a

    y1 = ident(x)
    y2 = ident(Poly([5, 6, 7, 8], ring))

    assert y1 is not x
    assert y1.tolist() == [1, 2, 3, 4]
    assert y2 is not y1
    assert y2.tolist() == [5, 6, 7, 8]


def test_jit_distinct_output_polys_keep_shared_storage():
    ring = large_prime_ring(4)

    @ren.jit
    def pair(a: Poly) -> tuple[Poly, Poly]:
        return a, Poly(a.node)

    out1 = pair(Poly([1, 2, 3, 4], ring))
    out2 = pair(Poly([5, 6, 7, 8], ring))

    assert out1[0] is not out1[1]
    assert out1[0].node.base.buffer is out1[1].node.base.buffer
    assert out2[0] is not out2[1]
    assert out2[0].node.base.buffer is out2[1].node.base.buffer
    assert out1[0].tolist() == out1[1].tolist() == [1, 2, 3, 4]
    assert out2[0].tolist() == out2[1].tolist() == [5, 6, 7, 8]


def test_jit_plan_key_distinguishes_input_buffer_aliasing():
    ring = large_prime_ring(4)
    shared = Poly([3, 4, 5, 6], ring).realize().node.base.buffer
    x1 = Poly._from_buffer(shared, ring, Domain.COEFF)
    x2 = Poly._from_buffer(shared, ring, Domain.COEFF)

    @ren.jit
    def add(a: Poly, b: Poly) -> Poly:
        return a + b

    assert add(x1, x2).tolist() == [6, 8, 10, 12]
    assert add(Poly([10, 20, 30, 40], ring), Poly([1, 2, 3, 4], ring)).tolist() == [11, 22, 33, 44]
    assert len(add._plans) == 2


@pytest.mark.parametrize("ir_view", [False, True])
def test_jit_tracks_input_views_by_ir_provenance(ir_view):
    ring3 = large_prime_ring(4)[:3]
    ring2 = large_prime_ring(4)[:2]

    def inputs(seed: int) -> tuple[Poly, Poly]:
        root = Buffer(BufferSpec(dtype=ring3.dtype, size=ring3.storage_size + ring3.n, device="python")).ensure_allocated()
        root.copyin(memoryview(bytes((i + seed) % 251 for i in range(root.nbytes))))
        if ir_view:
            source = Node.from_buffer(root)
            views = [Poly(Node(Op.BUFFER_VIEW, root.dtype, (source,), (ring3.storage_size, offset)), ring3) for offset in (0, ring3.n)]
        else:
            views = [Poly._from_buffer(root.view(ring3.storage_size, offset).ensure_allocated(), ring3, Domain.COEFF) for offset in (0, ring3.n)]
        return views[0], views[1]

    @ren.jit
    def f(_x: Poly, y: Poly) -> Poly:
        return y._to_ring(ring2).contiguous()

    for seed in (0, 17):
        x, y = inputs(seed)
        assert f(x, y).tolist() == y.to_quotient_ring(ring2).contiguous().tolist()


def test_poly_clone_snapshots_jit_output():
    ring = large_prime_ring(4)

    @ren.jit
    def double(x: Poly) -> Poly:
        return x + x

    out1 = double(Poly([1, 2, 3, 4], ring))
    snap = out1.clone().realize()

    out2 = double(Poly([10, 20, 30, 40], ring))
    assert out1 is not out2  # default fresh-output: each replay returns a new object
    assert snap.tolist() == [2, 4, 6, 8]
    assert out2.tolist() == [20, 40, 60, 80]
    assert snap.node.base.buffer is not out2.node.base.buffer


def test_jit_mul_polys():
    ring = large_prime_ring(4)

    @ren.jit
    def mul_polys(a: Poly, b: Poly) -> Poly:
        return a * b

    a = Poly([1, 2, 3, 4], ring)
    b = Poly([4, 3, 2, 1], ring)
    c1 = mul_polys(a, b)

    # Expected schoolbook negacyclic product in R_q[x]/(x^n+1).
    coeffs_a = [1, 2, 3, 4]
    coeffs_b = [4, 3, 2, 1]
    expected1 = neg_convolve_naive(coeffs_a, coeffs_b, n=ring.n, q=ring.q)
    assert c1.tolist() == expected1

    # Second call with different inputs should also work.
    a2 = Poly([0, 1, 0, 1], ring)
    b2 = Poly([1, 0, 1, 0], ring)
    c2 = mul_polys(a2, b2)
    assert c1 is not c2  # default fresh-output: each replay returns a new object

    coeffs_a2 = [0, 1, 0, 1]
    coeffs_b2 = [1, 0, 1, 0]
    expected2 = neg_convolve_naive(coeffs_a2, coeffs_b2, n=ring.n, q=ring.q)
    assert c2.tolist() == expected2


def test_jit_output_fed_back_into_same_function():
    """
    Regression: replaying a jitted function on its own prior output must match eager semantics.

    This specifically covers cases where realized outputs are in a different domain
    than the initial capture input.
    """
    ring = large_prime_ring(8)

    def step(x: Poly) -> Poly:
        return x + x

    @ren.jit
    def step_jit(x: Poly) -> Poly:
        return step(x)

    x_jit = Poly([1, 2, 3, 4, 5, 6, 7, 8], ring)
    for _ in range(3):
        x_jit = step_jit(x_jit)

    # Eager reference across call boundaries (realize each step like jitted calls do).
    x_ref = Poly([1, 2, 3, 4, 5, 6, 7, 8], ring)
    for _ in range(3):
        x_ref = step(x_ref)
        x_ref.realize()

    assert x_jit.tolist() == x_ref.tolist()


def test_jit_nested_polys_in_poly_container():
    ring = large_prime_ring(4)

    @dataclass(slots=True)
    class Wrapper(ren.PolyContainer):
        a: Poly
        scale: int

    @ren.jit
    def add_wrapped(w: Wrapper, b: Poly) -> Poly:
        # Only use w.a and b, but JIT should discover both Polys inside args.
        return w.a + b

    w1 = Wrapper(a=Poly([1, 2, 3, 4], ring), scale=1)
    b1 = Poly([4, 3, 2, 1], ring)
    c1 = add_wrapped(w1, b1)
    assert c1.tolist() == [5, 5, 5, 5]

    # Second call with different wrapped Poly and b should reuse the plan.
    w2 = Wrapper(a=Poly([2, 2, 2, 2], ring), scale=2)
    b2 = Poly([1, 1, 1, 1], ring)
    c2 = add_wrapped(w2, b2)
    assert c1 is not c2  # default fresh-output: each replay returns a new object
    assert c2.tolist() == [3, 3, 3, 3]


def test_jit_walker_does_not_retain_container_instances():
    ring = large_prime_ring(4)

    def collect_local_inputs():
        class Wrapped:
            poly: Poly

            def __init__(self, poly):
                self.poly = poly

        @ren.jit
        def double(value):
            return value.poly + value.poly

        poly = Poly([1, 2, 3, 4], ring)
        value = Wrapped(poly)
        assert double(value).tolist() == [2, 4, 6, 8]
        return double, weakref.ref(value), weakref.ref(poly)

    double, value_ref, poly_ref = collect_local_inputs()
    gc.collect()

    assert value_ref() is poly_ref() is None
    double.clear_jit()


def test_jit_multi_output_poly_container():
    ring = large_prime_ring(4)

    @dataclass(slots=True)
    class Pair(ren.PolyContainer):
        left: Poly
        right: Poly

    @ren.jit
    def add_sub(a: Poly, b: Poly) -> Pair:
        return Pair(a + b, a - b)

    def mod_q(vals: list[int]) -> list[int]:
        return [x % ring.q for x in vals]

    a1 = Poly([1, 2, 3, 4], ring)
    b1 = Poly([4, 3, 2, 1], ring)
    out1 = add_sub(a1, b1)
    assert out1.left.tolist() == mod_q([5, 5, 5, 5])
    assert out1.right.tolist() == mod_q([-3, -1, 1, 3])

    a2 = Poly([2, 2, 2, 2], ring)
    b2 = Poly([1, 1, 1, 1], ring)
    out2 = add_sub(a2, b2)
    assert out1 is not out2  # default fresh-output
    assert out2.left.tolist() == mod_q([3, 3, 3, 3])
    assert out2.right.tolist() == mod_q([1, 1, 1, 1])


def test_jit_nested_call_fails_before_inner_body():
    ring = RingSpec(n=4, moduli=(17, 97))
    inner_calls = 0

    @ren.jit
    def inner(x: Poly) -> Poly:
        nonlocal inner_calls
        inner_calls += 1
        return x + x

    @ren.jit
    def outer(x: Poly) -> Poly:
        return inner(x) + 1

    x = Poly([1, 2, 3, 4], ring)
    with pytest.raises(RuntimeError, match="nested jit"):
        outer(x)

    assert inner_calls == 0
    assert len(inner._plans) == 0
    assert len(outer._plans) == 0


@pytest.mark.parametrize("lazy_index", [0, 1])
def test_jit_unrealized_inputs(lazy_index: int):
    """JIT auto-realizes lazy inputs and reuses the capture for fresh inputs."""
    ring = large_prime_ring(4)
    captures = 0

    @ren.jit
    def add_polys(a: Poly, b: Poly) -> Poly:
        nonlocal captures
        captures += 1
        return a + b

    for values in ([1, 2, 3, 4], [4, 3, 2, 1]):
        # Batch realization prepares the lazy input while the constant stays device-free.
        constant = Poly.const(5, ring) + 2
        lazy = Poly(values, ring) * 2
        assert not lazy.is_realized
        args = (lazy, constant) if lazy_index == 0 else (constant, lazy)
        result = add_polys(*args)
        assert constant.device is None
        assert result.device == lazy.device
        assert result.tolist() == [values[0] * 2 + 7, *(value * 2 for value in values[1:])]

    assert captures == 1


def test_jit_rejects_unallocated_input():
    ring = large_prime_ring(4)
    poly = Poly(Node.new_buffer("python", ring.storage_size, ring.dtype), ring)

    with pytest.raises(RuntimeError, match="input Poly did not realize"):
        ren.jit(lambda x: x + 1)(poly)


@pytest.mark.parametrize("static_output", [False, True])
def test_jit_constant_output_poly(static_output: bool):
    ring = large_prime_ring(4)

    @ren.jit(static_output=static_output)
    def constant_poly(_x: Poly) -> Poly:
        return Poly.const(7, ring)

    x1 = Poly([1, 2, 3, 4], ring)
    y1 = constant_poly(x1)
    assert y1.tolist() == [7 % ring.q, 0, 0, 0]

    x2 = Poly([4, 3, 2, 1], ring)
    y2 = constant_poly(x2)
    assert y1 is y2
    assert y2.tolist() == [7 % ring.q, 0, 0, 0]


def test_jit_constant_input_poly():
    ring = large_prime_ring(4)

    @ren.jit
    def add(a: Poly, b: Poly) -> Poly:
        return a + b

    const_a1 = Poly.const(7, ring)
    b1 = Poly([1, 2, 3, 4], ring)
    out1 = add(const_a1, b1)
    assert out1.tolist() == [8, 2, 3, 4]

    const_a2 = Poly.const(7, ring)
    b2 = Poly([4, 3, 2, 1], ring)
    out2 = add(const_a2, b2)
    assert out1 is not out2  # default fresh-output
    assert out2.tolist() == [11, 3, 2, 1]


def test_jit_constant_input_poly_value_change_recaptures():
    ring = large_prime_ring(4)

    @ren.jit
    def add(a: Poly, b: Poly) -> Poly:
        return a + b

    out1 = add(Poly.const(7, ring), Poly([1, 2, 3, 4], ring))
    assert out1.tolist() == [8, 2, 3, 4]
    assert len(add._plans) == 1

    out2 = add(Poly.const(8, ring), Poly([4, 3, 2, 1], ring))
    assert out2.tolist() == [12, 3, 2, 1]
    assert len(add._plans) == 2


def test_jit_capture_replans_intermediate_buffers(monkeypatch: pytest.MonkeyPatch):
    """
    Regression test: capture-time replanning should find optimizable intermediates.

    If capture stores live runtime buffers directly, all candidates appear allocated
    and the planner returns an empty assignment map.
    """
    ring = large_prime_ring(256)
    jit_module = importlib.import_module("ren.engine.jit")

    planner_results: list[int] = []
    real_planner = jit_module.plan_memory
    from ren.engine.realize import buffer_binding, rewrite_buffers

    rewrites, normalized = [], []

    def tracking_planner(linear, noopt_buffers=None):
        assigned = real_planner(linear, noopt_buffers)
        planner_results.append(len(assigned))
        return assigned

    monkeypatch.setattr(jit_module, "plan_memory", tracking_planner)

    def tracking_rewrite(*args):
        rewrites.append(None)
        return rewrite_buffers(*args)

    def tracking_binding(node):
        normalized.append(node)
        return buffer_binding(node)

    monkeypatch.setattr(jit_module, "rewrite_buffers", tracking_rewrite)
    monkeypatch.setattr(importlib.import_module("ren.engine.schedule"), "buffer_binding", tracking_binding)

    @ren.jit
    def chained_views(x: Poly) -> Poly:
        y = x
        for _ in range(16):
            y = y._to_ring(ring).rotate(1).contiguous()
        return y

    x = Poly([1] * ring.n, ring).realize()
    normalized.clear()
    _ = chained_views(x)

    assert planner_results, "capture-time replanning was not invoked"
    assert planner_results[-1] > 0, "expected capture-time memory planner to remap at least one intermediate buffer"
    assert len(rewrites) == 1
    assert normalized and len(normalized) == len(set(normalized))


@pytest.mark.parametrize("realize", [ren.realize, Poly.realize], ids=["function", "method"])
@pytest.mark.parametrize("prune", [False, True])
def test_jit_capture_allows_realize(realize, prune):
    ring = large_prime_ring(4)
    small = RingSpec(n=ring.n, moduli=ring.moduli[:1])
    saved = []

    @ren.jit(prune=prune)
    def staged(x: Poly):
        y = x + 1
        z = y * 2
        realize(y)
        realize(z)
        saved.append(y)
        view = z.to_quotient_ring(small)
        realize(view)
        return y, view, view, view + 1

    outputs = []
    for value in (1, 5, 9):
        output = staged(Poly([value] * ring.n, ring))
        outputs.append(output)
        assert output[1] is output[2]
    for value, (y, view, _, result) in zip((1, 5, 9), outputs, strict=True):
        assert y.tolist() == [value + 1, value, value, value]
        assert view.tolist() == [2 * value + 2, 2 * value, 2 * value, 2 * value]
        assert result.tolist() == [2 * value + 3, 2 * value, 2 * value, 2 * value]
    assert len(saved) == 1
    assert saved[0].tolist() == [2, 1, 1, 1]


@pytest.mark.parametrize("read", [Poly.data, Poly.tobytes, Poly.tolist, Poly.to_centered_list, list])
@pytest.mark.parametrize("materialized", [False, True])
def test_jit_capture_rejects_host_reads(read, materialized):
    ring = large_prime_ring(4)
    saved = []

    @ren.jit
    def read_value(x: Poly) -> Poly:
        y = x + 1
        if materialized:
            y.realize()
        saved.append(y)
        read(y)
        return y

    x = Poly([1] * ring.n, ring)
    with pytest.raises(RuntimeError, match="cannot read polynomial data during JIT capture"):
        read_value(x)
    assert saved[0].tolist() == [2, 1, 1, 1]
    assert x.tolist() == [1, 1, 1, 1]


def test_jit_capture_rejects_random():
    ring = large_prime_ring(4)

    @ren.jit
    def add_noise(x: Poly) -> Poly:
        return x + Poly.ternary(x.ring)

    x = Poly([1] * ring.n, ring)

    with pytest.raises(RuntimeError, match="Random"):
        add_noise(x)


def test_jit_replay_does_not_mutate_or_retain_inputs_after_exception():
    jit_module = importlib.import_module("ren.engine.jit")
    spec = BufferSpec(dtype=dtypes.uint8, size=1, device="python")
    output = Buffer(spec)
    bound_input = Buffer(spec)
    input_id = id(bound_input)
    output_node = Node.from_buffer(output)
    param = Node.param(0, Node.from_buffer(bound_input))

    def fail(bufs: list[Buffer]) -> None:
        assert id(bufs[1]) == input_id
        raise RuntimeError("boom")

    body = Node.param(1, param).copy_to_device(output.device)
    plan = jit_module.CapturedPlan(
        units=(ExecUnit(fail, Node(Op.CALL, src=(body, output_node, param))),),
        bindings=[bound_input],
        output_plan=jit_module._OutputPlan(pool={}, build=lambda _: None),
        reused_output_bases=set(),
    )

    with pytest.raises(RuntimeError, match="boom"):
        plan.replay([bound_input])

    assert plan.units[0].call.src[2] is param
    input_ref = weakref.ref(bound_input)
    del bound_input
    gc.collect()
    assert input_ref() is None


def test_jit_replay_builds_fresh_output_after_execution():
    jit_module = importlib.import_module("ren.engine.jit")
    ring = large_prime_ring(4)
    output = Buffer(BufferSpec(dtype=ring.dtype, size=ring.storage_size, device="python"))
    events: list[str] = []

    executed: list[Buffer] = []

    def build(buffers):
        assert buffers == executed
        events.append("build")
        return "result"

    def execute(bufs: list[Buffer]) -> None:
        assert len(bufs) == 1 and bufs[0] is not output
        assert bufs[0].spec == output.spec and bufs[0].is_allocated()
        executed.extend(bufs)
        events.append("execute")

    param = Node.param(0, Node.from_buffer(output))
    body = Node.sink(param.store(Node.const(0, output.dtype)))
    plan = jit_module.CapturedPlan(
        units=(ExecUnit(execute, Node(Op.CALL, src=(body, param))),),
        bindings=[output],
        output_plan=jit_module._OutputPlan(pool={Buffer(output.spec): None}, build=build),
        reused_output_bases=set(),
    )

    assert plan.replay([]) == "result"
    assert events == ["execute", "build"]


def test_jit_prune_livein_data_preserved():
    """
    Regression: prune isolation must preserve live-in data from onetime execution.

    A buffer produced by one-time work remains a concrete replay input after
    its producer calls are removed.
    """
    ring = large_prime_ring(4)

    @ren.jit(prune=True)
    def f(x: Poly) -> Poly:
        c = (Poly([10, 20, 30, 40], ring) + Poly([1, 1, 1, 1], ring)).contiguous()
        return x + c

    x1 = Poly([1, 2, 3, 4], ring)
    y1 = f(x1)
    assert y1.tolist() == [12, 23, 34, 45]

    x2 = Poly([5, 6, 7, 8], ring)
    y2 = f(x2)
    assert y2.tolist() == [16, 27, 38, 49]

    plan = next(iter(f._plans.values()))
    assert all(source.op in {Op.BUFFER, Op.BUFFER_VIEW, Op.PARAM} for unit in plan.units for source in unit.call.src[1:])


def test_jit_prune_retains_latest_livein_write():
    """
    Regression for retaining the latest one-time live-in write.

    When a replay live-in is produced by a chain of onetime operations
    (multiple intermediate writes that may share arena storage), replay must
    observe the final computed value — not an earlier intermediate.
    """
    ring = large_prime_ring(64)
    ring2 = type(ring)(n=ring.n, moduli=ring.moduli[:2])
    ring1 = type(ring)(n=ring.n, moduli=ring.moduli[:1])

    def reference(x: Poly) -> Poly:
        # Chain of input-independent ops producing intermediates.
        c = Poly([(i * 7 + 3) % ring.q for i in range(ring.n)], ring)
        c = c._to_ring(ring1)._to_ring(ring2)._to_ring(ring1)
        c = (c + c).contiguous()
        return x._to_ring(c.ring) + c

    @ren.jit(prune=True)
    def f(x: Poly) -> Poly:
        return reference(x)

    x1 = Poly([1] * ring.n, ring)
    assert f(x1).tolist() == reference(x1).tolist()

    x2 = Poly([2] * ring.n, ring)
    assert f(x2).tolist() == reference(x2).tolist()


def test_jit_replay_preserves_buffer_view_base_invariant():
    """
    Regression test for replay failure in BUFFER_VIEW validation.

    A capture-time remap can replace a base buffer while leaving an allocated
    view buffer unchanged, which breaks view._base == base.base during replay.
    """
    ring = large_prime_ring(64)
    ring3 = type(ring)(n=ring.n, moduli=ring.moduli[:3])
    ring2 = type(ring)(n=ring.n, moduli=ring.moduli[:2])
    ring1 = type(ring)(n=ring.n, moduli=ring.moduli[:1])

    def transform(x: Poly) -> Poly:
        return x._to_ring(ring1)._to_ring(ring2)._to_ring(ring1)

    @ren.jit
    def f(x: Poly) -> Poly:
        return transform(x)

    x1 = Poly([1] * ring3.n, ring3)
    expected1 = transform(x1).tolist()
    y1 = f(x1)
    assert y1.tolist() == expected1
    assert all(unit.root_op is not Op.BUFFER_VIEW for unit in next(iter(f._plans.values())).units)

    x2 = Poly([2] * ring3.n, ring3)
    expected2 = transform(x2).tolist()
    y2 = f(x2)
    assert y1 is not y2  # default fresh-output: view outputs are cloned to independent buffers
    assert y1.tolist() == expected1
    assert y2.tolist() == expected2


def test_jit_replay_preserves_view_base_when_planned():
    """
    Exercise repeated ring views while their backing storage is arena-planned.
    """
    ring = large_prime_ring(64)
    ring3 = type(ring)(n=ring.n, moduli=ring.moduli[:3])
    ring2 = type(ring)(n=ring.n, moduli=ring.moduli[:2])
    ring1 = type(ring)(n=ring.n, moduli=ring.moduli[:1])

    seq: list[tuple[str, int | None]] = [
        ("to_r1", None),
        ("to_r1", None),
        ("rotate", 3),
        ("to_r2", None),
        ("to_r1", None),
        ("to_r1", None),
        ("addself", None),
        ("to_r1", None),
        ("addself", None),
        ("to_r2", None),
        ("contiguous", None),
        ("to_r2", None),
        ("contiguous", None),
        ("addself", None),
        ("contiguous", None),
        ("rotate", 5),
        ("addself", None),
        ("to_r2", None),
        ("to_r3", None),
        ("to_r2", None),
        ("contiguous", None),
        ("addself", None),
        ("rotate", 6),
        ("rotate", 7),
    ]

    def apply_seq(x: Poly) -> Poly:
        y = x
        for op, arg in seq:
            if op == "rotate":
                assert arg is not None
                y = y.rotate(arg)
            elif op == "contiguous":
                y = y.contiguous()
            elif op == "addself":
                y = y + y
            elif op == "to_r1":
                y = y._to_ring(ring1)
            elif op == "to_r2":
                y = y._to_ring(ring2)
            elif op == "to_r3":
                y = y._to_ring(ring3)
            else:  # pragma: no cover
                raise AssertionError(f"unexpected op {op}")
        return y

    @ren.jit
    def f(x: Poly) -> Poly:
        return apply_seq(x)

    x1 = Poly([1] * ring3.n, ring3)
    expected1 = apply_seq(x1).tolist()
    y1 = f(x1)
    assert y1.tolist() == expected1

    x2 = Poly([2] * ring3.n, ring3)
    expected2 = apply_seq(x2).tolist()
    y2 = f(x2)
    assert y2 is not y1  # default fresh-output
    assert y2.tolist() == expected2


def test_jit_fresh_feedback_skips_rebound_output_copy(monkeypatch: pytest.MonkeyPatch):
    ring = large_prime_ring(64)
    jit_module = importlib.import_module("ren.engine.jit")

    @ren.jit
    def step(x: Poly) -> Poly:
        return (x + 1).contiguous()

    x0 = Poly([i for i in range(ring.n)], ring)
    y0 = step(x0)  # capture

    snapshot_count = 0
    real_snapshot = jit_module._snapshot_into

    def tracking_snapshot(dst: Buffer, src: Buffer) -> None:
        nonlocal snapshot_count
        snapshot_count += 1
        real_snapshot(dst, src)

    monkeypatch.setattr(jit_module, "_snapshot_into", tracking_snapshot)

    # The replay output is rebound to fresh storage, so it cannot overwrite y0.
    _ = step(y0)
    assert snapshot_count == 0

    # New unrelated input should not trigger the alias guard.
    fresh = Poly([i + 3 for i in range(ring.n)], ring)
    _ = step(fresh)
    assert snapshot_count == 0


def test_snapshot_into_materializes_view_source(monkeypatch: pytest.MonkeyPatch):
    jit_module = importlib.import_module("ren.engine.jit")
    spec = BufferSpec(dtype=dtypes.uint8, size=4, device="python")
    source = Buffer(spec).ensure_allocated().copyin(memoryview(bytearray([1, 2, 3, 4])))
    view = source.view(2, 1)
    snapshot = Buffer(view.spec)

    monkeypatch.setattr(source._allocator, "supports_transfer", True)
    jit_module._snapshot_into(snapshot, view)

    assert list(snapshot.as_buffer()) == [2, 3]


def test_jit_reused_output_feedback_snapshots_input(monkeypatch: pytest.MonkeyPatch):
    ring = large_prime_ring(64)
    jit_module = importlib.import_module("ren.engine.jit")

    @ren.jit(static_output=True)
    def step(x: Poly) -> Poly:
        return (x + 1).contiguous()

    y = step(Poly([i for i in range(ring.n)], ring))
    snapshot_count = 0
    real_snapshot = jit_module._snapshot_into

    def tracking_snapshot(dst: Buffer, src: Buffer) -> None:
        nonlocal snapshot_count
        snapshot_count += 1
        real_snapshot(dst, src)

    monkeypatch.setattr(jit_module, "_snapshot_into", tracking_snapshot)

    assert step(y) is y
    assert snapshot_count == 1
    assert y.tolist()[:4] == [2, 1, 2, 3]


def test_jit_reused_output_feedback_through_view_snapshots_input(monkeypatch: pytest.MonkeyPatch):
    ring = large_prime_ring(64)
    small_ring = ring[:2]
    jit_module = importlib.import_module("ren.engine.jit")

    @ren.jit(static_output=True)
    def step(x: Poly) -> tuple[Poly, Poly]:
        return (x + x).contiguous(), x._to_ring(small_ring).rotate(1).contiguous()

    y, z = step(Poly([i for i in range(ring.n)], ring).ntt())
    source = y.clone().realize()
    expected_y = (source + source).contiguous().tolist()
    expected_z = source._to_ring(small_ring).rotate(1).contiguous().tolist()
    view = y.node.base.buffer.view(ring.storage_size, 0)
    snapshot_count = 0
    real_snapshot = jit_module._snapshot_into

    def tracking_snapshot(dst: Buffer, src: Buffer) -> None:
        nonlocal snapshot_count
        snapshot_count += 1
        real_snapshot(dst, src)

    monkeypatch.setattr(jit_module, "_snapshot_into", tracking_snapshot)

    out_y, out_z = step(Poly._from_buffer(view, ring, Domain.NTT))
    assert out_y is y and out_z is z
    assert snapshot_count == 1
    assert y.tolist() == expected_y
    assert z.tolist() == expected_z


def test_jit_prune_removes_input_independent_branch():
    ring = large_prime_ring(4)

    def reference(x: Poly) -> Poly:
        c = Poly([1, 2, 3, 4], ring).rotate(1)
        return x + c

    @ren.jit(prune=True)
    def with_prune(x: Poly) -> Poly:
        return reference(x)

    @ren.jit
    def without_prune(x: Poly) -> Poly:
        return reference(x)

    x1 = Poly([9, 8, 7, 6], ring)
    y0 = reference(x1)
    y1 = with_prune(x1)
    without_prune(x1)
    assert y1.tolist() == y0.tolist()
    assert len(next(iter(with_prune._plans.values())).units) < len(next(iter(without_prune._plans.values())).units)

    x2 = Poly([5, 6, 7, 8], ring)
    y0_2 = reference(x2)
    y1_2 = with_prune(x2)
    assert y1_2.tolist() == y0_2.tolist()


def test_jit_prune_replay_buffer_view_chains_match_eager():
    """Black-box regression for prune replay on input-dependent to_ring/view chains."""
    ring = large_prime_ring(64)
    ring3 = type(ring)(n=ring.n, moduli=ring.moduli[:3])
    ring2 = type(ring)(n=ring.n, moduli=ring.moduli[:2])
    ring1 = type(ring)(n=ring.n, moduli=ring.moduli[:1])

    def reference(x: Poly) -> Poly:
        const = Poly([(i * 17 + 3) % ring3.q for i in range(ring3.n)], ring3).rotate(1).contiguous()
        y = (x + const)._to_ring(ring2)._to_ring(ring1)._to_ring(ring2)
        y = y.rotate(3).contiguous()
        return y + y.rotate(1)

    @ren.jit(prune=True)
    def with_prune(x: Poly) -> Poly:
        return reference(x)

    x1 = Poly([(i * 5 + 7) % ring3.q for i in range(ring3.n)], ring3)
    expected1 = reference(x1).tolist()
    y1 = with_prune(x1)
    assert y1.tolist() == expected1

    x2 = Poly([(i * 9 + 11) % ring3.q for i in range(ring3.n)], ring3)
    expected2 = reference(x2).tolist()
    y2 = with_prune(x2)
    assert y2 is not y1  # default fresh-output
    assert y2.tolist() == expected2

    # Also cover input buffers that are themselves produced by view-heavy rewrites.
    x3 = Poly([(i * 13 + 19) % ring3.q for i in range(ring3.n)], ring3)._to_ring(ring2)._to_ring(ring3)
    expected3 = reference(x3).tolist()
    y3 = with_prune(x3)
    assert y3 is not y1  # default fresh-output
    assert y3.tolist() == expected3


def test_jit_prune_replay_liveins_are_pinned():
    """
    Fast regression for prune replay correctness with constant/view-heavy branches.

    Replay reads some buffers that are produced only by pruned-onetime kernels.
    Those live-ins must remain pinned (no replanning/remapping) for replay to
    produce correct results on subsequent calls.
    """
    ring = large_prime_ring(64)
    ring3 = type(ring)(n=ring.n, moduli=ring.moduli[:3])
    ring2 = type(ring)(n=ring.n, moduli=ring.moduli[:2])
    ring1 = type(ring)(n=ring.n, moduli=ring.moduli[:1])

    seq: list[tuple[str, int | None]] = [
        ("to_r2", None),
        ("contiguous", None),
        ("contiguous", None),
        ("to_r1", None),
        ("rotate", 1),
        ("to_r3", None),
        ("to_r1", None),
        ("to_r1", None),
        ("contiguous", None),
        ("to_r1", None),
        ("to_r2", None),
        ("to_r2", None),
        ("addself", None),
        ("to_r1", None),
        ("rotate", 2),
        ("addself", None),
    ]

    def apply_seq(x: Poly) -> Poly:
        y = x
        for op, arg in seq:
            if op == "rotate":
                assert arg is not None
                y = y.rotate(arg)
            elif op == "contiguous":
                y = y.contiguous()
            elif op == "addself":
                y = y + y
            elif op == "to_r1":
                y = y._to_ring(ring1)
            elif op == "to_r2":
                y = y._to_ring(ring2)
            elif op == "to_r3":
                y = y._to_ring(ring3)
            else:  # pragma: no cover
                raise AssertionError(f"unexpected op {op}")
        return y

    def reference(x: Poly) -> Poly:
        c = Poly([i % 11 for i in range(ring3.n)], ring3)
        k = apply_seq(c)
        return (x._to_ring(k.ring) + k)._to_ring(ring3)

    @ren.jit(prune=True)
    def f(x: Poly) -> Poly:
        return reference(x)

    x1 = Poly([1] * ring3.n, ring3)
    x2 = Poly([2] * ring3.n, ring3)
    y1 = f(x1)
    assert y1.tolist() == reference(x1).tolist()
    y2 = f(x2)
    assert y1 is not y2  # default fresh-output
    assert y2.tolist() == reference(x2).tolist()


def test_jit_prune_multi_output_replay_matches_eager():
    """
    Regression for prune replay on multi-output functions with two independent
    constant chains sharing arena storage.  Also exercises the case where the
    input is itself an arena view (e.g. output of a prior memory-planned
    computation like ``ct * ct``), which must not poison the input-view set.
    """
    ring = large_prime_ring(64)
    ring3 = type(ring)(n=ring.n, moduli=ring.moduli[:3])
    ring2 = type(ring)(n=ring.n, moduli=ring.moduli[:2])
    ring1 = type(ring)(n=ring.n, moduli=ring.moduli[:1])

    seq: list[tuple[str, int | None]] = [
        ("to_r3", None),
        ("rotate", 4),
        ("to_r3", None),
        ("contiguous", None),
        ("to_r1", None),
        ("to_r2", None),
        ("rotate", 5),
        ("mulself", None),
        ("rotate", 5),
        ("addself", None),
        ("mulself", None),
        ("rotate", 5),
        ("to_r2", None),
        ("to_r3", None),
        ("to_r2", None),
    ]

    def apply_seq(x: Poly) -> Poly:
        y = x
        for op, arg in seq:
            if op == "rotate":
                assert arg is not None
                y = y.rotate(arg)
            elif op == "contiguous":
                y = y.contiguous()
            elif op == "addself":
                y = y + y
            elif op == "mulself":
                y = y * y
            elif op == "to_r1":
                y = y._to_ring(ring1)
            elif op == "to_r2":
                y = y._to_ring(ring2)
            elif op == "to_r3":
                y = y._to_ring(ring3)
            else:  # pragma: no cover
                raise AssertionError(f"unexpected op {op}")
        return y

    def reference(x: Poly) -> tuple[Poly, Poly]:
        c1 = Poly([i % 11 for i in range(ring3.n)], ring3)
        c2 = Poly([(3 * i + 7) % 13 for i in range(ring3.n)], ring3)
        k1, k2 = apply_seq(c1), apply_seq(c2)
        y1 = (x._to_ring(k1.ring) + k1)._to_ring(ring3)
        y2 = (x._to_ring(k2.ring) * k2)._to_ring(ring3)
        return y1, y2

    @ren.jit(prune=True)
    def f(x: Poly) -> tuple[Poly, Poly]:
        return reference(x)

    # Test with standalone inputs and with arena-view inputs (from a prior
    # memory-planned computation).
    for make_input in [
        lambda v: Poly([v] * ring3.n, ring3),
        lambda v: (Poly([v] * ring3.n, ring3) + Poly([0] * ring3.n, ring3)).contiguous(),
    ]:
        f._plans.clear()  # reset between variants
        x1 = make_input(1)
        x1.realize()
        x2 = make_input(2)
        x2.realize()

        exp1_a, exp1_b = reference(x1)
        out1_a, out1_b = f(x1)
        assert out1_a.tolist() == exp1_a.tolist()
        assert out1_b.tolist() == exp1_b.tolist()

        exp2_a, exp2_b = reference(x2)
        out2_a, out2_b = f(x2)
        assert out2_a.tolist() == exp2_a.tolist()
        assert out2_b.tolist() == exp2_b.tolist()


def test_jit_clear_jit_drops_plans_and_recaptures():
    """clear_jit() drops captured plans and forces re-capture.

    The jitted function takes a non-Poly ``op`` argument that selects
    between two structurally different graphs (addition vs negacyclic
    multiplication). A changed ``op`` captures a second plan, and the
    original ``op`` reuses the first. clear_jit() drops both.
    """
    ring = large_prime_ring(4)

    capture_count = 0

    @ren.jit
    def f(a: Poly, b: Poly, op: Literal["add", "mul"]) -> Poly:
        nonlocal capture_count
        capture_count += 1
        if op == "add":
            return a + b
        elif op == "mul":
            return a * b

    a = Poly([1, 2, 3, 4], ring)
    b = Poly([4, 3, 2, 1], ring)
    expected_add = [5, 5, 5, 5]
    expected_mul = neg_convolve_naive(a.tolist(), b.tolist(), n=ring.n, q=ring.q)

    # First call: captures the "add" branch.
    out1 = f(a, b, "add")
    assert out1.tolist() == expected_add
    assert capture_count == 1
    assert len(f._plans) == 1

    # op="mul" is part of the plan key, so it captures the multiplicative branch.
    assert f(a, b, "mul").tolist() == expected_mul
    assert f(a, b, "add").tolist() == expected_add
    assert capture_count == 2
    assert len(f._plans) == 2

    f.clear_jit()
    assert len(f._plans) == 0

    assert f(a, b, "mul").tolist() == expected_mul
    assert capture_count == 3
    assert len(f._plans) == 1


def test_jit_plans_distinguish_argument_structure():
    ring = large_prime_ring(4)
    c1, c2, c3 = (Poly([value] * ring.n, ring).realize() for value in (1, 10, 100))

    @ren.jit
    def combine(left: list[Poly], right: list[Poly]) -> Poly:
        return sum(left[1:], left[0]) * 2 + sum(right[1:], right[0])

    assert combine([c1, c2], [c3]).tolist()[0] == 122
    assert combine([c1], [c2, c3]).tolist()[0] == 112
    assert combine([c1], [c2, c2]).tolist()[0] == 22

    @ren.jit
    def total(polys: list[Poly]) -> Poly:
        return sum(polys[1:], polys[0])

    assert total([c1, c2]).tolist()[0] == 11
    assert total([c1, c1, c2]).tolist()[0] == 12


def test_jit_plans_specialize_on_values_inside_arguments(monkeypatch: pytest.MonkeyPatch):
    ring = large_prime_ring(4)
    x = Poly([1] * ring.n, ring).realize()

    class Shift:
        def __init__(self, offsets: list[int]):
            self.offsets = offsets

    @ren.jit
    def shift(x: Poly, s: Shift) -> Poly:
        return x + s.offsets[0]

    assert [shift(x, Shift([offset])).tolist()[0] for offset in (1, 2, 1)] == [2, 3, 2]
    assert len(shift._plans) == 2

    monkeypatch.setattr(importlib.import_module("ren.engine.jit"), "MAX_PLANS", 2)
    with pytest.raises(RuntimeError, match=r"s\.offsets\[0\] took 3 values"):
        shift(x, Shift([3]))


def test_prune_tracks_every_output():
    from ren.engine.jit import _prepare_prune

    x, a, b, y = (Node.new_buffer("python", 8, dtypes.uint32) for _ in range(4))
    p0, p1, p2 = (Node.param(i, x) for i in range(3))
    first = Node(Op.CALL, src=(Node.sink(p0.store(p1), p2.store(p1)), a, x, b))
    second = Node(Op.CALL, src=(Node.sink(p0.store(p1)), y, b))
    linear = Node(Op.LINEAR, src=(first, second))

    assert _prepare_prune(linear, [x.buffer]) is linear


def test_prune_preserves_nonfirst_constant_output(monkeypatch):
    from ren.engine.jit import _prepare_prune
    from ren.engine.memory import plan_memory

    x, a, b, y = (Node.new_buffer("python", 8, dtypes.uint32) for _ in range(4))
    p0, p1, p2 = (Node.param(i, x) for i in range(3))
    first = Node(Op.CALL, src=(Node.sink(p0.store(Node.const(1)), p1.store(Node.const(2))), a, b))
    second = Node(Op.CALL, src=(Node.sink(p0.store(p1 + p2)), y, x, b))
    linear = Node(Op.LINEAR, src=(first, second))

    def lower(onetime):
        assert onetime.src == (first,)

        def constants(bufs):
            for value, buf in enumerate(bufs, 1):
                buf.copyin(memoryview(bytes([value]) * buf.nbytes))

        return [ExecUnit(constants, first)]

    monkeypatch.setattr(importlib.import_module("ren.engine.jit"), "make_exec_units", lower)
    replay = _prepare_prune(linear, [x.buffer])
    assert replay.src == (second,)
    assert bytes(b.buffer.as_buffer()) == bytes([2]) * b.buffer.nbytes
    # Preparation precedes memory planning: every constant output is allocated
    # and therefore excluded from the replay arena, without a snapshot copy.
    assigned = plan_memory(replay, {y.buffer})
    assert a.buffer not in assigned and b.buffer not in assigned


def test_jit_bound_method_keeps_owner_alive():
    ring = large_prime_ring(4)

    class Model:
        @ren.jit
        def forward(self, x):
            return x + 1

    owner = Model()
    assert Model.forward(self=owner, x=Poly([1, 2, 3, 4], ring)).tolist() == [2, 2, 3, 4]
    owner_ref = weakref.ref(owner)
    forward = owner.forward
    del owner
    gc.collect()
    assert owner_ref() is not None
    assert forward(Poly([1, 2, 3, 4], ring)).tolist() == [2, 2, 3, 4]
    plan_ref = weakref.ref(next(iter(forward._plans.values())))
    del forward
    gc.collect()
    assert owner_ref() is None
    assert plan_ref() is None


def test_jit_class_namespace_reuses_one_plan():
    ring = large_prime_ring(4)
    captures = []

    class Ops:
        @ren.jit
        def double(x: Any):
            captures.append(None)
            return x + x

    inputs = [Poly([i, 2, 3, 4], ring) for i in range(5)]
    for i, x in enumerate(inputs):
        assert Ops.double(x).tolist() == [2 * i, 4, 6, 8]
    assert len(captures) == 1
    assert not Ops.double._bound_methods
    Ops.double.clear_jit()
    assert Ops.double(inputs[0]).tolist() == [0, 4, 6, 8]
    assert len(captures) == 2


def test_jit_scope_reuses_plans_and_retains_owners():
    class Owner:
        pass

    ring = large_prime_ring(4)
    x = Poly([1, 2, 3, 4], ring).realize()
    captures = []

    @ren.jit
    def step(x):
        captures.append(factor)
        return x * factor

    owners = []
    for factor in (2, 3, 2):
        owner = Owner()
        owners.append(weakref.ref(owner))
        with ren.jit.scope(factor, owner=owner):
            assert step(x).tolist() == [v * factor for v in (1, 2, 3, 4)]
    del owner
    gc.collect()
    assert captures == [2, 3]
    assert owners[0]() is not None and owners[1]() is not None
    step.clear_jit()
    gc.collect()
    assert all(owner() is None for owner in owners)


def test_jit_method_plans_belong_to_instance():
    ring = large_prime_ring(4)
    captures = []

    class AddConstant:
        def __init__(self, value: int):
            self.value = value

        @ren.jit
        def run(model, x: Poly) -> Poly:
            captures.append(model.value)
            return x + model.value

    first, second = AddConstant(1), AddConstant(2)
    x = Poly([0, 0, 0, 0], ring).realize()
    assert assert_type(AddConstant.run(model=first, x=x), Poly).tolist() == [1, 0, 0, 0]
    assert AddConstant.run(model=second, x=x).tolist() == [2, 0, 0, 0]
    assert assert_type(first.run(x), Poly).tolist() == [1, 0, 0, 0]
    assert AddConstant.run(first, x).tolist() == [1, 0, 0, 0]
    assert second.run(x).tolist() == [2, 0, 0, 0]
    assert captures == [1, 2]

    first.run.clear_jit()
    assert first.run(x).tolist() == [1, 0, 0, 0]
    assert second.run(x).tolist() == [2, 0, 0, 0]
    assert captures == [1, 2, 1]
    AddConstant.run.clear_jit()
    assert AddConstant.run(first, x).tolist() == [1, 0, 0, 0]
    assert second.run(x).tolist() == [2, 0, 0, 0]
    assert captures == [1, 2, 1, 1, 2]

    plan_ref = weakref.ref(next(iter(first.run._plans.values())))
    del first
    gc.collect()
    assert plan_ref() is None


@pytest.mark.parametrize("kwargs", [{}, {"factor": 3}], ids=["poly-order", "scalar-keyword"])
def test_jit_method_call_styles_share_argument_bindings(kwargs):
    ring = large_prime_ring(4)
    captures = []

    class Model:
        @ren.jit
        def run(model, z: Poly, a: Poly, factor=2) -> Poly:
            captures.append(None)
            return z + factor * a

    owner = Model()
    for i in (1, 2):
        z, a = Poly([i, 0, 0, 0], ring), Poly([i + 2, 0, 0, 0], ring)
        expected = [i + kwargs.get("factor", 2) * (i + 2), 0, 0, 0]
        assert owner.run(z=z, a=a, **kwargs).tolist() == expected
        assert Model.run(owner, z=z, a=a, **kwargs).tolist() == expected
        assert Model.run(model=owner, z=z, a=a, **kwargs).tolist() == expected
    assert len(captures) == 1


@pytest.mark.parametrize("slots", [False, True], ids=["dict", "slots"])
def test_jit_method_rebinds_owner_polys(slots):
    ring = large_prime_ring(4)
    captures = []

    @dataclass(slots=slots, weakref_slot=slots)
    class AddPoly:
        bias: Poly

        @ren.jit
        def run(self, x: Poly) -> Poly:
            captures.append(None)
            return x + self.bias

    owner = AddPoly(Poly([1, 2, 3, 4], ring))
    x = Poly([1, 1, 1, 1], ring)
    first = owner.run(x)
    owner.bias = Poly([4, 3, 2, 1], ring)
    second = owner.run(x)
    assert first.tolist() == [2, 3, 4, 5]
    assert second.tolist() == [5, 4, 3, 2]
    assert len(captures) == 1
    assert ren.PolyContainer.polys(owner) == [owner.bias]


@pytest.mark.parametrize("enabled", [True, False])
def test_jit_gc_scope_covers_capture_cleanup_and_replay(monkeypatch: pytest.MonkeyPatch, enabled: bool):
    jit_module = importlib.import_module("ren.engine.jit")
    states = []

    class CaptureLocal:
        def __del__(self):
            states.append(gc.isenabled())

    capture = ren.jit._capture
    replay = jit_module.CapturedPlan.replay

    def capture_with_local(*args, **kwargs):
        _local = CaptureLocal()
        return capture(*args, **kwargs)

    def checked_replay(*args, **kwargs):
        states.append(gc.isenabled())
        return replay(*args, **kwargs)

    monkeypatch.setattr(ren.jit, "_capture", capture_with_local)
    monkeypatch.setattr(jit_module.CapturedPlan, "replay", checked_replay)
    x = Poly([1, 2, 3, 4], large_prime_ring(4)).realize()
    step = ren.jit(lambda x, factor: x * factor)
    was_enabled = gc.isenabled()
    try:
        (gc.enable if enabled else gc.disable)()
        for _ in range(2):
            assert step(x, 2).tolist() == [2, 4, 6, 8]
            assert gc.isenabled() is enabled
        assert step(x, 3).tolist() == [3, 6, 9, 12]
        assert gc.isenabled() is enabled
        assert states == [False, False, False]
    finally:
        (gc.enable if was_enabled else gc.disable)()
