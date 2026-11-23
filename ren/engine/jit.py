import atexit
import weakref
from collections.abc import Callable, Generator, Hashable, Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import InitVar, dataclass, field
from inspect import signature
from typing import Any, Concatenate, Protocol, Self, overload
from types import MethodType

from ren import tree
from ren.container import PolyContainer
from ren.backend import GraphRunner
from ren.buffer import Buffer
from ren.context import DEBUG, GRAPH
from ren.device import Device
from ren.dtype import DType
from ren.engine.memory import plan_memory
from ren.engine.realize import ExecUnit, call_buffers, capturing, make_exec_units, rewrite_buffers
from ren.ir import Node, Op
from ren.kernel import get_output_slots
from ren.poly import Poly, realize
from ren.prof import trace
from ren.ring import Domain, RingSpec
from ren.utils import disable_gc, unwrap

MAX_PLANS = 16
_JIT_CONTEXT: ContextVar[tuple[Hashable, object | None]] = ContextVar("JIT_CONTEXT", default=(None, None))


@dataclass(frozen=True, slots=True)
class RealizedArgSpec:
    ring: RingSpec
    size: int
    dtype: DType
    device: str
    domain: Domain | None
    buffer_slot: int


ArgSpec = RealizedArgSpec | Node


def _specs_and_bufs(arg_polys: list[Poly]) -> tuple[tuple[ArgSpec, ...], list[Buffer]]:
    specs: list[ArgSpec] = []
    input_bufs: list[Buffer] = []
    buffer_slots: dict[Buffer, int] = {}
    for poly in arg_polys:
        node = poly.node
        base = node.base
        if (realized := base.realized) is None:
            if poly.device is None:
                specs.append(node)
                continue
            realize(*arg_polys)
            if poly.device is not None and not poly.is_realized:
                raise RuntimeError("ren.jit: input Poly did not realize")
            return _specs_and_bufs(arg_polys)
        if (buffer_slot := buffer_slots.get(realized)) is None:
            buffer_slot = buffer_slots[realized] = len(input_bufs)
            input_bufs.append(realized)
        specs.append(RealizedArgSpec(poly.ring, realized.size, base.dtype, base.device, poly.domain, buffer_slot))
    return tuple(specs), input_bufs


type _ArgKey = tuple[Hashable, tuple[Hashable, ...]]
type _PlanKey = tuple[tuple[ArgSpec, ...], tuple[_ArgKey, ...], Hashable, str | None]


def _varying_value(arguments: dict[str, object], plan_key: _PlanKey, plan_keys: Iterable[_PlanKey]) -> str:
    """Name the argument value that took the most distinct values across plans with one structure."""
    structures = [structure for structure, _ in plan_key[1]]
    rows = [
        [value for _, values in key[1] for value in values]
        for key in (*plan_keys, plan_key)
        if key[2:] == plan_key[2:] and [structure for structure, _ in key[1]] == structures
    ]
    counts = [len(set(column)) for column in zip(*rows)]
    if not counts or max(counts) < 2:
        return ""
    paths = [path for name, value in arguments.items() for path in tree.paths(value, leaf=Poly, root=name)]
    index = max(range(len(counts)), key=counts.__getitem__)
    return f"; {paths[index]} took {counts[index]} values. Pass values that vary between calls as Poly arguments"


def _snapshot_into(dst: Buffer, src: Buffer) -> None:
    """Copy data from *src* into *dst*, using D2D transfer when available."""
    assert dst.device == src.device and dst.nbytes == src.nbytes
    src.ensure_allocated()
    dst.ensure_allocated()
    dev = Device[src.device]
    if dev.allocator.supports_transfer:
        dev.allocator.transfer(dst._buf, src._buf, src.nbytes, dev, dev)
    else:
        dst.copyin(src.as_buffer())


def _prepare_prune(linear: Node, bindings: list[Buffer]) -> Node:
    """Execute input-independent calls once and return the replay calls."""
    changing = {buf.base for buf in bindings}
    replay, onetime = [], []
    for call in linear.src:
        bufs = call_buffers(call)
        if any(buf.base in changing for buf in bufs):
            changing.update(bufs[slot].base for slot in get_output_slots(call.src[0]))
            replay.append(call)
        else:
            onetime.append(call)
    for unit in make_exec_units(linear.replace(src=tuple(onetime))):
        unit()
    return linear.replace(src=tuple(replay))


@dataclass(slots=True)
class _OutputPlan:
    """Replay output construction and reserved storage with weakly tracked borrowers."""

    pool: dict[Buffer, weakref.ReferenceType[Buffer] | None]
    build: Callable[[list[Buffer]], Any]

    def acquire(self) -> list[Buffer]:
        outputs = []
        for storage, borrower in self.pool.items():
            if borrower is None:
                storage.ensure_allocated()
            elif borrower() is not None:
                outputs.append(Buffer(storage.spec).allocate())
                continue

            output = _OutputBuffer(storage)
            self.pool[storage] = weakref.ref(output)
            outputs.append(output)
        return outputs

    @staticmethod
    def normalize(out_polys: list[Poly], input_bufs: list[Buffer], capture_calls: list[Node]) -> list[Poly]:
        """Clone input aliases, views, oversized outputs, and buffers read through views.

        Capture realizes these clones into output-sized storage. Each replay
        copies into fresh buffers.
        """
        input_bases = {buf.base for buf in input_bufs}
        viewed = {buf._base for call in capture_calls for buf in call_buffers(call) if buf._base is not None}
        clones: dict[tuple[Buffer, RingSpec, Domain | None], Poly] = {}
        unowned: dict[int, Poly] = {}
        for p in {id(poly): poly for poly in out_polys}.values():
            if p.node.base.op not in {Op.BUFFER, Op.BUFFER_VIEW}:
                continue
            buf = p.node.base.buffer
            if p.node.base.op is Op.BUFFER and buf.base not in input_bases and p.ring.nbytes == buf.base.nbytes and buf not in viewed:
                continue
            key = (buf, p.ring, p.domain)
            if (clone := clones.get(key)) is None:
                unowned[id(p)] = clones[key] = p.clone()
            else:
                unowned[id(p)] = Poly(clone.node)
        if not unowned:
            return out_polys
        return [unowned.get(id(poly), poly) for poly in out_polys]

    @classmethod
    def capture(cls, spec: tree.Spec[Any], out_polys: list[Poly], written: set[Buffer]) -> tuple[_OutputPlan, list[Buffer]]:
        slots: dict[Buffer, int] = {}
        outputs: list[tuple[int, int, RingSpec, Domain]] = []
        initial: list[Poly | None] = [None] * len(out_polys)
        aliases: list[tuple[int, int]] = []
        first: dict[int, int] = {}
        for destination, p in enumerate(out_polys):
            if (source := first.get(id(p))) is not None:
                initial[destination] = initial[source]
                if initial[source] is None:
                    aliases.append((destination, source))
                continue
            first[id(p)] = destination
            if p.node.base.op is not Op.BUFFER or (b := p.node.base.buffer) not in written:
                initial[destination] = p
                continue
            if (slot := slots.get(b)) is None:
                slot = slots[b] = len(slots)
            outputs.append((destination, slot, p.ring, unwrap(p.domain)))

        from_buffer = Poly._from_buffer

        def build(buffers: list[Buffer]) -> Any:
            values = initial.copy()
            for destination, slot, ring, domain in outputs:
                values[destination] = from_buffer(buffers[slot], ring, domain)
            for destination, source in aliases:
                values[destination] = values[source]
            return spec.unflatten(values)

        return cls({Buffer(buf.spec): None for buf in slots}, build), list(slots)


class _OutputBuffer(Buffer):
    __slots__ = ("_storage",)

    def __init__(self, storage: Buffer):
        super().__init__(storage.spec)
        self._storage = storage
        self._buf = storage._buf

    def deallocate(self):
        assert self._buf is not None, "buffer must be allocated to deallocate"
        del self._buf, self._storage


@dataclass(slots=True, eq=False, weakref_slot=True)
class CapturedPlan:
    """Parameterized replay plan for one argument and context specialization."""

    units: tuple[ExecUnit, ...]
    bindings: InitVar[list[Buffer]]
    output_plan: _OutputPlan
    reused_output_bases: set[Buffer]
    owner: object | None = None  # Keeps resources referenced by the specialization key alive.
    runner: GraphRunner | None = field(init=False, default=None)

    def __post_init__(self, bindings: list[Buffer]) -> None:
        """Build backend-specific runner for single-device captures when available."""
        if self.units and GRAPH:
            devices = {unit.device for unit in self.units}
            if len(devices) != 1:
                if DEBUG >= 1:
                    print(f"graph runner: rejected, multi-device {devices}")
            elif (graph := Device[next(iter(devices))].graph) is not None:
                try:
                    self.runner = graph(self.units, bindings)
                except (ValueError, RuntimeError) as exc:
                    if DEBUG >= 1:
                        print(f"graph runner: rejected, {exc}")
        if self.runner is not None:
            if DEBUG >= 1:
                copies = sum(unit.root_op is Op.COPY for unit in self.units)
                print(f"graph runner: created with {len(self.units)} units ({copies} copies)")
            _CAPTURED_PLANS.add(self)

    def replay(self, bindings: list[Buffer]) -> Any:
        """Resolve replay inputs and re-run this captured plan."""
        with trace("jit.replay"):
            if self.reused_output_bases:
                for input_idx, input_buf in enumerate(bindings):
                    if input_buf.base not in self.reused_output_bases:
                        continue
                    snapshot = Buffer(input_buf.spec)
                    _snapshot_into(snapshot, input_buf)
                    bindings[input_idx] = snapshot

            output_bufs = self.output_plan.acquire()
            bindings.extend(output_bufs)
            if self.runner is not None:
                self.runner(bindings)
            else:
                for unit in self.units:
                    unit(bindings)
            return self.output_plan.build(output_bufs)

    def close(self) -> None:
        _CAPTURED_PLANS.discard(self)
        if self.runner is not None:
            self.runner.close()
            self.runner = None
        self.output_plan.pool.clear()


_CAPTURED_PLANS: weakref.WeakSet[CapturedPlan] = weakref.WeakSet()
atexit.register(lambda: [plan.close() for plan in list(_CAPTURED_PLANS)])


class _Jitted[**P, R](Protocol):
    _plans: dict[_PlanKey, CapturedPlan]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...
    def clear_jit(self) -> None: ...


class jit[**P, R]:
    """Capture one parameterized execution plan per argument shape and replay it.

    Decorate a callable with ``@jit`` or ``@jit(prune=True)``.

    Each replay returns fresh output buffers, so a still-live previous result --
    or a lazy clone of it -- is never overwritten by the next call.
    ``static_output=True`` skips the output rebuild and rewrites the same object
    in place when the caller can consume each result before the next replay."""

    @overload
    def __new__(cls, fn: Callable[P, R], *, prune: bool = False, static_output: bool = False) -> Self: ...

    @overload
    def __new__[**Q, S](cls, fn: None = None, *, prune: bool = False, static_output: bool = False) -> Callable[[Callable[Q, S]], jit[Q, S]]: ...

    def __new__(cls, fn=None, *, prune: bool = False, static_output: bool = False):
        if fn is None:

            def decorate[**Q, S](fn: Callable[Q, S]) -> jit[Q, S]:
                if not callable(fn):
                    raise TypeError("jit decorator expected a function")
                return jit(fn, prune=prune, static_output=static_output)

            return decorate
        return super().__new__(cls)

    def __init__(self, fn: Callable[P, R], *, prune: bool = False, static_output: bool = False):
        if not callable(fn):
            raise TypeError("jit decorator expected a function")
        self.fn = fn
        self.prune = prune
        self.static_output = static_output
        self._plans: dict[_PlanKey, CapturedPlan] = {}
        self._bound_methods: dict[int, tuple[weakref.ReferenceType, Self]] = {}
        self._owner_type: type | None = None
        self._signature = signature(fn)

    @staticmethod
    @contextmanager
    def scope(key: Hashable, *, owner: object | None = None) -> Generator[None]:
        """Specialize JIT calls in this scope and retain owner in captured plans."""
        with _JIT_CONTEXT.set((key, owner)):
            yield

    def __set_name__(self, owner: type, _name: str) -> None:
        self._owner_type = owner

    @overload
    def __get__(self, obj: None, _=None) -> Self: ...

    @overload
    def __get__[T, **Q](self: Callable[Concatenate[T, Q], R], obj: T, _=None) -> _Jitted[Q, R]: ...

    def __get__(self, obj, _=None):
        # Class-level access returns the descriptor so callers can clear all instance plans.
        if obj is None:
            return self
        bound_methods = self._bound_methods
        oid = id(obj)
        if (entry := bound_methods.get(oid)) is None:
            try:
                obj_ref = weakref.ref(obj, lambda _: bound_methods.pop(oid))
            except TypeError as exc:
                raise TypeError("ren.jit: jitted method instances must support weak references") from exc
            bound = type(self)(self.fn, prune=self.prune, static_output=self.static_output)
            bound_methods[oid] = entry = (obj_ref, bound)
        return MethodType(entry[1], obj)

    def clear_jit(self) -> None:
        """Drop all captured plans and release backend runner resources.

        After calling this, the next invocation of the jit-decorated function will
        re-capture from scratch. Raises if called while a capture is in progress.
        """
        instances = (self, *(bound for _, bound in self._bound_methods.values()))
        if any(instance in capturing for instance in instances):
            raise RuntimeError("ren.jit: clear_jit() called during capture")
        for instance in instances:
            for plan in instance._plans.values():
                plan.close()
            instance._plans.clear()

    def _capture(
        self,
        args: tuple,
        kwargs: dict,
        plan_key: _PlanKey,
        input_bufs: list[Buffer],
        owner: object | None,
        arguments: dict[str, object],
    ) -> Any:
        """Capture and build the replay plan for one argument specialization."""
        if len(self._plans) >= MAX_PLANS:
            raise RuntimeError(f"ren.jit: too many captured plans ({len(self._plans)}){_varying_value(arguments, plan_key, self._plans)}")
        with trace("jit.capture"):
            assert not capturing, "ren.jit: nested capture invariant violated"
            calls: list[Node] = []
            capturing[self] = calls
            try:
                ret = self.fn(*args, **kwargs)

                out_polys, output_spec = PolyContainer.flatten(ret)
                if not out_polys:
                    raise TypeError("ren.jit: outputs must contain at least one Poly")

                realize(*out_polys)
                if (normalized := _OutputPlan.normalize(out_polys, input_bufs, calls)) is not out_polys:
                    realize(*normalized)
                    out_polys = normalized
                    ret = output_spec.unflatten(out_polys)
            finally:
                del capturing[self]

            linear = Node(Op.LINEAR, src=tuple(calls))

            if self.prune:
                linear = _prepare_prune(linear, input_bufs)

            output_bases = {p.node.base.buffer.base for p in out_polys if p.node.base.op is Op.BUFFER}

            assigned = plan_memory(linear, output_bases | {buf.base for buf in input_bufs})
            written: set[Buffer] = set()
            for call in linear.src:
                bufs = call_buffers(call)
                written.update(bufs[slot] for slot in get_output_slots(call.src[0]))

            if self.static_output:
                output_plan = _OutputPlan(pool={}, build=lambda _: ret)
                output_bufs: list[Buffer] = []
                reused_output_bases = output_bases & {buf.base for buf in written}
            else:
                output_plan, output_bufs = _OutputPlan.capture(output_spec, out_polys, written)
                reused_output_bases = set()

            for buf in output_bufs:
                if buf in input_bufs:
                    raise RuntimeError("ren.jit: output buffer still doubles as an input")
            capture_bindings = input_bufs + output_bufs
            binding_slots = {buf: slot for slot, buf in enumerate(capture_bindings)}
            replacement_nodes: dict[Buffer, Node] = {}

            def replace_buffer(node: Node, buf: Buffer) -> Node | None:
                if (slot := binding_slots.get(buf)) is not None:
                    return Node.param(slot, node)
                if (replacement := assigned.get(buf)) is None:
                    return None
                if (replacement_node := replacement_nodes.get(replacement)) is None:
                    replacement.ensure_allocated()
                    replacement_node = replacement_nodes[replacement] = Node.from_buffer(replacement)
                return replacement_node

            linear = rewrite_buffers(linear, replace_buffer)
            units = tuple(make_exec_units(linear))

            plan = CapturedPlan(units, capture_bindings, output_plan, reused_output_bases, owner)

            self._plans[plan_key] = plan
            return ret

    @disable_gc()
    def __call__(self, /, *args: P.args, **kwargs: P.kwargs) -> R:
        if self._owner_type is not None:
            bound = self._signature.bind(*args, **kwargs)
            if bound.args and isinstance(bound.args[0], self._owner_type):
                if not args:
                    kwargs.pop(next(iter(bound.arguments)))
                return self.__get__(bound.args[0])(*args[1:], **kwargs)

        if capturing:
            raise RuntimeError("ren.jit: nested jit is not supported")

        # TODO: Reject Polys hidden in subclasses of tree's opaque scalar types, or require explicit static opt-in.
        arguments = self._signature.bind(*args, **kwargs).arguments
        arg_polys: list[Poly] = []
        arg_keys: list[_ArgKey] = []
        for value in arguments.values():
            polys, key = tree.key(value, leaf=Poly)
            arg_polys += polys
            arg_keys.append(key)
        if not arg_polys:
            raise RuntimeError("ren.jit: function must receive at least one Poly argument")
        arg_specs, input_bufs = _specs_and_bufs(arg_polys)
        context_key, owner = _JIT_CONTEXT.get()
        plan_key = (arg_specs, tuple(arg_keys), context_key, Device._override)

        if (plan := self._plans.get(plan_key)) is not None:
            return plan.replay(input_bufs)
        return self._capture(args, kwargs, plan_key, input_bufs, owner, arguments)
