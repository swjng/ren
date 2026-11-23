"""Stored-state traversal and reusable reconstruction plans.

Records expose stored slots and instance dictionaries, not properties or serialization
hooks. Classes other than scalars and arrays can instead define __tree_reduce__(self)
returning (rebuild, children), where children is a tuple of values to traverse and
rebuild receives one replacement per child, in order. Supported collections are list,
tuple, namedtuple, set, frozenset, dict and MappingProxyType. Arrays retain identity;
object arrays may contain only supported scalars. Cycles and unsupported objects
raise. Layouts cache neither instances nor their leaf counts.
"""

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache, partial
from itertools import islice
from numbers import Number
from operator import itemgetter
from pathlib import Path
from types import FunctionType, GetSetDescriptorType, MappingProxyType, MemberDescriptorType
from typing import Any, cast

import numpy as np
from flint import acb, arb

_SCALARS = (Number, str, bytes, bytearray, range, Path, Enum, np.generic, acb, arb, type(None), type)
_DISALLOW_INSTANTIATION = 1 << 7  # CPython's Py_TPFLAGS_DISALLOW_INSTANTIATION.
type _Rebuild = Callable[[Sequence[Any]], Any]
type _Read = Callable[[Sequence[Any]], Sequence[Any]]
type _Memo[T] = dict[int, tuple[object, T | None]]  # Sources stay alive until the walk ends.


@dataclass(frozen=True, slots=True)
class _Layout:
    """Class operations only; decompose returning None means the value was validated as opaque."""

    children: Callable[[Any], Iterable[Any] | None]
    decompose: Callable[[Any], tuple[Iterable[Any], _Rebuild] | None]
    names: Callable[[Any], Iterable[str]] | None = None  # Path labels in children order; None labels by index.


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Spec[Root]:
    """An immutable plan; each unflatten creates fresh structural parents."""

    count: int
    template: tuple[Any, ...]  # Negative slots index this reversed template; selected leaves index from the front.
    steps: tuple[tuple[int, _Rebuild, _Read], ...]  # (target slot, rebuild, child reader), in dependency order.

    def unflatten(self, leaves: Iterable[object]) -> Root:
        values: list[Any] = [*islice(leaves, self.count + 1), *self.template]
        if len(values) != self.count + len(self.template):
            raise ValueError("incorrect number of replacement leaves")
        for slot, rebuild, read in self.steps:
            values[slot] = rebuild(read(values))
        return values[self.count if self.template else 0]


def collect[Leaf](*roots: object, leaf: type[Leaf]) -> list[Leaf]:
    """Collect selected positions in storage order, visiting shared containers once."""
    leaves, seen = [], {}
    _collect(roots, leaves, seen, leaf)
    return leaves


def extract[Root, Leaf](value: Root, *, leaf: type[Leaf]) -> tuple[list[Leaf], Spec[Root]]:
    """Return selected values and a reusable reconstruction plan that does not retain them.

    Plans keep each __tree_reduce__ rebuild, so it must not reference its source
    instance or selected values.
    """
    leaves, template, steps = [], [], []
    _capture(value, leaves, template, steps, {}, leaf)
    return leaves, Spec(len(leaves), tuple(reversed(template)), tuple(steps))


def replace[Root, Leaf](value: Root, fn: Callable[[Leaf], Leaf], *, leaf: type[Leaf]) -> Root:
    """Apply fn per selected position and rebuild changed parents.

    Replacements are terminal. The callback must not mutate the input. Reconstruction
    bypasses constructors, properties and copy/pickle hooks unless __tree_reduce__
    rebuilds. Recursion is bounded by Python's stack limit. The result never mutates
    the input or prior outputs.
    """
    return cast(Root, fn(value)) if isinstance(value, leaf) else _replace(value, fn, {}, leaf)


def key[Leaf](value: object, *, leaf: type[Leaf]) -> tuple[list[Leaf], tuple[Hashable, tuple[Hashable, ...]]]:
    """Return selected values and a hashable key for everything else.

    The key is (structure, values). Structure records classes, mapping keys and shared
    containers. Values snapshot each opaque value in traversal order, so mutating an
    array or bytearray after the call does not change the key.
    """
    leaves, values = [], []
    structure = _key(value, leaves, values, None, "", {}, leaf)
    return leaves, (structure, tuple(values))


def paths[Leaf](value: object, *, leaf: type[Leaf], root: str = "") -> list[str]:
    """Return the path of each value in key(value), in the same order."""
    result: list[str] = []
    _key(value, [], [], result, root, {}, leaf)
    return result


def _snapshot(value: object) -> Hashable:
    if isinstance(value, (np.ndarray, np.void)):
        return value.dtype.str, value.shape, tuple(value.flat) if value.dtype.hasobject else value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    return cast(Hashable, value)


def _key[Leaf](
    value: object, leaves: list[Leaf], values: list[Hashable], paths: list[str] | None, path: str, memo: _Memo[int], leaf: type[Leaf]
) -> Hashable:
    if isinstance(value, leaf):
        leaves.append(value)
        return leaf
    cls = type(value)
    layout = _layout(cls, leaf)
    if layout is None or (children := layout.children(value)) is None:
        values.append(_snapshot(value))
        if paths is not None:
            paths.append(path)
        return cls
    if (oid := id(value)) in memo:
        if (index := memo[oid][1]) is None:
            raise ValueError("containers cannot contain cycles")
        return index
    memo[oid] = value, None
    keys = tuple(sorted(cast(Mapping, value))) if cls in (dict, MappingProxyType) else None
    if paths is None:
        result = cls, keys, tuple([_key(child, leaves, values, None, path, memo, leaf) for child in children])
    else:
        children = tuple(children)
        names = layout.names(value) if layout.names is not None else (f"[{index}]" for index in range(len(children)))
        result = cls, keys, tuple(_key(child, leaves, values, paths, path + name, memo, leaf) for child, name in zip(children, names))
    memo[oid] = value, len(memo)
    return result


def _array(value: np.ndarray | np.void, leaf: type) -> None:
    if value.dtype.hasobject:
        if isinstance(value, np.void):
            raise TypeError("NumPy object arrays may only contain scalars")
        if value.dtype.names:
            for name in value.dtype.names:
                _array(value[name], leaf)
        elif any(not issubclass(cls, _SCALARS) or issubclass(cls, (leaf, np.void)) for cls in set(map(type, value.flat))):
            raise TypeError("NumPy object arrays may only contain scalars")


@lru_cache(maxsize=256)
def _layout(cls: type, leaf: type) -> _Layout | None:
    if cls in (list, tuple, set, frozenset):
        return _Layout(iter, lambda value: (tuple(value), cls))
    if cls in (dict, MappingProxyType):

        def keys(value):
            if collect(*value, leaf=leaf):
                raise TypeError(f"mapping keys cannot contain {leaf.__name__}s")
            return sorted(value)

        def decompose_mapping(value):
            ordered = keys(value)
            indices = dict(zip(ordered, range(len(ordered))))
            order = tuple((key, indices[key]) for key in value)

            def restore(values):
                result = {key: values[index] for key, index in order}
                return result if cls is dict else cls(result)

            return tuple(value[key] for key in ordered), restore

        return _Layout(lambda value: map(value.__getitem__, keys(value)), decompose_mapping, lambda value: (f"[{key!r}]" for key in keys(value)))
    if issubclass(cls, (np.ndarray, np.void)):
        return _Layout(partial(_array, leaf=leaf), partial(_array, leaf=leaf))
    if cls is object or issubclass(cls, _SCALARS):
        return None
    if (reduce := getattr(cls, "__tree_reduce__", None)) is not None:

        def decompose_reduced(value):
            rebuild, children = reduce(value)
            return children, lambda values: rebuild(*values)

        return _Layout(lambda value: reduce(value)[1], decompose_reduced)
    if issubclass(cls, tuple) and isinstance(getattr(cls, "_fields", None), tuple) and not cls.__dictoffset__:
        restore = partial(tuple.__new__, cls)
        labels = tuple(f".{name}" for name in getattr(cls, "_fields"))
        return _Layout(tuple.__iter__, lambda value: (tuple.__iter__(value), restore), lambda _: labels)
    if issubclass(cls, (list, tuple, set, frozenset, Mapping)) or any(
        base.__flags__ & _DISALLOW_INSTANTIATION or (base.__new__ is not object.__new__ and not isinstance(base.__new__, FunctionType))
        for base in cls.__mro__
    ):
        raise TypeError(f"{cls.__name__} is not a supported storage type")
    slots, slot_names = [], []
    for base in reversed(cls.__mro__):
        names = vars(base).get("__slots__", ())
        if not isinstance(names, (str, tuple, list, dict)):
            raise TypeError(f"{base.__name__} has an unsupported __slots__ declaration")
        for name in (names,) if isinstance(names, str) else names:
            label = name
            if name.startswith("__") and not name.endswith("__"):
                name = f"_{base.__name__.lstrip('_')}{name}"
            if isinstance(member := vars(base).get(name), MemberDescriptorType):
                slots.append(member)
                slot_names.append(label)
    dictionary = next((vars(base)["__dict__"] for base in cls.__mro__ if isinstance(vars(base).get("__dict__"), GetSetDescriptorType)), None)

    accessors = tuple((slot.__get__, slot.__set__) for slot in slots)

    def children(value):
        for get, _ in accessors:
            try:
                yield get(value)
            except AttributeError:
                continue
        if dictionary is not None:
            yield from dictionary.__get__(value).values()

    def names(value):
        for (get, _), name in zip(accessors, slot_names):
            try:
                get(value)
            except AttributeError:
                continue
            yield f".{name}"
        if dictionary is not None:
            yield from (f".{name}" for name in dictionary.__get__(value))

    def restore_record(state, data, values):
        result = object.__new__(cls)
        for put, value in zip(state, values):
            put(result, value)
        if dictionary is not None:
            dictionary.__get__(result).update(zip(data, values[len(state) :]))
        return result

    def decompose_record(value):
        state = {}
        for get, put in accessors:
            try:
                state[put] = get(value)
            except AttributeError:
                continue  # Uninitialized slots have no stored value.
        data = dictionary.__get__(value) if dictionary is not None else {}
        return (*state.values(), *data.values()), partial(restore_record, tuple(state), tuple(data))

    return _Layout(children, decompose_record, names)


def _collect[Leaf](values: Iterable[object], leaves: list[Leaf], seen: dict[int, object | None], leaf: type[Leaf]) -> None:
    for value in values:
        if isinstance(value, leaf):
            leaves.append(value)
        elif (layout := _layout(type(value), leaf)) is not None:
            if (oid := id(value)) in seen:
                if seen[oid] is None:
                    raise ValueError("containers cannot contain cycles")
                continue
            seen[oid] = None
            _collect(layout.children(value) or (), leaves, seen, leaf)
            seen[oid] = value


def _capture[Leaf](
    value: object, leaves: list[Leaf], template: list[Any], steps: list[tuple[int, _Rebuild, _Read]], memo: _Memo[int], leaf: type[Leaf]
) -> int:
    if isinstance(value, leaf):
        leaves.append(value)
        return len(leaves) - 1
    if (oid := id(value)) in memo:
        if (slot := memo[oid][1]) is None:
            raise ValueError("containers cannot contain cycles")
        return slot
    layout = _layout(type(value), leaf)
    node = layout.decompose(value) if layout is not None else None
    if node is not None:
        memo[oid] = value, None
        children, restore = node
        refs = [_capture(child, leaves, template, steps, memo, leaf) for child in children]
        read = itemgetter(*refs) if len(refs) > 1 else lambda values: [values[ref] for ref in refs]
        steps.append((-len(template) - 1, restore, read))
    template.append(value if node is None else None)
    memo[oid] = value, -len(template)
    return -len(template)


def _replace[Leaf](value: Any, fn: Callable[[Leaf], Leaf], memo: _Memo[Any], leaf: type[Leaf]) -> Any:
    if (oid := id(value)) in memo:
        if (result := memo[oid][1]) is None:
            raise ValueError("containers cannot contain cycles")
        return result
    layout = _layout(type(value), leaf)
    node = layout.decompose(value) if layout is not None else None
    if node is None:
        return value
    memo[oid] = value, None
    children, restore = node
    results, changed = [], False
    for child in children:
        result = fn(child) if isinstance(child, leaf) else _replace(child, fn, memo, leaf)
        results.append(result)
        changed = changed or result is not child
    result = restore(results) if changed else value
    memo[oid] = value, result
    return result
