from ren.engine.pattern import RuleSet, pat, rules
from ren.ir import GroupOp, Op
from ren.ir.node import Node
from ren.ring import Domain


def collect_realize(realize: set[Node]) -> RuleSet:
    """Collect nodes that must be materialized before indexification.

    A node needs realization when it sits at a materialization boundary:
      - SINK outputs (final results consumed externally)
      - Conversion ops (NTT, INTT, TO_RING, etc.) and their inputs, except
        elementwise inputs of an automorphism, which its scatter kernel computes
      - CALL inputs and AFTER values
      - BUFFERIZE / CONTIGUOUS values and random polynomials

    Existing buffers are reused. Constants stay inline unless an input requires storage.

    For example, a chain like:
        SINK(ADD(NTT(x), y))
    marks ADD and NTT, plus x if it is not already realized (NTT is a conversion boundary).
    """

    def realize_conversion(n: Node) -> None:
        if n.view_offset is None:
            realize.add(n)
            # Conversion inputs need concrete buffers, including constants. Their
            # storage size can differ from the output's iteration space (e.g. FROM_RNS).
            inline = GroupOp.Elementwise if n.op is Op.AUTOMORPHISM else frozenset()
            realize.update(s for s in n.src if s.base.op not in GroupOp.AlwaysRealized - {Op.CONST} | inline)

    return rules([
        (pat.n(Op.SINK), lambda n: realize.update(s for s in n.src if s._device is not None and s.base.op not in GroupOp.AlwaysRealized)),
        (pat(Op.BUFFERIZE, pat.x), lambda x: realize.add(x) if x.base.op not in GroupOp.AlwaysRealized else None),
        (pat.n(GroupOp.Conversion), realize_conversion),
        (pat.n(Op.CONTIGUOUS), lambda n: realize.add(n)),
        (pat.n(Op.AS_RING, pat(Op.RANDOM)), lambda n: realize.add(n)),
        (pat.n(Op.AFTER), lambda n: realize.add(n.src[0]) if n._device is not None and n.src[0].base.op not in GroupOp.AlwaysRealized else None),
        (pat.n(Op.CALL), lambda n: realize.update(s for s in n.inputs if s.base.op is not Op.BUFFERIZE and not s.has_buffer_identity)),
    ])


def bufferize_plan(root: Node) -> set[Node]:
    """Determine which nodes under a SINK root must be materialized before indexification."""
    realize: set[Node] = set()
    nodes = root.toposort()
    collect = collect_realize(realize)
    for node in nodes:
        collect.match(node)

    # Preserve metadata and view boundaries around materialized values.
    # Dependencies precede their wrappers, so one forward pass covers nested views.
    for n in nodes:
        if (n.op in GroupOp.Metadata or n.view_offset is not None) and n.src[0] in realize:
            realize.add(n)

    return realize


def reuse_outputs(root: Node) -> RuleSet:
    """If a consumer needs a value that is also an output, read it from that output's buffer."""
    outputs: dict[Node, Node] = {}
    for output in root.src:
        if output.op is not Op.AFTER:
            continue
        for store in output.src[1:]:
            if store.op is Op.STORE and store.src[0] is output.src[0]:
                value = store.src[1]
                value = value.src[0] if value.op is Op.CONTIGUOUS else value
                storage = output.as_ring(value.ring, value.domain) if value.ring is not None else output
                outputs[value.bufferize(output.device)] = outputs[value.contiguous().bufferize(output.device)] = storage

    return rules([(pat.b(Op.BUFFERIZE), lambda b: outputs.get(b))])


def bufferize(realize: set[Node]) -> RuleSet:
    """Mark inputs that need a buffer. For example, x becomes BUFFERIZE(x) in x + y.
    If x has no device yet, the STORE that uses it supplies one.
    """

    def materialize(source: Node, device: str | None) -> Node:
        return source.bufferize(source._device or device) if source in realize else source

    return rules([
        # Only the first input to AFTER carries a value.
        (pat.n(Op.AFTER), lambda n: n.replace(src=(materialize(n.src[0], n._device), *n.src[1:]))),
        (pat(Op.STORE, (pat.target, pat(Op.CONTIGUOUS, pat.value))), lambda target, value: target.store(value)),
        (pat.n(Op.CALL), lambda n: n.replace(src=(n.src[0], *(materialize(s, n._device) for s in n.inputs)))),
        (pat.n(Op.SINK), lambda n: n.replace(src=tuple(materialize(s, s._device) if s._device is not None else s for s in n.src))),
        (
            pat.n(GroupOp.Elementwise | GroupOp.Conversion | GroupOp.Ring | {Op.AS_RING, Op.CONTIGUOUS}),
            lambda n: n.replace(src=tuple(materialize(s, n._device) for s in n.src)),
        ),
    ])


def _to_store(b: Node, x: Node) -> Node | None:
    if b._device is None:
        return None
    if x.has_buffer_identity and x.device == b.device:
        return x
    dst = x.empty_like(b.device)
    return dst.after(dst.store(x))


# Lower BUFFERIZE markers into STORE ops backed by fresh allocation requests.
# A storage view already has a buffer; other values become AFTER(dst, STORE(dst, x)).
# CONTIGUOUS wrappers are stripped first since the STORE itself guarantees contiguity.
#
# Example:
#   BUFFERIZE(ADD(x, y))
#     -> dst = AS_RING(ALLOC(size=n*L), ring, domain)
#       AFTER(dst, STORE(dst, ADD(x, y)))
bufferize_to_store = rules([
    (pat.b(Op.BUFFERIZE, pat(Op.CONTIGUOUS, pat.x)), _to_store),
    (pat.b(Op.BUFFERIZE, pat.x), _to_store),
])


def _buffer_view(n: Node) -> Node | None:
    if n.view_offset is None:
        return None
    src = n.src[0]
    while src.op is Op.AS_RING:
        src = src.src[0]
    if not src.has_buffer_identity:
        return None

    dst_ring, dst_domain = n.ring, n.domain
    assert dst_ring is not None
    # NOTE: assumes storage-rows == len(moduli). Unreachable for FROM_RNS raw-limb buffers today
    # (nothing sub-views them), but would mis-size if narrowed-limb reads ever slice one -- such a
    # view's rows are limb_count, not len(moduli). See Node._storage_size for the analogous handling.
    size = dst_ring.n * len(dst_ring.moduli)
    return Node(Op.BUFFER_VIEW, n.dtype, src=(src,), arg=(size, n.view_offset)).as_ring(dst_ring, dst_domain or Domain.COEFF)


buffer_views = rules([(pat.n(Op.TO_RING), _buffer_view)])
