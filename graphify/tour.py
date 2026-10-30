"""Guided tours: a reading order through the codebase.

A tour starts where execution starts — a page, an API route, or a function
nothing else calls — and follows the flow outward depth-first (calls,
requests, route handlers, rendered components, links), so a web request reads
as component -> API call -> route -> handler -> data layer. Every step says
why it comes next. A tour closes with a few foundations: the code the most
other code depends on.

Deterministic and offline, like graphify/explain.py, whose explanations each
step carries. Shared by graph.html (the stepper), ``graphify tour`` and the
MCP ``get_tour`` tool.
"""
from __future__ import annotations

import re
from pathlib import PurePosixPath

import networkx as nx

from graphify.explain import _DEPENDENT_RELATIONS, node_kind, relation_index
from graphify.security import sanitize_label

# Outgoing relations a tour follows, with how a step reached through each one
# is introduced ("{parent} calls {node}."). Order = priority between relations.
FLOW_RELATIONS: dict[str, str] = {
    "handled_by": "{parent} is served by {node}.",
    "requests": "{parent} sends a request to {node}.",
    "uses_component": "{parent} renders {node}.",
    "calls": "{parent} calls {node}.",
    "instantiates": "{parent} creates a {node}.",
    "indirect_call": "{parent} hands {node} over to be called later.",
    "navigates": "{parent} links to {node}.",
}
_START_KINDS = frozenset({"page", "API route", "function", "method", "UI component"})
_INCOMING_STARTS = frozenset({"calls", "indirect_call", "requests", "navigates", "handled_by", "uses_component"})
_REACH_CAP = 400
DEFAULT_STEPS = 10
_FOUNDATION_STEPS = 2


# Tests and generated code are not where a reader should start or what they
# need as foundations (a test helper "used by 38 others" is not architecture).
_NOT_READING_MATERIAL = re.compile(
    r"(^|/)(tests?|__tests__|spec|e2e|node_modules|vendor|dist|build)/"
    r"|(^|/)(test_[^/]*|[^/]*_test\.\w+|[^/]*\.(test|spec|gen|generated|d)\.\w+)$"
)


def _label(G: nx.Graph, nid: str) -> str:
    """Name for prose: a page reads as "the /users/:id page"."""
    d = G.nodes[nid]
    if d.get("route_method") == "PAGE" and d.get("route_path"):
        return f"the {sanitize_label(str(d['route_path']))} page"
    return sanitize_label(str(d.get("label") or nid))


def _step_reason(G: nx.Graph, rel: str, parent: str, child: str) -> str:
    p, c = _label(G, parent), _label(G, child)
    if p == c:  # an API handler calling the same-named CRUD function: say which is which
        p += f" ({PurePosixPath(str(G.nodes[parent].get('source_file') or '')).name})"
        c += f" ({PurePosixPath(str(G.nodes[child].get('source_file') or '')).name})"
    text = FLOW_RELATIONS[rel].format(parent=p, node=c)
    # Capitalize the article only; code names keep their case (update_user()).
    return "The" + text[3:] if text.startswith("the ") else text


def _readable(G: nx.Graph, nid: str) -> bool:
    """Real project code: sourced, `code` (not a package.json dependency
    concept or an external stub), and not test or generated code."""
    d = G.nodes[nid]
    src = str(d.get("source_file") or "")
    return bool(src) and d.get("file_type") == "code" and not _NOT_READING_MATERIAL.search(src)


def _flow_children(G: nx.Graph, index: dict, nid: str) -> list[tuple[str, str]]:
    """(relation, child) for each outgoing flow edge to real, sourced code."""
    out = []
    seen = set()
    for direction, rel, nb, _ in index.get(nid, []):
        if direction != "out" or rel not in FLOW_RELATIONS or nb in seen or nb not in G:
            continue
        if not G.nodes[nb].get("source_file") or G.nodes[nb].get("file_type") != "code":
            continue  # external stubs (ref_express, ...) are not part of the code to read
        seen.add(nb)
        out.append((rel, nb))
    return out


def _reach(G: nx.Graph, index: dict, start: str) -> set[str]:
    seen = {start}
    stack = [start]
    while stack and len(seen) < _REACH_CAP:
        for _, child in _flow_children(G, index, stack.pop()):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def suggest_starts(G: nx.Graph, index: dict | None = None, limit: int = 3) -> list[str]:
    """Where tours should begin: pages, routes and uncalled functions, ranked
    by how much code they lead to. Later picks must reach mostly new code, so
    the suggested tours cover different parts of the app."""
    index = index if index is not None else relation_index(G)
    candidates = []
    for nid in G.nodes:
        if not _readable(G, nid):
            continue
        rels = index.get(nid, [])
        kind = node_kind(G, nid, rels)
        if kind not in _START_KINDS:
            continue
        called = any(direction == "in" and r in _INCOMING_STARTS for direction, r, _, _ in rels)
        # A page is opened by links and an API route by requests; both still
        # start a flow. Anything else that is called is a mid-flow step.
        if called and kind not in ("page", "API route"):
            continue
        reach = _reach(G, index, nid)
        if len(reach) > 1:
            priority = {"page": 0, "API route": 1}.get(kind, 2)
            candidates.append((-len(reach), priority, str(nid), reach))
    candidates.sort(key=lambda c: c[:3])
    picked: list[str] = []
    covered: set[str] = set()
    for _, _, nid, _reach_set in candidates:
        if len(picked) >= limit:
            break
        # Judge overlap on what the tour would actually show, not everything
        # reachable: two tours walking mostly the same steps are one tour.
        walk = {s["id"] for s in build_tour(G, nid, index=index, max_steps=DEFAULT_STEPS - _FOUNDATION_STEPS,
                                              with_foundations=False)["steps"]}
        if covered and len(walk - covered) < len(walk) / 2:
            continue
        picked.append(nid)
        covered |= walk
    if not picked:
        # A library with no visible entry point: start from the most connected node.
        best = max((n for n in G.nodes if _readable(G, n)),
                   key=lambda n: (G.degree(n), str(n)), default=None)
        if best is not None:
            picked.append(best)
    return picked


def build_tour(G: nx.Graph, start: str, *, index: dict | None = None,
               max_steps: int = DEFAULT_STEPS, with_foundations: bool = True) -> dict:
    """A tour from ``start``: ``{"title", "start", "steps": [{"id", "reason",
    "via", "from"}]}``. Depth-first along FLOW_RELATIONS (bigger branches
    first), then up to two foundations the walk did not reach."""
    index = index if index is not None else relation_index(G)
    max_steps = max(1, max_steps)
    rels = index.get(start, [])
    kind = node_kind(G, start, rels)
    reach = _reach(G, index, start)
    name = _label(G, start)
    if kind in ("page", "API route"):
        first = f"Start at {name if kind == 'page' else f'the {name} route'}: this is where the flow enters the code."
    elif len(reach) > 1:
        first = (f"Start here: nothing in the scanned code calls {name} directly, so this is where the flow "
                 f"begins. It leads to {len(reach) - 1} other piece{'s' if len(reach) != 2 else ''} of code.")
    else:
        first = f"Start here: {name}."
    steps = [{"id": start, "reason": first, "via": None, "from": None}]
    visited = {start}
    sizes: dict[str, int] = {}

    def size(nid: str) -> int:
        if nid not in sizes:
            sizes[nid] = len(_reach(G, index, nid))
        return sizes[nid]

    order = list(FLOW_RELATIONS)
    # Depth-first: follow one request all the way down before the next branch.
    stack = [(start, iter(sorted(_flow_children(G, index, start),
                                 key=lambda rc: (order.index(rc[0]), -size(rc[1]), str(rc[1])))))]
    flow_budget = max_steps - (_FOUNDATION_STEPS if max_steps > 4 and with_foundations else 0)
    while stack and len(steps) < flow_budget:
        parent, children = stack[-1]
        nxt = next(((rel, c) for rel, c in children if c not in visited), None)
        if nxt is None:
            stack.pop()
            continue
        rel, child = nxt
        visited.add(child)
        steps.append({
            "id": child,
            "reason": _step_reason(G, rel, parent, child),
            "via": rel,
            "from": parent,
        })
        stack.append((child, iter(sorted(_flow_children(G, index, child),
                                         key=lambda rc: (order.index(rc[0]), -size(rc[1]), str(rc[1]))))))

    foundations = []
    for nid in (G.nodes if with_foundations else ()):
        if nid in visited or not _readable(G, nid):
            continue
        n_rels = index.get(nid, [])
        if node_kind(G, nid, n_rels) in ("comment", "file"):
            continue
        dependents = {nb for direction, r, nb, _ in n_rels if direction == "in" and r in _DEPENDENT_RELATIONS}
        if len(dependents) >= 2:
            foundations.append((-len(dependents), str(nid), nid, len(dependents)))
    for _, _, nid, n in sorted(foundations)[: max(0, min(_FOUNDATION_STEPS, max_steps - len(steps)))]:
        steps.append({
            "id": nid,
            "reason": f"Foundation: {n} other pieces of code depend on {_label(G, nid)}, so it is worth knowing.",
            "via": "foundation",
            "from": None,
        })
    where = PurePosixPath(str(G.nodes[start].get("source_file") or "")).name
    title = f"Follow {name}" + (f" ({where})" if where and kind not in ("page", "API route") else "")
    return {"title": title, "start": start, "steps": steps}


def suggest_tours(G: nx.Graph, *, index: dict | None = None, limit: int = 3,
                  max_steps: int = DEFAULT_STEPS) -> list[dict]:
    index = index if index is not None else relation_index(G)
    return [build_tour(G, s, index=index, max_steps=max_steps)
            for s in suggest_starts(G, index, limit)]


def format_tour(G: nx.Graph, tour: dict, *, reader=None, code_lines: int = 15) -> list[str]:
    """Text rendering for ``graphify tour`` and MCP ``get_tour``: each step's
    reason, its plain-language explanation, and (with a reader) a short code
    excerpt."""
    from graphify.explain import explain_node, format_explanation

    steps = tour["steps"]
    out = [f"Tour: {sanitize_label(tour['title'])} ({len(steps)} step{'s' if len(steps) != 1 else ''})"]
    for i, step in enumerate(steps, 1):
        nid = step["id"]
        d = G.nodes[nid]
        ex = explain_node(G, nid, community_name=d.get("community_name"), reader=reader)
        loc = sanitize_label(str(d.get("source_file") or ""))
        if ex.get("line"):
            loc += f":{ex['line']}"
        name = sanitize_label(str(d.get("label") or nid))
        out += ["", f"Step {i}/{len(steps)}: {name}  [{ex['kind']} · {loc}]", f"  Why: {step['reason']}"]
        code = ex.get("code")
        if code and len(code["lines"]) > code_lines:
            ex["code"] = {**code, "lines": code["lines"][:code_lines], "truncated": True}
        # The explanation block, minus its leading blank line.
        out += ["  " + line if line else line for line in format_explanation(ex, str(d.get("source_file") or ""))[1:]]
    return out
