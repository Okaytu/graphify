"""Plain-language explanations of graph nodes — what a piece of code IS, what
it DOES in the codebase, WHY it exists (its own docstrings/comments) and the
code itself.

Deterministic and offline: everything here is derived from the graph that is
already built (edge relations, ``source_location``, ``rationale_for`` nodes)
plus a bounded read of the source file. No LLM call, no API cost. Shared by
``graph.html`` (the node panel and the codebase overview) and the
``graphify explain`` CLI so both describe a node the same way.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path

import networkx as nx

from graphify.security import sanitize_label

# relation -> (heading when this node is the edge SOURCE, heading when TARGET).
# Ordered: the panel lists groups in this order, unknown relations last.
RELATION_PHRASES: dict[str, tuple[str, str]] = {
    "contains": ("Contains", "Defined in"),
    "defines": ("Contains", "Defined in"),
    "method": ("Methods", "Method of"),
    "inherits": ("Inherits from", "Inherited by"),
    "extends": ("Inherits from", "Inherited by"),
    "implements": ("Implements", "Implemented by"),
    "calls": ("Calls", "Called by"),
    "indirect_call": ("Calls indirectly", "Called indirectly by"),
    "instantiates": ("Creates", "Created by"),
    "uses_component": ("Renders", "Rendered by"),
    "uses": ("Uses", "Used by"),
    "imports": ("Imports", "Imported by"),
    "imports_from": ("Imports", "Imported by"),
    "re_exports": ("Re-exports", "Re-exported by"),
    "includes": ("Includes", "Included by"),
    "depends_on": ("Depends on", "Depended on by"),
    "crate_depends_on": ("Depends on", "Depended on by"),
    "references": ("References", "Referenced by"),
    "references_constant": ("References", "Referenced by"),
    "navigates": ("Navigates to", "Navigated to from"),
    "tests": ("Tests", "Tested by"),
    "cites": ("Cites", "Cited by"),
}

# Incoming relations that mean "this other thing breaks if I change".
_DEPENDENT_RELATIONS = frozenset({
    "calls", "indirect_call", "imports", "imports_from", "re_exports", "uses",
    "uses_component", "references", "references_constant", "inherits", "extends",
    "implements", "instantiates", "includes", "depends_on", "crate_depends_on",
})

_UI_SUFFIXES = (".tsx", ".jsx", ".vue", ".svelte", ".astro")
_DOC_ATTRS = ("docstring", "description", "summary", "rationale")
_MAX_DOCS = 5
_SNIPPET_MAX_LINES = 30
_FILE_HEAD_LINES = 15
_LINE_CLIP = 240
_MAX_SOURCE_BYTES = 2_000_000
_LEAD_COMMENT_LOOKBACK = 25
# Lines that may sit directly above a definition and describe it. Python
# triple-quoted strings are deliberately absent: a docstring above a `def` is
# the enclosing body's docstring, not this definition's (its own is below).
_COMMENT_START = ("//", "/*", "*", "#", "@", "--", "<!--", "%", ";")
_CLOSERS = ("}", ")", "]", "end", "</")


def humanize_relation(relation: str, outgoing: bool) -> str:
    phrases = RELATION_PHRASES.get(relation)
    if phrases:
        return phrases[0] if outgoing else phrases[1]
    words = (relation or "related to").replace("_", " ")
    return words[:1].upper() + words[1:] if outgoing else f"{words[:1].upper()}{words[1:]} (incoming)"


def relation_index(G: nx.Graph) -> dict[str, list[tuple[str, str, str, dict]]]:
    """One pass over the edges: node id -> [(direction, relation, neighbor, data)].

    Direction is the edge's TRUE direction — the ``_src``/``_tgt`` markers win
    over arc order, because an undirected graph canonicalizes endpoint order
    (#563/#2309). ``direction`` is ``"out"`` when the node is the source.
    """
    index: dict[str, list[tuple[str, str, str, dict]]] = defaultdict(list)
    for u, v, data in G.edges(data=True):
        src = data.get("_src", u)
        tgt = data.get("_tgt", v)
        if src not in (u, v) or tgt not in (u, v):
            src, tgt = u, v
        rel = str(data.get("relation") or "")
        index[src].append(("out", rel, tgt, data))
        index[tgt].append(("in", rel, src, data))
    return index


def _label(G: nx.Graph, nid: str) -> str:
    return sanitize_label(str(G.nodes[nid].get("label", nid))) if nid in G else str(nid)


def _line_of(location) -> int | None:
    m = re.search(r"(\d+)", str(location or ""))
    return int(m.group(1)) if m else None


def node_kind(G: nx.Graph, nid: str, rels: list[tuple[str, str, str, dict]] | None = None) -> str:
    """Best-effort human kind: function, method, class, UI component, file, ..."""
    d = G.nodes[nid]
    ft = str(d.get("file_type") or "")
    if ft == "rationale":
        return "comment"
    if ft and ft != "code":
        return ft
    rels = rels if rels is not None else []
    label = str(d.get("label") or nid)
    src = str(d.get("source_file") or "")
    in_rels = {r for direction, r, _, _ in rels if direction == "in"}
    out_rels = {r for direction, r, _, _ in rels if direction == "out"}
    if src and label == Path(src).name and not ({"contains", "defines", "method"} & in_rels):
        return "file"
    if "method" in in_rels or label.startswith("."):
        return "method"
    if d.get("_callable_class") or "method" in out_rels or {"inherits", "extends"} & out_rels:
        return "class"
    if label.endswith("()"):
        name = label[:-2]
        if src.endswith(_UI_SUFFIXES) and name[:1].isupper():
            return "UI component"
        return "function"
    if not ft:
        return "concept"
    return "symbol"


def node_docs(G: nx.Graph, nid: str, rels: list[tuple[str, str, str, dict]] | None = None) -> list[str]:
    """The node's own explanation of itself: docstrings and rationale comments
    (``rationale_for`` nodes pointing at it) plus any semantic description the
    LLM pass attached. Ordered by source line, de-duplicated, capped."""
    d = G.nodes[nid]
    found: list[tuple[int, str]] = []
    for attr in _DOC_ATTRS:
        val = d.get(attr)
        if isinstance(val, str) and val.strip():
            found.append((-1, val.strip()))
    for direction, rel, nb, _ in rels or []:
        if direction == "in" and rel == "rationale_for" and nb in G:
            nd = G.nodes[nb]
            found.append((_line_of(nd.get("source_location")) or 0, str(nd.get("label") or "")))
    seen: set[str] = set()
    docs: list[str] = []
    for _, text in sorted(found, key=lambda t: t[0]):
        text = sanitize_label(text)
        if text and text not in seen:
            seen.add(text)
            docs.append(text)
    return docs[:_MAX_DOCS]


def _group(G: nx.Graph, rels) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for direction, rel, nb, _ in rels:
        if rel == "rationale_for" or nb not in G:
            continue
        key = humanize_relation(rel, direction == "out")
        if nb not in groups[key]:
            groups[key].append(nb)
    return groups


def _names(G: nx.Graph, ids: list[str], limit: int = 3) -> str:
    shown = ", ".join(_label(G, i) for i in ids[:limit])
    # "(+N others)", not "and N more": `explain` reserves "more" for its own
    # truncated-listing footer, which scripts and tests look for.
    return shown + (f" (+{len(ids) - limit} others)" if len(ids) > limit else "")


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def summarize_node(
    G: nx.Graph,
    nid: str,
    rels: list[tuple[str, str, str, dict]] | None = None,
    community_name: str | None = None,
) -> str:
    """A few plain-English sentences: what the node is, where it lives, what it
    does with the rest of the codebase, and what changing it would touch."""
    d = G.nodes[nid]
    rels = rels if rels is not None else []
    kind = node_kind(G, nid, rels)
    label = _label(G, nid)
    groups = _group(G, rels)
    src = sanitize_label(str(d.get("source_file") or ""))
    line = _line_of(d.get("source_location"))

    article = "an" if kind[:1].lower() in "aeiou" else "a"
    first = f"{label} is {article} {kind}"
    if kind == "method" and groups.get("Method of"):
        first += f" of {_names(G, groups['Method of'], 1)}"
    if src:
        first += f" in {src}" + (f" (line {line})" if line and kind != "file" else "")
    sentences = [first + "."]

    if kind == "comment":
        targets = [nb for direction, rel, nb, _ in rels if direction == "out" and rel == "rationale_for"]
        where = (f" in {src}" + (f" (line {line})" if line else "")) if src else ""
        about = f", explaining why {_names(G, targets, 2)} exists or works the way it does" if targets else ""
        return f"A docstring or comment{where}{about}."

    if community_name:
        sentences.append(f"It belongs to the “{sanitize_label(community_name)}” area of the codebase.")

    defined = groups.get("Contains", [])
    if defined:
        sentences.append(f"It defines {_plural(len(defined), 'symbol')}: {_names(G, defined)}.")
    methods = groups.get("Methods", [])
    if methods:
        sentences.append(f"It has {_plural(len(methods), 'method')}: {_names(G, methods)}.")
    parents = groups.get("Inherits from", [])
    if parents:
        sentences.append(f"It builds on {_names(G, parents)}.")
    imports = groups.get("Imports", [])
    if imports:
        sentences.append(f"It imports {_names(G, imports)}.")
    calls = groups.get("Calls", [])
    callers = groups.get("Called by", [])
    if calls:
        sentences.append(f"It calls {_names(G, calls)}.")
    if callers:
        sentences.append(f"It is called by {_names(G, callers)}.")

    if kind in ("function", "method", "UI component"):
        if calls and not callers:
            sentences.append(
                "Nothing in the scanned code calls it directly, so it is probably an entry point "
                "(route/event handler, exported API, CLI command), called dynamically, or unused."
            )
        elif callers and not calls:
            sentences.append("It calls nothing else in the codebase: a self-contained building block.")

    dependents = {nb for direction, rel, nb, _ in rels if direction == "in" and rel in _DEPENDENT_RELATIONS}
    if dependents:
        files = {str(G.nodes[nb].get("source_file") or "") for nb in dependents if nb in G}
        files.discard("")
        where = f" across {_plural(len(files), 'file')}" if files else ""
        sentences.append(f"Changing it directly affects {_plural(len(dependents), 'dependent')}{where}.")
    elif kind not in ("file", "concept", "document", "paper", "image"):
        sentences.append("No other scanned code depends on it, so changes stay local.")
    return " ".join(sentences)


# ── Source snippets ──────────────────────────────────────────────────────────

def snippets_enabled() -> bool:
    """``GRAPHIFY_HTML_CODE=0`` keeps source code out of graph.html (e.g. when
    the file will be shared outside the team)."""
    return os.environ.get("GRAPHIFY_HTML_CODE", "1").strip().lower() not in ("0", "false", "no", "off")


def resolve_source(src: str, graph_path: str | Path) -> Path | None:
    """Locate ``source_file`` on disk, confined to the project root.

    Reuses reflect's root search (``.graphify_root`` marker, graphify-out
    layout, flat layout). A path that climbs out with ``..`` or an absolute
    path outside every candidate root is refused, so a crafted graph.json
    cannot pull arbitrary files into the HTML.
    """
    if not src:
        return None
    from graphify.reflect import _resolve_source_path
    p = Path(src)
    if ".." in p.parts:
        return None
    found = _resolve_source_path(src, Path(graph_path))
    if found is None:
        return None
    if p.is_absolute():
        out_dir = Path(graph_path).resolve().parent
        roots = [out_dir, out_dir.parent]
        try:
            recorded = (out_dir / ".graphify_root").read_text(encoding="utf-8-sig").strip()
            if recorded:
                roots.append(Path(recorded).resolve())
        except (OSError, ValueError):
            pass
        resolved = found.resolve()
        if not any(resolved.is_relative_to(r) for r in roots):
            return None
    return found


def _indent(s: str) -> int:
    return len(s) - len(s.lstrip())


def snippet_from_lines(lines: list[str], line: int, *, whole_file_head: bool = False,
                       max_lines: int = _SNIPPET_MAX_LINES) -> dict | None:
    """Cut the definition starting at 1-based ``line`` out of ``lines``.

    Includes the comment / JSDoc / decorator block directly above it (that is
    usually where the "why" lives) and follows the body by indentation until it
    dedents back to the definition's level. Closing-bracket lines are kept;
    ``) {`` / ``):`` continuations of a multi-line signature do not end it.
    """
    if whole_file_head:
        chunk = lines[:_FILE_HEAD_LINES]
        if not chunk:
            return None
        return {"start": 1, "lines": [ln[:_LINE_CLIP] for ln in chunk],
                "truncated": len(lines) > len(chunk)}
    if line < 1 or line > len(lines):
        return None
    i0 = line - 1
    start = i0
    while start > 0 and i0 - (start - 1) <= _LEAD_COMMENT_LOOKBACK:
        prev = lines[start - 1].strip()
        if prev and prev.startswith(_COMMENT_START):
            start -= 1
        else:
            break
    base = _indent(lines[i0])
    end = i0 + 1
    while end < len(lines) and end - start < max_lines:
        text = lines[end]
        stripped = text.strip()
        if stripped and _indent(text) <= base:
            if stripped.startswith(_CLOSERS):
                end += 1
                if stripped.endswith(("{", ":", "(", "=>")):
                    continue
            break
        end += 1
    while end > i0 + 1 and not lines[end - 1].strip():
        end -= 1
    truncated = end - start >= max_lines and end < len(lines)
    return {"start": start + 1, "lines": [ln[:_LINE_CLIP] for ln in lines[start:end]],
            "truncated": truncated}


class SnippetReader:
    """Reads each source file at most once per export."""

    def __init__(self, graph_path: str | Path):
        self.graph_path = graph_path
        self._cache: dict[str, list[str] | None] = {}

    def lines(self, src: str) -> list[str] | None:
        if src not in self._cache:
            text = None
            path = resolve_source(src, self.graph_path)
            try:
                if path is not None and path.stat().st_size <= _MAX_SOURCE_BYTES:
                    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                text = None
            self._cache[src] = text
        return self._cache[src]

    def snippet(self, G: nx.Graph, nid: str, kind: str) -> dict | None:
        d = G.nodes[nid]
        if str(d.get("file_type") or "") != "code":
            return None
        lines = self.lines(str(d.get("source_file") or ""))
        if not lines:
            return None
        if kind == "file":
            return snippet_from_lines(lines, 1, whole_file_head=True)
        line = _line_of(d.get("source_location"))
        return snippet_from_lines(lines, line) if line else None


def explain_node(
    G: nx.Graph,
    nid: str,
    *,
    index: dict | None = None,
    community_name: str | None = None,
    reader: SnippetReader | None = None,
) -> dict:
    """Everything the UI/CLI shows about one node, as plain data."""
    rels = (index if index is not None else relation_index(G)).get(nid, [])
    kind = node_kind(G, nid, rels)
    d = G.nodes[nid]
    line = _line_of(d.get("source_location"))
    out = {
        "kind": kind,
        "summary": summarize_node(G, nid, rels, community_name),
        "docs": node_docs(G, nid, rels),
        "line": line,
    }
    if reader is not None:
        code = reader.snippet(G, nid, kind)
        if code:
            out["code"] = code
            lead = leading_comment(code, line) if kind != "file" else ""
            if lead and not any(lead in doc or doc in lead for doc in out["docs"]):
                out["docs"].insert(0, lead)
    return out


def leading_comment(code: dict, line: int | None) -> str:
    """Text of the JSDoc / ``//`` / ``#`` block above the definition line in a
    snippet — for languages whose doc comments the AST pass does not turn into
    rationale nodes (JS/TS JSDoc, Go, Rust, C#, ...). Decorators are skipped."""
    if not line:
        return ""
    parts: list[str] = []
    for text in code["lines"][: max(0, line - code["start"])]:
        t = text.strip()
        if t.startswith("@"):
            continue
        t = re.sub(r"^(/\*\*?|\*/|\*|///?|#+|--|<!--)", "", t)
        t = re.sub(r"(\*/|-->)$", "", t).strip()
        if t:
            parts.append(t)
    return sanitize_label(" ".join(parts)) if parts else ""


# ── Codebase overview ────────────────────────────────────────────────────────

def codebase_overview(G: nx.Graph, communities: dict[int, list[str]],
                      community_labels: dict[int, str] | None = None,
                      index: dict | None = None, top: int = 6) -> dict:
    """The big picture shown before any node is clicked: what the codebase is
    made of, its foundations (most depended-upon code), its likely entry points
    and its main areas."""
    index = index if index is not None else relation_index(G)
    kinds: dict[str, int] = defaultdict(int)
    foundations: list[tuple[int, str]] = []
    entries: list[tuple[int, str]] = []
    for nid in G.nodes:
        rels = index.get(nid, [])
        kind = node_kind(G, nid, rels)
        kinds[kind] += 1
        if kind in ("comment", "file"):
            continue
        dependents = {nb for direction, r, nb, _ in rels if direction == "in" and r in _DEPENDENT_RELATIONS}
        if dependents:
            foundations.append((len(dependents), nid))
        if kind in ("function", "method", "UI component"):
            calls = sum(1 for direction, r, _, _ in rels if direction == "out" and r == "calls")
            called = any(direction == "in" and r in ("calls", "indirect_call") for direction, r, _, _ in rels)
            if calls and not called:
                entries.append((calls, nid))
    foundations.sort(key=lambda t: (-t[0], str(t[1])))
    entries.sort(key=lambda t: (-t[0], str(t[1])))
    areas = []
    for cid, members in sorted(communities.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:top]:
        members = [m for m in members if m in G]
        if not members:
            continue
        key = max(members, key=lambda m: (G.degree(m), str(m)))
        areas.append({
            "name": sanitize_label((community_labels or {}).get(cid, f"Community {cid}")),
            "size": len(members),
            "key": key,
        })
    return {
        "kinds": dict(sorted(kinds.items(), key=lambda kv: -kv[1])),
        "foundations": [{"id": nid, "dependents": n} for n, nid in foundations[:top]],
        "entry_points": [{"id": nid, "calls": n} for n, nid in entries[:top]],
        "areas": areas,
    }
