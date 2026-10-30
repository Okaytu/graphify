"""Web route map: the HTTP routes and pages a codebase defines, and the
frontend calls and links that reach them.

A post-pass over ``extract()``'s output. It scans the source text of the
extracted JS/TS/Python files (no tree-sitter, no LLM) and adds:

* a ``route`` node per route the backend declares — ``GET /api/users/:id``
  (Express/Koa/Fastify/Hono-style ``app.get('/x', handler)``, Next.js
  ``app/**/route.ts`` and ``pages/api``, FastAPI/Flask decorators) — and per
  page (Next.js ``page.tsx`` / ``pages/``, react-router and vue-router
  ``path:`` declarations), contained by its file;
* ``handled_by`` edges from a route to the function that serves it;
* ``requests`` edges from the function that calls ``fetch``/``axios``/... to
  the route it hits, and ``navigates`` edges from a link or ``router.push`` to
  the page it opens.

Matching a call to a route is a deduction (``/users/${id}`` vs
``/users/:id``), so ``requests``/``navigates`` edges are ``INFERRED``; only
calls that match a route the scanned code defines produce an edge.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from graphify.extractors.base import _file_stem, _make_id

JS_SUFFIXES = frozenset({".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue", ".svelte", ".astro"})
PY_SUFFIXES = frozenset({".py"})
_MAX_BYTES = 1_000_000
_HTTP_METHODS = ("get", "post", "put", "patch", "delete", "head", "options")
# Method of a route that is a page, and of a call that is a navigation.
PAGE = "PAGE"
NAV = "NAV"
# Route methods that accept any request method.
_ANY = frozenset({"ALL", "ANY"})

# Segment markers.
PARAM = ":"     # route parameter: matches exactly one segment
REST = "**"     # catch-all: matches the rest of the path
DYN = "${}"     # dynamic segment in a call: matches only a route parameter


@dataclass
class Route:
    method: str
    path: str
    framework: str
    file: str
    line: int
    handler: str | None = None
    id: str = ""
    # Declared on a router that is mounted elsewhere (`app.use('/api', router)`,
    # `include_router(..., prefix=...)`): the real URL has an unknown prefix.
    mounted: bool = False
    obj: str = ""  # the router variable the route was declared on

    def mount(self, prefix: str) -> None:
        """Prepend the prefix the router is mounted under, and re-key."""
        self.path = (prefix.rstrip("/") + ("" if self.path == "/" else self.path)) or "/"
        self.mounted = False
        self.id = ""
        self.__post_init__()

    def __post_init__(self):
        if not self.id:
            digest = hashlib.sha1(f"{self.method} {self.path}".encode()).hexdigest()[:6]
            self.id = _make_id("route", _file_stem(Path(self.file)), self.method, self.path, digest)

    @property
    def label(self) -> str:
        return f"{self.method} {self.path}"


@dataclass
class Call:
    method: str          # HTTP method, or NAV for a link / router.push
    segments: list[str]
    prefix_unknown: bool  # `${API_URL}/users` or an absolute URL: match a path suffix
    file: str
    line: int
    raw: str = field(default="", repr=False)


# ── Backend route declarations ───────────────────────────────────────────────

_QUOTED = r"""(?P<q>['"`])(?P<path>/[^'"`\n]*)(?P=q)"""
# A call URL may also start with a base-URL expression or an origin.
_CALL_QUOTED = r"""(?P<q>['"`])(?P<path>(?:/|\$\{|https?://)[^'"`\n]*)(?P=q)"""

# app.use('/api/users', auth, usersRouter) -> usersRouter is mounted at /api/users.
_MOUNT_RE = re.compile(r"\b[A-Za-z_$][\w$]*\s*\.\s*use\s*\(\s*" + _QUOTED + r"\s*,(?P<rest>[^\n]*)")
_IMPORT_RE = re.compile(
    r"""\bimport\s+(?:(?P<default>[\w$]+)\s*,?\s*)?(?:\{(?P<named>[^}]*)\}\s*)?from\s*['"](?P<mod>\.[^'"]+)['"]"""
)
_REQUIRE_RE = re.compile(
    r"""\b(?:const|let|var)\s+(?:(?P<default>[\w$]+)|\{(?P<named>[^}]*)\})\s*=\s*require\(\s*['"](?P<mod>\.[^'"]+)['"]"""
)

# app.get('/x', handler) / router.post(...) / fastify.delete(...). The object
# must look like a server or router, and a handler argument must follow the
# path — that is what tells a registration apart from a client `api.get('/x')`.
_JS_REGISTER_RE = re.compile(
    r"\b(?P<obj>[A-Za-z_$][\w$]*)\s*\.\s*(?P<m>get|post|put|patch|delete|head|options|all|use)\s*\(\s*"
    + _QUOTED + r"\s*,"
)
_SERVER_OBJ_RE = re.compile(r"(?:^|[a-z_])(?:app|router|server|fastify|routes?|api_?router|hono)$", re.I)

# react-router <Route path="/x" element={<Page />} /> and route objects
# { path: '/x', element: <Page/> } / { path: '/x', component: Page } (vue-router).
_JSX_ROUTE_RE = re.compile(r"<Route\b[^>]*?\bpath\s*=\s*\{?\s*" + _QUOTED + r"(?P<rest>[^>]*)")
_ROUTE_OBJ_RE = re.compile(
    r"\bpath\s*:\s*" + _QUOTED + r"\s*,\s*(?:element|component|Component)\s*:\s*(?:<\s*)?(?P<comp>[A-Z][\w$]*)"
)

# TanStack Router file routes: createFileRoute('/_layout/items/$id')({ component: Item })
_FILE_ROUTE_RE = re.compile(r"\bcreateFileRoute\s*\(\s*" + _QUOTED + r"\s*\)\s*\(\s*\{")
_COMPONENT_KEY_RE = re.compile(r"\bcomponent\s*:\s*([A-Z][\w$]*)")

# FastAPI / Flask 2: @app.get("/x") / @router.post("/x"); Flask: @bp.route("/x", methods=[...]).
_PY_DECORATOR_RE = re.compile(
    r"^\s*@(?P<obj>[A-Za-z_][\w.]*)\.(?P<m>get|post|put|patch|delete|head|options|api_route|route)\s*\(\s*"
    r"(?P<q>['\"])(?P<path>/[^'\"\n]*)(?P=q)(?P<rest>[^\n]*)",
    re.M,
)
# router = APIRouter(prefix="/users") / bp = Blueprint("x", __name__, url_prefix="/users")
_PY_PREFIX_RE = re.compile(
    r"^\s*(\w+)\s*=\s*(?:\w+\.)?(?:APIRouter|Blueprint)\s*\([^)]*?\b(?:url_)?prefix\s*=\s*['\"](/[^'\"]*)['\"]",
    re.M,
)
_PY_DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)", re.M)
_PY_METHODS_RE = re.compile(r"methods\s*=\s*[\[(]([^\])]*)[\])]")

_NEXT_EXPORT_RE = re.compile(
    r"^export\s+(?:async\s+)?(?:function\s+|const\s+)(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b", re.M
)
_DEFAULT_EXPORT_RE = re.compile(
    r"^export\s+default\s+(?:async\s+)?(?:function\s*\*?\s*)?(?P<name>[A-Za-z_$][\w$]*)?", re.M
)

# ── Frontend calls ───────────────────────────────────────────────────────────

_FETCH_RE = re.compile(r"(?<![\w$.])(?:fetch|\$fetch|ofetch|useFetch|useSWR|useSWRImmutable)\s*\(\s*" + _CALL_QUOTED)
_CLIENT_RE = re.compile(
    r"\b(?P<obj>[A-Za-z_$][\w$]*)\s*\.\s*(?P<m>get|post|put|patch|delete|del|head)\s*(?:<[^>\n]*>)?\s*\(\s*" + _CALL_QUOTED
)
_NAV_RE = re.compile(
    r"(?:\b(?:href|to)\s*=\s*\{?\s*|\b(?:router\.(?:push|replace)|navigate|redirect|navigateTo)\s*\(\s*)" + _QUOTED
)
# Request-options objects: `client.post({ url: '/api/v1/users', ... })` (openapi-ts
# and similar generated clients), `axios({ method: 'put', url: '/x' })`, `$.ajax({ url })`.
_URL_PROP_RE = re.compile(r"\burl\s*:\s*" + _CALL_QUOTED)
_OBJ_METHOD_CALL_RE = re.compile(r"\.\s*(get|post|put|patch|delete)\s*(?:<[^()]*>)?\s*\(\s*\{")
_METHOD_OPT_RE = re.compile(r"""\bmethod\s*:\s*['"](\w+)['"]""", re.I)


def _line_at(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _last_identifier(rest: str) -> str | None:
    """The handler named by the last argument of a registration call:
    ``auth, getUser)`` -> getUser, ``users.create)`` -> create. Inline
    arrow/function handlers have no name and give None."""
    depth, end = 0, len(rest)
    for i, ch in enumerate(rest):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:
                end = i
                break
            depth -= 1
    args = rest[:end].strip()
    if not args or "=>" in args or "function" in args:
        return None
    last = args.rsplit(",", 1)[-1].strip()
    m = re.fullmatch(r"(?:[A-Za-z_$][\w$]*\.)*([A-Za-z_$][\w$]*)", last)
    return m.group(1) if m else None


def _nextjs_path(rel: str) -> tuple[str, str] | None:
    """(route path, kind) for a Next.js file-convention route, where kind is
    "api" (app router ``route.ts`` / ``pages/api``) or "page". Only called for
    files under a directory holding ``next.config.*`` — a bare ``pages/``
    folder is just as common in react-router apps, where it is not routing."""
    parts = PurePosixPath(rel).parts
    stem = PurePosixPath(rel).stem
    if "app" in parts and stem in ("route", "page"):
        segs = list(parts[parts.index("app") + 1:-1])
        kind = "api" if stem == "route" else "page"
    elif "pages" in parts:
        segs = list(parts[parts.index("pages") + 1:-1]) + ([] if stem == "index" else [stem])
        if stem.startswith("_") or not segs and stem != "index":
            return None
        kind = "api" if segs[:1] == ["api"] else "page"
    else:
        return None
    out = []
    for seg in segs:
        if seg.startswith("(") and seg.endswith(")") or seg.startswith("@"):
            continue  # route groups and parallel-route slots are not in the URL
        m = re.fullmatch(r"\[\[?\.\.\.(\w+)\]?\]", seg)
        if m:
            out.append(f":{m.group(1)}*")
            continue
        m = re.fullmatch(r"\[(\w+)\]", seg)
        out.append(f":{m.group(1)}" if m else seg)
    return "/" + "/".join(out), kind


def scan_routes(rel: str, text: str, *, nextjs: bool = False) -> list[Route]:
    """Routes and pages declared in one file (``rel`` is the POSIX path
    relative to the project root). ``nextjs``: the file sits in a Next.js
    project, so its path itself may declare a route."""
    suffix = PurePosixPath(rel).suffix.lower()
    routes: list[Route] = []
    if suffix in PY_SUFFIXES:
        prefixes = {pm.group(1): pm.group(2) for pm in _PY_PREFIX_RE.finditer(text)}
        default_fw = "FastAPI" if "fastapi" in text else "Flask" if "flask" in text.lower() else "Python"
        for m in _PY_DECORATOR_RE.finditer(text):
            obj = m.group("obj").rsplit(".", 1)[-1]
            if not _SERVER_OBJ_RE.search(obj) and not re.search(r"(?:bp|blueprint|api)$", obj, re.I):
                continue
            line = _line_at(text, m.start("path"))
            d = _PY_DEF_RE.search(text, m.end())
            handler = d.group(1) if d and _line_at(text, d.start()) - line <= 15 else None
            if m.group("m") in ("route", "api_route"):
                mm = _PY_METHODS_RE.search(m.group("rest"))
                methods = re.findall(r"\w+", mm.group(1)) if mm else ["GET"]
            else:
                methods = [m.group("m")]
            path = prefixes.get(obj, "").rstrip("/") + m.group("path")
            for method in methods:
                routes.append(Route(method.upper(), path, default_fw, rel, line, handler,
                                    mounted=obj != "app"))
        return routes
    if suffix not in JS_SUFFIXES:
        return routes

    nxt = _nextjs_path(rel) if nextjs else None
    if nxt:
        path, kind = nxt
        if kind == "api" and PurePosixPath(rel).stem == "route":
            for m in _NEXT_EXPORT_RE.finditer(text):
                routes.append(Route(m.group(1), path, "Next.js", rel, _line_at(text, m.start()), m.group(1)))
        else:
            d = _DEFAULT_EXPORT_RE.search(text)
            if d:
                method = "ANY" if kind == "api" else PAGE
                routes.append(Route(method, path, "Next.js", rel, _line_at(text, d.start()), d.group("name")))
    for m in _JS_REGISTER_RE.finditer(text):
        if m.group("m") == "use" or not _SERVER_OBJ_RE.search(m.group("obj")):
            continue
        # Arguments often continue over several lines (`router.get(\n  '/x',\n  auth, ...`).
        rest = text[m.end():m.end() + 600].lstrip()
        if not rest or rest[0] in "{[":
            continue  # `api.post('/x', { body })` is a client call, not a registration
        obj = m.group("obj")
        routes.append(Route(m.group("m").upper(), m.group("path"), "Express-style", rel,
                            _line_at(text, m.start("path")), _last_identifier(rest),
                            mounted=not re.fullmatch(r"(?:app|server|fastify|hono)", obj, re.I), obj=obj))
    for m in _JSX_ROUTE_RE.finditer(text):
        comp = re.search(r"(?:element\s*=\s*\{\s*<\s*|component\s*=\s*\{\s*)([A-Z][\w$]*)", m.group("rest"))
        routes.append(Route(PAGE, m.group("path"), "react-router", rel,
                            _line_at(text, m.start("path")), comp.group(1) if comp else None))
    for m in _FILE_ROUTE_RE.finditer(text):
        # `_layout` segments are pathless layouts; `$id` is a parameter.
        segs = [f":{seg[1:]}" if seg.startswith("$") and len(seg) > 1 else seg
                for seg in m.group("path").split("/") if seg and not seg.startswith("_")]
        comp = _COMPONENT_KEY_RE.search(text, m.end(), m.end() + 400)
        routes.append(Route(PAGE, "/" + "/".join(segs), "TanStack Router", rel,
                            _line_at(text, m.start("path")), comp.group(1) if comp else None))
    for m in _ROUTE_OBJ_RE.finditer(text):
        routes.append(Route(PAGE, m.group("path"), "client router", rel,
                            _line_at(text, m.start("path")), m.group("comp")))
    return routes


def _call_segments(raw: str, concat_tail: bool) -> tuple[list[str], bool] | None:
    s = raw.strip()
    prefix_unknown = False
    m = re.match(r"https?://[^/]+", s)
    if m:
        s, prefix_unknown = s[m.end():], True
    elif s.startswith("${"):
        close = s.find("}")
        if close < 0:
            return None
        s, prefix_unknown = s[close + 1:], True
    if not s.startswith("/"):
        return None
    s = re.split(r"[?#]", s, maxsplit=1)[0]
    # `${id}` (template), `{id}` / `:id` (generated clients keep placeholders).
    segs = [DYN if "${" in seg or seg[0] + seg[-1] == "{}" or seg.startswith(":") else seg
            for seg in s.split("/") if seg]
    if concat_tail and s.endswith("/"):
        segs.append(DYN)
    return segs, prefix_unknown


def scan_calls(rel: str, text: str) -> list[Call]:
    """HTTP requests and page navigations made from one JS/TS file."""
    if PurePosixPath(rel).suffix.lower() not in JS_SUFFIXES:
        return []
    calls: list[Call] = []

    def add(m, method: str):
        tail = text[m.end():m.end() + 3].lstrip()
        parsed = _call_segments(m.group("path"), concat_tail=tail.startswith("+"))
        if parsed:
            calls.append(Call(method, parsed[0], parsed[1], rel, _line_at(text, m.start("path")), m.group("path")))

    for m in _FETCH_RE.finditer(text):
        opts = _METHOD_OPT_RE.search(text[m.end():m.end() + 300].split(")", 1)[0])
        add(m, opts.group(1).upper() if opts else "GET")
    for m in _CLIENT_RE.finditer(text):
        if _SERVER_OBJ_RE.search(m.group("obj")):
            continue  # app.get / router.post: a registration, handled by scan_routes
        method = m.group("m").upper()
        add(m, "DELETE" if method == "DEL" else method)  # superagent spells it .del()
    for m in _URL_PROP_RE.finditer(text):
        if text[m.end():m.end() + 40].lstrip().startswith(";"):
            continue  # `url: '/x';` is a TypeScript type member, not a request
        # The `method:` key must sit in the same object literal as `url:`.
        open_at = text.rfind("{", max(0, m.start() - 400), m.start())
        close_at = text.find("}", m.end())
        obj = text[open_at if open_at >= 0 else m.start():close_at if close_at >= 0 else m.end()]
        opt = _METHOD_OPT_RE.search(obj)
        call = list(_OBJ_METHOD_CALL_RE.finditer(text[max(0, m.start() - 400):m.start()]))
        add(m, opt.group(1).upper() if opt else call[-1].group(1).upper() if call else "GET")
    for m in _NAV_RE.finditer(text):
        add(m, NAV)
    return calls


# ── Matching ─────────────────────────────────────────────────────────────────

def route_segments(path: str) -> list[str]:
    segs: list[str] = []
    for seg in path.split("/"):
        if not seg:
            continue
        if (seg in ("*", "**") or seg.endswith(("*", "+")) or seg.startswith("[...")
                or seg.startswith("[[...") or re.fullmatch(r"\{\w+:path\}", seg)):
            segs.append(REST)
            break
        if (seg.startswith(":") or seg[0] + seg[-1] in ("{}", "<>", "[]")):
            segs.append(PARAM)
        else:
            segs.append(seg)
    return segs


def _match(call: list[str], route: list[str]) -> int | None:
    """Number of literal segments matched, or None when they do not match."""
    literal = 0
    for i, rs in enumerate(route):
        if rs == REST:
            return literal if len(call) > i else None
        if i >= len(call):
            return None
        cs = call[i]
        if rs == PARAM:
            continue
        if cs != rs:
            return None  # a dynamic call segment only matches a parameter
        literal += 1
    return literal if len(call) == len(route) else None


def _method_ok(call: str, route: str) -> bool:
    if call == NAV or route == PAGE:
        return call == NAV and route == PAGE
    return route in _ANY or call == route


def match_call(call: Call, routes: list[Route]) -> list[Route]:
    """Best-matching routes for a call: most literal segments win; an unknown
    prefix (``${API}/users``) matches a route's path suffix with at least one
    literal segment. Ties are all returned."""
    scored: list[tuple[int, Route]] = []
    for route in routes:
        if not _method_ok(call.method, route.method):
            continue
        rsegs = route_segments(route.path)
        best = None
        # (call offset, route offset): an unknown call prefix (`${API}/users`)
        # drops leading route segments; a mounted route (`router.get` under
        # `app.use('/api', router)`) drops leading call segments.
        pairs = [(0, 0)]
        if call.prefix_unknown:
            pairs += [(0, k) for k in range(1, len(rsegs) + 1) if REST not in rsegs[:k]]
        if route.mounted:
            pairs += [(j, 0) for j in range(1, len(call.segments) + 1)]
        for j, k in pairs:
            lit = _match(call.segments[j:], rsegs[k:])
            if lit is not None and (j == k == 0 or lit > 0):
                score = lit * 10 - j - k
                best = score if best is None else max(best, score)
        if best is not None:
            scored.append((best, route))
    if not scored:
        return []
    top = max(s for s, _ in scored)
    return [r for s, r in scored if s == top]


# ── Graph integration ────────────────────────────────────────────────────────

def _enclosing(nodes_by_line: list[tuple[int, int, str]], line: int) -> str | None:
    best = None
    for start, end, nid in nodes_by_line:
        if start <= line <= end and (best is None or start >= best[0]):
            best = (start, nid)
    return best[1] if best else None


def _callable_spans(nodes: list[dict], lines: list[str]) -> list[tuple[int, int, str]]:
    from graphify.explain import snippet_from_lines
    spans = []
    for n in nodes:
        label = str(n.get("label") or "")
        m = re.search(r"(\d+)", str(n.get("source_location") or ""))
        if not label.endswith("()") or not m:
            continue
        start = int(m.group(1))
        snip = snippet_from_lines(lines, start, max_lines=100_000)
        if snip:
            spans.append((start, snip["start"] + len(snip["lines"]) - 1, n["id"]))
    return spans


def _edge(src: str, tgt: str, relation: str, confidence: str, rel: str, line: int) -> dict:
    return {"source": src, "target": tgt, "relation": relation, "confidence": confidence,
            "source_file": rel, "source_location": f"L{line}", "weight": 1.0}


def link_routes(paths: list[Path], root: Path, nodes: list[dict], edges: list[dict],
                context_nodes: list[dict] | None = None) -> None:
    """Add route nodes and handled_by / requests / navigates edges in place.

    ``nodes``/``edges`` are extract()'s output with source_file already
    relative to ``root``. ``context_nodes`` (an incremental rebuild's
    unchanged corpus) contributes its route nodes and handler candidates, so
    a changed caller still links to a route in an unchanged file.
    """
    texts: dict[str, str] = {}
    for p in paths:
        p = Path(p)
        suffix = p.suffix.lower()
        if suffix not in JS_SUFFIXES and suffix not in PY_SUFFIXES:
            continue
        try:
            rel = (p if p.is_absolute() else Path.cwd() / p).resolve().relative_to(root).as_posix()
            if p.stat().st_size > _MAX_BYTES:
                continue
            texts[rel] = p.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue
    if not texts:
        return

    by_file: dict[str, list[dict]] = {}
    by_label: dict[str, list[dict]] = {}
    for n in [*nodes, *(context_nodes or [])]:
        by_file.setdefault(str(n.get("source_file") or ""), []).append(n)
        by_label.setdefault(str(n.get("label") or ""), []).append(n)
    node_ids = {n.get("id") for n in nodes}

    routes: list[Route] = []
    next_dirs: dict[Path, bool] = {}
    for rel, text in texts.items():
        routes.extend(scan_routes(rel, text, nextjs=_in_nextjs_project(root, rel, next_dirs)))
    by_mod_file, by_name = express_mounts(texts)
    for r in routes:
        if r.mounted and r.obj:
            prefix = by_mod_file.get(r.file) or by_name.get(r.obj)
            if prefix:
                r.mount(prefix)
    seen: set[str] = set()
    for r in routes:
        if r.id in seen:
            continue
        seen.add(r.id)
        nodes.append({
            "id": r.id, "label": r.label, "file_type": "code", "source_file": r.file,
            "source_location": f"L{r.line}", "route_method": r.method, "route_path": r.path,
            "route_framework": r.framework,
            **({"route_mounted": True} if r.mounted else {}),
        })
        file_id = _make_id(_file_stem(Path(r.file)))
        if file_id in node_ids:
            edges.append(_edge(file_id, r.id, "contains", "EXTRACTED", r.file, r.line))
        if r.handler:
            handler, conf = _resolve_handler(r, by_file, by_label)
            if handler:
                edges.append(_edge(r.id, handler, "handled_by", conf, r.file, r.line))

    # Routes from unchanged files (incremental rebuild) are match targets too.
    fresh_files = set(texts)
    targets = list(routes)
    for n in context_nodes or []:
        if n.get("route_path") and n.get("source_file") not in fresh_files:
            line = int(re.sub(r"\D", "", str(n.get("source_location") or "")) or 0)
            targets.append(Route(str(n.get("route_method") or ""), str(n["route_path"]),
                                 str(n.get("route_framework") or ""), str(n["source_file"]), line,
                                 id=n["id"], mounted=bool(n.get("route_mounted"))))
    if not targets:
        return
    seen_edges: set[tuple[str, str, str]] = set()
    for rel, text in texts.items():
        calls = scan_calls(rel, text)
        if not calls:
            continue
        spans = _callable_spans(by_file.get(rel, []), text.splitlines())
        file_id = _make_id(_file_stem(Path(rel)))
        for call in calls:
            caller = _enclosing(spans, call.line) or (file_id if file_id in node_ids else None)
            if caller is None:
                continue
            relation = "navigates" if call.method == NAV else "requests"
            for route in match_call(call, targets):
                key = (caller, route.id, relation)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                edges.append(_edge(caller, route.id, relation, "INFERRED", rel, call.line))


def _resolve_module(importer: str, mod: str, known: set[str]) -> str | None:
    base = PurePosixPath(importer).parent / mod
    parts: list[str] = []
    for part in base.parts:
        if part == "..":
            if parts:
                parts.pop()
        elif part != ".":
            parts.append(part)
    stem = "/".join(parts)
    for cand in (stem, *(stem + ext for ext in (".ts", ".js", ".tsx", ".jsx", ".mjs", ".cjs")),
                 *(f"{stem}/index{ext}" for ext in (".ts", ".js", ".tsx", ".jsx"))):
        if cand in known:
            return cand
    return None


def express_mounts(texts: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """Prefixes routers are mounted under via ``x.use('/prefix', router)``.

    Returns (router module file -> prefix, router variable name -> prefix).
    The file map comes from following the mounted identifier's import, so a
    default-exported router renamed on import still resolves; the name map is
    the fallback for a router declared in the mounting file or imported under
    its own name from somewhere the scan did not reach.
    """
    known = set(texts)
    by_file: dict[str, str] = {}
    by_name: dict[str, str] = {}
    for rel, text in texts.items():
        mounts = [(m.group("path"), _last_identifier(m.group("rest"))) for m in _MOUNT_RE.finditer(text)]
        mounts = [(prefix, ident) for prefix, ident in mounts if ident]
        if not mounts:
            continue
        imported: dict[str, str] = {}
        for m in (*_IMPORT_RE.finditer(text), *_REQUIRE_RE.finditer(text)):
            target = _resolve_module(rel, m.group("mod"), known)
            if not target:
                continue
            names = [m.group("default")] if m.group("default") else []
            for item in (m.group("named") or "").split(","):
                item = item.strip()
                if item:
                    names.append(item.split(" as ")[-1].strip())
            for name in names:
                imported[name] = target
        for prefix, ident in mounts:
            by_name.setdefault(ident, prefix)
            if ident in imported:
                by_file.setdefault(imported[ident], prefix)
    return by_file, by_name


def _in_nextjs_project(root: Path, rel: str, memo: dict[Path, bool]) -> bool:
    """True when a directory between the file and the project root holds a
    ``next.config.*`` (monorepos keep it in ``apps/web/`` and the like)."""
    d = (root / rel).parent
    chain = []
    while True:
        if d in memo:
            found = memo[d]
            break
        chain.append(d)
        if any((d / f"next.config.{ext}").is_file() for ext in ("js", "mjs", "cjs", "ts")):
            found = True
            break
        if d == root or root not in d.parents:
            found = False
            break
        d = d.parent
    for c in chain:
        memo[c] = found
    return found


def _resolve_handler(route: Route, by_file: dict, by_label: dict) -> tuple[str | None, str]:
    """The node for a route's named handler: the same file first (EXTRACTED),
    else a unique same-named function anywhere (INFERRED, typically an
    imported controller)."""
    # `name()` for functions/methods, bare `Name` for a class (a React class component).
    names = (f"{route.handler}()", f".{route.handler}()", route.handler)
    local = [n for n in by_file.get(route.file, []) if n.get("label") in names]
    if local:
        return local[0]["id"], "EXTRACTED"
    found = [n for name in names for n in by_label.get(name, []) if n.get("source_file")]
    if len(found) == 1:
        return found[0]["id"], "INFERRED"
    return None, ""


def route_table(G) -> list[dict]:
    """Every route node with its handler(s), callers and navigators, sorted by
    path — the data behind ``graphify routes`` and the graph.html overview."""
    from graphify.explain import relation_index
    index = relation_index(G)
    rows = []
    for nid, d in G.nodes(data=True):
        if not d.get("route_path"):
            continue
        rels = index.get(nid, [])
        rows.append({
            "id": nid,
            "method": str(d.get("route_method") or ""),
            "path": str(d.get("route_path")),
            "framework": str(d.get("route_framework") or ""),
            "source_file": str(d.get("source_file") or ""),
            "source_location": str(d.get("source_location") or ""),
            "handlers": sorted({nb for direction, r, nb, _ in rels if direction == "out" and r == "handled_by"}),
            "callers": sorted({nb for direction, r, nb, _ in rels
                               if direction == "in" and r in ("requests", "navigates")}),
        })
    rows.sort(key=lambda r: (r["method"] == PAGE, r["path"], r["method"]))
    return rows


def format_routes(G, *, max_callers: int = 5) -> list[str]:
    """Text route map: each API route and page with the code that serves it
    and the code that calls or links to it. Shared by ``graphify routes`` and
    the MCP ``list_routes`` tool."""
    from graphify.security import sanitize_label

    def name(nid: str) -> str:
        return sanitize_label(str(G.nodes[nid].get("label") or nid)) if nid in G else str(nid)

    def where(nid: str) -> str:
        d = G.nodes[nid] if nid in G else {}
        return sanitize_label(str(d.get("source_file") or ""))

    rows = route_table(G)
    if not rows:
        return ["No routes found. graphify detects Express/Koa/Fastify/Hono-style "
                "`app.get('/x', handler)`, Next.js app/ and pages/ routes, FastAPI/Flask "
                "decorators, and react-router / vue-router `path:` declarations."]
    api = [r for r in rows if r["method"] != PAGE]
    pages = [r for r in rows if r["method"] == PAGE]
    out: list[str] = []
    for title, group in (("API routes", api), ("Pages", pages)):
        if not group:
            continue
        out.append(f"{title} ({len(group)}):")
        width = max(len(r["path"]) for r in group)
        for r in group:
            method = "" if r["method"] == PAGE else f"{r['method']:<7}"
            loc = f"{sanitize_label(r['source_file'])}:{r['source_location'].lstrip('L')}"
            handler = ", ".join(name(h) for h in r["handlers"]) or "inline handler"
            out.append(f"  {method}{sanitize_label(r['path']):<{width}}  {loc}  -> {handler}")
            callers = r["callers"]
            if callers:
                shown = ", ".join(f"{name(c)} ({where(c)})" for c in callers[:max_callers])
                extra = f" (+{len(callers) - max_callers} others)" if len(callers) > max_callers else ""
                verb = "opened from" if r["method"] == PAGE else "called from"
                out.append(f"  {'':<{len(method)}}{'':<{width}}  {verb}: {shown}{extra}")
        out.append("")
    uncalled = sum(1 for r in api if not r["callers"])
    if uncalled:
        out.append(f"{uncalled} API route(s) have no caller in the scanned code "
                   "(served to another app, an external client, or unused).")
    if any(G.nodes[r["id"]].get("route_mounted") for r in rows):
        out.append("Some routes sit on a router mounted elsewhere; their full URL has a prefix "
                   "graphify could not resolve.")
    return out
