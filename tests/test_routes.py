"""Web route map (graphify/routes.py): route detection, call matching, and the
route/handler/caller edges extract() adds."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphify.routes import (
    NAV,
    PAGE,
    Call,
    Route,
    express_mounts,
    format_routes,
    match_call,
    scan_calls,
    scan_routes,
)


def _labels(routes):
    return sorted((r.method, r.path, r.handler) for r in routes)


# ── Detection ────────────────────────────────────────────────────────────────

def test_express_registrations_and_handlers():
    src = (
        "const app = express();\n"
        "app.get('/api/health', (req, res) => res.json({}));\n"
        "usersRouter.post('/', requireAuth, users.create);\n"
        "app.get('/api/items/:id', getItem);\n"
        "app.use('/api/users', usersRouter);\n"
    )
    routes = scan_routes("server/index.ts", src)
    assert _labels(routes) == [
        ("GET", "/api/health", None),
        ("GET", "/api/items/:id", "getItem"),
        ("POST", "/", "create"),
    ]
    by_path = {r.path: r for r in routes}
    assert by_path["/"].mounted and by_path["/"].obj == "usersRouter"
    assert not by_path["/api/health"].mounted


def test_client_calls_are_not_registrations():
    src = "api.post('/users', payload);\naxios.get('/users', { params });\nrequests.del(`/users/${id}`);\n"
    assert scan_routes("web/api.ts", src) == []
    calls = scan_calls("web/api.ts", src)
    assert [(c.method, c.segments) for c in calls] == [
        ("POST", ["users"]), ("GET", ["users"]), ("DELETE", ["users", "${}"]),
    ]


def test_nextjs_routes_need_next_config():
    route_src = "export async function GET(req) {}\nexport const POST = async () => {};\n"
    page_src = "export default function UserPage() { return null }\n"
    assert scan_routes("app/api/users/[id]/route.ts", route_src) == []
    api = scan_routes("app/api/users/[id]/route.ts", route_src, nextjs=True)
    assert _labels(api) == [("GET", "/api/users/:id", "GET"), ("POST", "/api/users/:id", "POST")]
    page = scan_routes("src/app/(shop)/products/[...slug]/page.tsx", page_src, nextjs=True)
    assert _labels(page) == [(PAGE, "/products/:slug*", "UserPage")]
    legacy = scan_routes("pages/api/login.ts", "export default async function handler(req, res) {}",
                         nextjs=True)
    assert _labels(legacy) == [("ANY", "/api/login", "handler")]
    assert scan_routes("pages/_app.tsx", page_src, nextjs=True) == []


def test_python_fastapi_and_flask():
    fastapi = (
        "from fastapi import FastAPI, APIRouter\n"
        "app = FastAPI()\n"
        'router = APIRouter(prefix="/orders")\n'
        '@app.get("/status")\n'
        "def status():\n    return {}\n"
        '@router.post("/{order_id}/pay")\n'
        "async def pay(order_id: int):\n    return {}\n"
    )
    routes = scan_routes("api/main.py", fastapi)
    assert _labels(routes) == [("GET", "/status", "status"), ("POST", "/orders/{order_id}/pay", "pay")]
    assert {r.path: r.framework for r in routes}["/status"] == "FastAPI"
    flask = (
        "from flask import Flask, Blueprint\n"
        "bp = Blueprint('users', __name__, url_prefix='/users')\n"
        "@bp.route('/<int:uid>', methods=['GET', 'DELETE'])\n"
        "def user(uid):\n    pass\n"
    )
    assert _labels(scan_routes("api/users.py", flask)) == [
        ("DELETE", "/users/<int:uid>", "user"), ("GET", "/users/<int:uid>", "user"),
    ]


def test_client_side_router_pages():
    jsx = '<Route path="/users/:id" element={<UserPage />} />\n'
    objs = "const routes = [{ path: '/settings', component: Settings }];\n"
    assert _labels(scan_routes("src/App.tsx", jsx)) == [(PAGE, "/users/:id", "UserPage")]
    assert _labels(scan_routes("src/router.ts", objs)) == [(PAGE, "/settings", "Settings")]
    tanstack = 'export const Route = createFileRoute("/_layout/items/$itemId")({\n  component: Item,\n})\n'
    assert _labels(scan_routes("src/routes/_layout/items.$itemId.tsx", tanstack)) == [
        (PAGE, "/items/:itemId", "Item"),
    ]


def test_call_shapes():
    src = (
        "fetch(`/api/users/${id}?full=1`);\n"
        "fetch('/api/users/' + id, { method: 'DELETE' });\n"
        "fetch(`${API_URL}/health`);\n"
        "useSWR('/api/me');\n"
        "<Link href={`/users/${u.id}`}>x</Link>\n"
        "router.push('/login');\n"
    )
    got = [(c.method, c.segments, c.prefix_unknown) for c in scan_calls("web/a.tsx", src)]
    assert ("GET", ["api", "users", "${}"], False) in got
    assert ("DELETE", ["api", "users", "${}"], False) in got
    assert ("GET", ["health"], True) in got
    assert ("GET", ["api", "me"], False) in got
    assert (NAV, ["users", "${}"], False) in got
    assert (NAV, ["login"], False) in got
    assert scan_calls("api/main.py", src) == []


def test_request_option_objects():
    src = (
        "return (options.client ?? client).post<Res, Err>({\n"
        "    responseType: 'json',\n"
        "    url: '/api/v1/password-recovery/{email}',\n"
        "});\n"
        "axios({ method: 'put', url: `/api/items/${id}` });\n"
        "export type ReadItemsData = { url: '/api/v1/items/'; };\n"
    )
    got = [(c.method, c.segments) for c in scan_calls("web/sdk.gen.ts", src)]
    assert got == [("POST", ["api", "v1", "password-recovery", "${}"]), ("PUT", ["api", "items", "${}"])]


# ── Matching ─────────────────────────────────────────────────────────────────

def _route(method, path, **kw):
    return Route(method, path, "test", "f.ts", 1, **kw)


def _call(method, segs, prefix_unknown=False):
    return Call(method, segs, prefix_unknown, "c.ts", 1)


def test_match_param_literal_and_method():
    by_id = _route("GET", "/users/:id")
    me = _route("GET", "/users/me")
    post = _route("POST", "/users/:id")
    routes = [by_id, me, post]
    assert match_call(_call("GET", ["users", "${}"]), routes) == [by_id]   # dynamic only hits a param
    assert match_call(_call("GET", ["users", "me"]), routes) == [me]      # most literal segments win
    assert match_call(_call("POST", ["users", "7"]), routes) == [post]
    assert match_call(_call("GET", ["orders", "1"]), routes) == []
    assert match_call(_call("PUT", ["users", "1"]), [_route("ALL", "/users/:id")])  # ALL takes any method


def test_match_pages_only_from_navigation():
    page = _route(PAGE, "/users/:id")
    assert match_call(_call(NAV, ["users", "${}"]), [page]) == [page]
    assert match_call(_call("GET", ["users", "1"]), [page]) == []
    assert match_call(_call(NAV, ["api", "x"]), [_route("GET", "/api/x")]) == []


def test_match_unknown_prefixes():
    health = _route("GET", "/api/health")
    assert match_call(_call("GET", ["health"], prefix_unknown=True), [health]) == [health]
    mounted = _route("GET", "/:id/pay", mounted=True)
    assert match_call(_call("GET", ["api", "orders", "${}", "pay"]), [mounted]) == [mounted]
    # a mounted route needs at least one literal segment to match a suffix
    assert match_call(_call("GET", ["api", "orders", "${}"]), [_route("GET", "/:id", mounted=True)]) == []


def test_catch_all():
    rest = _route("GET", "/files/*")
    assert match_call(_call("GET", ["files", "a", "b"]), [rest]) == [rest]
    assert match_call(_call("GET", ["files"]), [rest]) == []


def test_express_mounts_follow_imports():
    texts = {
        "server/index.ts": "import users from './routes/users';\napp.use('/api/users', auth, users);\n",
        "server/routes/users.ts": "const router = Router();\nrouter.get('/:id', getUser);\nexport default router;\n",
    }
    by_file, by_name = express_mounts(texts)
    assert by_file == {"server/routes/users.ts": "/api/users"}
    assert by_name == {"users": "/api/users"}
    r = scan_routes("server/routes/users.ts", texts["server/routes/users.ts"])[0]
    old_id = r.id
    r.mount(by_file[r.file])
    assert (r.path, r.mounted) == ("/api/users/:id", False) and r.id != old_id


# ── extract() integration ────────────────────────────────────────────────────

def _webapp(tmp_path: Path) -> Path:
    files = {
        "server/index.ts": (
            "import express from 'express';\n"
            "import { usersRouter } from './users';\n"
            "const app = express();\n"
            "app.use('/api/users', usersRouter);\n"
        ),
        "server/users.ts": (
            "import { Router } from 'express';\n"
            "export const usersRouter = Router();\n"
            "export async function getUser(req, res) {\n"
            "  res.json({ id: req.params.id });\n"
            "}\n"
            "usersRouter.get('/:id', getUser);\n"
        ),
        "web/api.ts": (
            "export async function loadUser(id: string) {\n"
            "  const res = await fetch(`/api/users/${id}`);\n"
            "  return res.json();\n"
            "}\n"
        ),
        "web/App.tsx": (
            "import { Routes, Route, Link } from 'react-router-dom';\n"
            "export function UserPage() { return null; }\n"
            "export function App() {\n"
            "  return <Routes><Route path=\"/users/:id\" element={<UserPage />} /></Routes>;\n"
            "}\n"
            "export function Nav() {\n"
            "  return <Link to=\"/users/42\">me</Link>;\n"
            "}\n"
        ),
    }
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return tmp_path


def _extract(root: Path) -> dict:
    from graphify.extract import collect_files, extract
    return extract(collect_files(root), root=root, parallel=False)


def _edges(result, relation):
    labels = {n["id"]: n["label"] for n in result["nodes"]}
    return sorted((labels.get(e["source"], e["source"]), labels.get(e["target"], e["target"]), e["confidence"])
                  for e in result["edges"] if e["relation"] == relation)


def test_extract_adds_route_map(tmp_path):
    result = _extract(_webapp(tmp_path))
    routes = {n["label"]: n for n in result["nodes"] if n.get("route_path")}
    assert set(routes) == {"GET /api/users/:id", "PAGE /users/:id"}
    api = routes["GET /api/users/:id"]
    assert api["source_file"] == "server/users.ts" and api["source_location"] == "L6"
    assert api["route_framework"] == "Express-style" and "route_mounted" not in api
    assert api["_origin"] == "ast"
    assert _edges(result, "handled_by") == [
        ("GET /api/users/:id", "getUser()", "EXTRACTED"),
        ("PAGE /users/:id", "UserPage()", "EXTRACTED"),
    ]
    assert _edges(result, "requests") == [("loadUser()", "GET /api/users/:id", "INFERRED")]
    assert ("Nav()", "PAGE /users/:id", "INFERRED") in _edges(result, "navigates")
    assert ("users.ts", "GET /api/users/:id", "EXTRACTED") in _edges(result, "contains")

    # Route edges are schema-valid and never dangle. (extract() itself leaves
    # external-package import stubs like `ref_express` dangling; not ours.)
    from graphify.validate import validate_extraction
    ids = {n["id"] for n in result["nodes"]}
    route_edges = [e for e in result["edges"] if e["relation"] in ("handled_by", "requests", "navigates")
                   or str(e["target"]).startswith("route_")]
    assert validate_extraction({"nodes": result["nodes"], "edges": route_edges}) == []
    assert all(e["source"] in ids and e["target"] in ids for e in route_edges)


def test_route_explanations_and_listing(tmp_path):
    from graphify.build import build_from_json
    from graphify.explain import explain_node

    G = build_from_json(_extract(_webapp(tmp_path)))
    rid = next(n for n, d in G.nodes(data=True) if d.get("label") == "GET /api/users/:id")
    handler = next(n for n, d in G.nodes(data=True) if d.get("label") == "getUser()")
    caller = next(n for n, d in G.nodes(data=True) if d.get("label") == "loadUser()")

    route_ex = explain_node(G, rid)
    assert route_ex["kind"] == "API route"
    assert "The code that runs for it is getUser()." in route_ex["summary"]
    assert "The frontend reaches it from loadUser()." in route_ex["summary"]
    assert "runs whenever GET /api/users/:id is requested" in explain_node(G, handler)["summary"]
    caller_summary = explain_node(G, caller)["summary"]
    assert "It calls the backend: GET /api/users/:id." in caller_summary
    assert "self-contained" not in caller_summary

    text = "\n".join(format_routes(G))
    assert "API routes (1):" in text and "Pages (1):" in text
    assert "-> getUser()" in text and "called from: loadUser() (web/api.ts)" in text
    assert "opened from: Nav() (web/App.tsx)" in text


def test_html_overview_lists_routes(tmp_path):
    from graphify.build import build_from_json
    from graphify.export import to_html

    G = build_from_json(_extract(_webapp(tmp_path)))
    out = tmp_path / "graphify-out" / "graph.html"
    out.parent.mkdir()
    to_html(G, {0: list(G.nodes)}, str(out))
    html = out.read_text(encoding="utf-8")
    overview = json.loads(html.split("const OVERVIEW = ", 1)[1].split(";\n", 1)[0])
    assert overview["route_count"] == 2
    assert [G.nodes[r["id"]]["label"] for r in overview["routes"]] == ["GET /api/users/:id", "PAGE /users/:id"]
    assert overview["routes"][0]["callers"] == 1
    assert "Routes &amp; pages" in html or "Routes & pages" in html


def test_cli_and_mcp_list_routes(tmp_path):
    import subprocess
    import sys

    from graphify.build import build_from_json
    from graphify.export import to_json

    root = _webapp(tmp_path)
    G = build_from_json(_extract(root))
    gp = root / "graphify-out" / "graph.json"
    gp.parent.mkdir()
    to_json(G, {0: list(G.nodes)}, str(gp))

    out = subprocess.run([sys.executable, "-m", "graphify", "routes", "--graph", str(gp)],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "GET    /api/users/:id" in out.stdout and "-> getUser()" in out.stdout
    as_json = subprocess.run([sys.executable, "-m", "graphify", "routes", "--json", "--graph", str(gp)],
                             capture_output=True, text=True, timeout=120)
    rows = json.loads(as_json.stdout)
    assert [(r["method"], r["path"]) for r in rows] == [("GET", "/api/users/:id"), (PAGE, "/users/:id")]

    pytest.importorskip("mcp")
    pytest.importorskip("starlette")
    from graphify import serve as serve_mod
    from tests.test_serve_http import _call_tool, _client, _init_session

    app = serve_mod._build_http_app(str(gp), json_response=True)
    with _client(app) as client:
        headers = _init_session(client)
        text = _call_tool(client, headers, "list_routes", {}, rid=2)
    assert "GET    /api/users/:id" in text and "called from: loadUser()" in text
