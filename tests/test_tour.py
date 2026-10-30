"""Guided tours (graphify/tour.py)."""
from __future__ import annotations

import json
import subprocess
import sys

import networkx as nx
import pytest

from graphify.tour import build_tour, format_tour, suggest_starts, suggest_tours
from tests.test_routes import _extract, _webapp


def _tour_app(tmp_path):
    """The route-map fixture, with the page actually loading its data."""
    root = _webapp(tmp_path)
    (root / "web" / "App.tsx").write_text(
        "import { Routes, Route, Link } from 'react-router-dom';\n"
        "import { loadUser } from './api';\n"
        "export function UserPage() {\n"
        "  return loadUser('1');\n"
        "}\n"
        "export function App() {\n"
        "  return <Routes><Route path=\"/users/:id\" element={<UserPage />} /></Routes>;\n"
        "}\n",
        encoding="utf-8",
    )
    return root


def _web_graph(tmp_path):
    from graphify.build import build_from_json
    return build_from_json(_extract(_tour_app(tmp_path)))


def _by_label(G, label):
    return next(n for n, d in G.nodes(data=True) if d.get("label") == label)


def _walk(G, tour):
    return [G.nodes[s["id"]]["label"] for s in tour["steps"]]


def test_tour_follows_page_to_handler_through_the_api(tmp_path):
    G = _web_graph(tmp_path)
    tour = build_tour(G, _by_label(G, "PAGE /users/:id"))
    assert _walk(G, tour)[:2] == ["PAGE /users/:id", "UserPage()"]
    # The request is followed all the way to its handler before anything else.
    walk = _walk(G, tour)
    assert walk.index("UserPage()") < walk.index("GET /api/users/:id") < walk.index("getUser()")
    reasons = [s["reason"] for s in tour["steps"]]
    assert reasons[0] == "Start at the /users/:id page: this is where the flow enters the code."
    assert "The /users/:id page is served by UserPage()." in reasons
    assert "GET /api/users/:id is served by getUser()." in reasons
    assert any(r.startswith("loadUser() sends a request to GET /api/users/:id") for r in reasons)
    assert tour["title"] == "Follow the /users/:id page"


def test_suggested_starts_are_entry_points(tmp_path):
    G = _web_graph(tmp_path)
    starts = [G.nodes[s]["label"] for s in suggest_starts(G)]
    assert starts[0] == "PAGE /users/:id"
    assert "getUser()" not in starts and "loadUser()" not in starts  # both are mid-flow


def _toy(extra_nodes=(), extra_edges=()):
    G = nx.DiGraph()
    def node(nid, label, src="src/app.py", ft="code"):
        G.add_node(nid, label=label, source_file=src, file_type=ft, source_location="L1")
    node("main", "main()")
    node("a", "load()")
    node("b", "parse()")
    node("util", "helper()")
    for nid, label, src, ft in extra_nodes:
        node(nid, label, src, ft)
    for u, v, rel in [("main", "a", "calls"), ("a", "b", "calls"), *extra_edges]:
        G.add_edge(u, v, relation=rel, confidence="EXTRACTED")
    return G


def test_tests_generated_code_and_packages_are_not_tour_material():
    G = _toy(
        extra_nodes=[("t", "test_load()", "tests/test_app.py", "code"),
                     ("gen", "client()", "web/sdk.gen.ts", "code"),
                     ("pkg", "react", "package.json", "concept"),
                     ("x1", "x1()", "src/x.py", "code"), ("x2", "x2()", "src/x.py", "code"),
                     ("x3", "x3()", "src/x.py", "code")],
        extra_edges=[("t", "a", "calls"), ("t", "util", "calls"), ("gen", "util", "calls"),
                     ("x1", "pkg", "imports"), ("x2", "pkg", "imports"), ("x3", "pkg", "imports"),
                     ("x1", "util", "calls"), ("x2", "util", "calls"), ("a", "pkg", "calls")],
    )
    assert "t" not in suggest_starts(G) and "gen" not in suggest_starts(G)
    tour = build_tour(G, "main")
    ids = [s["id"] for s in tour["steps"]]
    assert ids[:3] == ["main", "a", "b"]
    assert "pkg" not in ids                     # a package.json concept is not code to read
    foundations = [s for s in tour["steps"] if s["via"] == "foundation"]
    assert [s["id"] for s in foundations] == ["util"] and len(foundations) <= 2


def test_same_named_steps_say_which_file():
    G = _toy(extra_nodes=[("crud", "load()", "src/crud.py", "code")], extra_edges=[("a", "crud", "calls")])
    reasons = [s["reason"] for s in build_tour(G, "main")["steps"]]
    assert "load() (app.py) calls load() (crud.py)." in reasons


def test_step_budget_and_distinct_tours():
    G = _toy(extra_nodes=[("other", "cli()", "src/cli.py", "code"), ("o2", "run()", "src/cli.py", "code")],
             extra_edges=[("other", "o2", "calls")])
    assert len(build_tour(G, "main", max_steps=2)["steps"]) == 2
    titles = [t["title"] for t in suggest_tours(G)]
    assert titles == ["Follow main() (app.py)", "Follow cli() (cli.py)"]


def test_format_tour_text(tmp_path):
    G = _web_graph(tmp_path)
    text = "\n".join(format_tour(G, build_tour(G, _by_label(G, "PAGE /users/:id"), max_steps=4)))
    assert text.startswith("Tour: Follow the /users/:id page (4 steps)")
    assert "Step 1/4: PAGE /users/:id  [page · web/App.tsx:7]" in text
    assert "Step 2/4: UserPage()  [UI component" in text
    assert "  Why: The /users/:id page is served by UserPage()." in text
    assert "is a UI component" in text


def test_html_overview_offers_tours(tmp_path):
    from graphify.export import to_html
    G = _web_graph(tmp_path)
    out = tmp_path / "graphify-out" / "graph.html"
    out.parent.mkdir()
    to_html(G, {0: list(G.nodes)}, str(out))
    html = out.read_text(encoding="utf-8")
    overview = json.loads(html.split("const OVERVIEW = ", 1)[1].split(";\n", 1)[0])
    assert overview["tours"] and all(t["steps"] for t in overview["tours"])
    step = overview["tours"][0]["steps"][1]
    assert set(step) == {"id", "reason", "from"} and step["from"] == overview["tours"][0]["steps"][0]["id"]
    assert "function showTourStep()" in html and "data-tour-nav" in html
    assert "\x00" not in html


def test_cli_and_mcp_tour(tmp_path):
    from graphify.export import to_json
    root = _tour_app(tmp_path)
    from graphify.build import build_from_json
    G = build_from_json(_extract(root))
    gp = root / "graphify-out" / "graph.json"
    gp.parent.mkdir()
    to_json(G, {0: list(G.nodes)}, str(gp))

    def run(*args):
        return subprocess.run([sys.executable, "-m", "graphify", "tour", *args, "--graph", str(gp)],
                              capture_output=True, text=True, timeout=120)

    listed = run("--list")
    assert listed.returncode == 0 and listed.stdout.startswith("1. Follow ")
    full = run("loadUser", "--steps", "3")
    assert full.returncode == 0, full.stderr
    assert "Tour: Follow loadUser() (api.ts) (3 steps)" in full.stdout
    assert "Why: loadUser() sends a request to GET /api/users/:id." in full.stdout
    assert "Code (web/api.ts:1):" in full.stdout
    assert "Code (" not in run("loadUser", "--no-code").stdout
    assert run("no_such_symbol_anywhere").returncode == 4

    pytest.importorskip("mcp")
    pytest.importorskip("starlette")
    from graphify import serve as serve_mod
    from tests.test_serve_http import _call_tool, _client, _init_session
    app = serve_mod._build_http_app(str(gp), json_response=True)
    with _client(app) as client:
        headers = _init_session(client)
        text = _call_tool(client, headers, "get_tour", {"start": "loadUser()", "steps": 3}, rid=2)
        suggested = _call_tool(client, headers, "get_tour", {}, rid=3)
    assert text.startswith("Tour: Follow loadUser()") and "Code (" not in text
    assert suggested.startswith("Tour: Follow ")
