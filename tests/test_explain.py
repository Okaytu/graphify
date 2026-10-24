"""Plain-language node explanations (graphify/explain.py) and their use in
graph.html and `graphify explain`."""
from __future__ import annotations

import json
import re

import networkx as nx

from graphify.explain import (
    codebase_overview,
    explain_node,
    leading_comment,
    node_kind,
    relation_index,
    resolve_source,
    snippet_from_lines,
    SnippetReader,
    summarize_node,
)
from graphify.export import to_html


def _web_graph() -> nx.Graph:
    """Undirected like build() output: true direction lives in _src/_tgt."""
    G = nx.Graph()
    G.add_node("api", label="api.ts", file_type="code", source_file="src/api.ts", source_location="L1")
    G.add_node("load", label="loadProfile()", file_type="code", source_file="src/api.ts", source_location="L4")
    G.add_node("val", label="validateUser()", file_type="code", source_file="src/auth.ts", source_location="L1")
    G.add_node("card", label="ProfileCard()", file_type="code", source_file="src/Card.tsx", source_location="L1")
    G.add_node("store", label="ProfileStore", file_type="code", source_file="src/api.ts", source_location="L10",
               _callable_class=True)
    G.add_node("get", label=".get()", file_type="code", source_file="src/api.ts", source_location="L11")
    G.add_node("why", label="Keeps the payload shape stable", file_type="rationale",
               source_file="src/auth.ts", source_location="L2")

    def edge(s, t, rel):
        # Insert reversed so arc order disagrees with the true direction.
        G.add_edge(t, s, relation=rel, confidence="EXTRACTED", _src=s, _tgt=t)

    edge("api", "load", "contains")
    edge("api", "store", "contains")
    edge("store", "get", "method")
    edge("get", "load", "calls")
    edge("load", "val", "calls")
    edge("card", "load", "calls")
    edge("why", "val", "rationale_for")
    return G


def test_relation_index_uses_true_direction():
    G = _web_graph()
    idx = relation_index(G)
    assert ("out", "calls", "val", G.edges["load", "val"]) in idx["load"]
    assert ("in", "calls", "load", G.edges["load", "val"]) in idx["val"]


def test_node_kinds():
    G = _web_graph()
    idx = relation_index(G)
    assert node_kind(G, "api", idx["api"]) == "file"
    assert node_kind(G, "load", idx["load"]) == "function"
    assert node_kind(G, "store", idx["store"]) == "class"
    assert node_kind(G, "get", idx["get"]) == "method"
    assert node_kind(G, "card", idx["card"]) == "UI component"
    assert node_kind(G, "why", idx["why"]) == "comment"


def test_summary_describes_role_and_impact():
    G = _web_graph()
    idx = relation_index(G)
    s = summarize_node(G, "load", idx["load"], community_name="Profiles")
    assert s.startswith("loadProfile() is a function in src/api.ts (line 4).")
    assert "Profiles" in s
    assert "It calls validateUser()." in s
    assert "It is called by" in s and ".get()" in s and "ProfileCard()" in s
    assert "affects 2 dependents across 2 files" in s

    leaf = summarize_node(G, "val", idx["val"])
    assert "calls nothing else" in leaf

    entry = summarize_node(G, "card", idx["card"])
    assert "entry point" in entry

    method = summarize_node(G, "get", idx["get"])
    assert method.startswith(".get() is a method of ProfileStore")


def test_docs_come_from_rationale_nodes_and_attributes():
    G = _web_graph()
    G.nodes["load"]["rationale"] = "Single place that talks to the profile API."
    e_val = explain_node(G, "val")
    e_load = explain_node(G, "load")
    assert e_val["docs"] == ["Keeps the payload shape stable"]
    assert e_load["docs"] == ["Single place that talks to the profile API."]


def test_snippet_follows_body_and_leading_comments():
    ts = [
        "import x from 'y';",
        "/**",
        " * Loads a profile.",
        " */",
        "export async function loadProfile(",
        "  id: string,",
        ") {",
        "  return fetch(id);",
        "}",
        "",
        "export const other = 1;",
    ]
    snip = snippet_from_lines(ts, 5)
    assert snip["start"] == 2
    assert snip["lines"][-1] == "}"
    assert "other" not in "\n".join(snip["lines"])
    assert leading_comment(snip, 5) == "Loads a profile."

    py = [
        "class Handler:",
        '    """Class docstring, not part of handle()."""',
        "    @route('/x')",
        "    def handle(self):",
        "        return 1",
        "",
        "    def other(self):",
        "        return 2",
    ]
    snip = snippet_from_lines(py, 4)
    assert snip["start"] == 3  # decorator kept, enclosing docstring not
    assert snip["lines"] == ["    @route('/x')", "    def handle(self):", "        return 1"]
    assert leading_comment(snip, 4) == ""


def test_snippet_is_capped():
    lines = ["def big():"] + [f"    x = {i}" for i in range(100)]
    snip = snippet_from_lines(lines, 1, max_lines=10)
    assert len(snip["lines"]) == 10
    assert snip["truncated"] is True


def test_resolve_source_is_confined_to_project(tmp_path):
    out = tmp_path / "proj" / "graphify-out"
    out.mkdir(parents=True)
    (tmp_path / "proj" / "a.py").write_text("x = 1\n")
    (tmp_path / "secret.txt").write_text("nope\n")
    gp = out / "graph.json"
    assert resolve_source("a.py", gp) == tmp_path / "proj" / "a.py"
    assert resolve_source("../secret.txt", gp) is None
    assert resolve_source(str(tmp_path / "secret.txt"), gp) is None


def test_codebase_overview():
    G = _web_graph()
    ov = codebase_overview(G, {0: ["api", "load", "val", "store", "get"], 1: ["card", "why"]},
                           {0: "Profiles", 1: "UI"})
    assert ov["foundations"][0] == {"id": "load", "dependents": 2}
    assert {e["id"] for e in ov["entry_points"]} == {"get", "card"}
    assert ov["areas"][0]["name"] == "Profiles"
    assert ov["kinds"]["function"] == 2


def _project(tmp_path):
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "graphify-out").mkdir()
    (root / "src" / "api.ts").write_text(
        "import { validateUser } from './auth';\n"
        "\n"
        "/** Loads the profile. </script><script>alert(1)</script> */\n"
        "export async function loadProfile(id) {\n"
        "  return validateUser(await fetch(id));\n"
        "}\n",
        encoding="utf-8",
    )
    G = nx.Graph()
    G.add_node("load", label="loadProfile()", file_type="code", source_file="src/api.ts", source_location="L4")
    G.add_node("val", label="validateUser()", file_type="code", source_file="src/auth.ts", source_location="L1")
    G.add_edge("val", "load", relation="calls", confidence="EXTRACTED", _src="load", _tgt="val")
    return root, G


def _explain_json(html: str) -> dict:
    m = re.search(r"const EXPLAIN = (.*?);\n", html)
    assert m
    return json.loads(m.group(1))


def test_to_html_embeds_explanations_and_code(tmp_path):
    root, G = _project(tmp_path)
    out = root / "graphify-out" / "graph.html"
    to_html(G, {0: ["load", "val"]}, str(out), community_labels={0: "Profiles"})
    html = out.read_text(encoding="utf-8")
    ex = _explain_json(html)
    load = ex["load"]
    assert load["kind"] == "function"
    assert "It calls validateUser()." in load["summary"]
    assert load["code"]["start"] == 3
    assert any("export async function loadProfile" in ln for ln in load["code"]["lines"])
    assert load["docs"] and load["docs"][0].startswith("Loads the profile.")
    # Source text cannot close the <script> element it is embedded in.
    assert "</script><script>alert(1)" not in html
    assert "const OVERVIEW = " in html and "const REL_PHRASES = " in html


def test_to_html_code_opt_out(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHIFY_HTML_CODE", "0")
    root, G = _project(tmp_path)
    out = root / "graphify-out" / "graph.html"
    to_html(G, {0: ["load", "val"]}, str(out))
    ex = _explain_json(out.read_text(encoding="utf-8"))
    assert "code" not in ex["load"]
    assert ex["load"]["summary"]  # explanations stay; only source is withheld


def test_show_info_renders_explanation(tmp_path):
    from tests.test_export import _run_node_info_harness

    root, G = _project(tmp_path)
    out = root / "graphify-out" / "graph.html"
    to_html(G, {0: ["load", "val"]}, str(out))
    info = _run_node_info_harness(out.read_text(encoding="utf-8"), "load")
    assert '<div class="kind-badge">function</div>' in info
    assert "loadProfile() is a function in src/api.ts" in info
    assert '<div class="section-title">Calls (1)</div>' in info
    assert "validateUser()" in info
    assert 'class="code-snippet"' in info
    assert "&lt;/script&gt;" in info
    assert "<script>alert(1)" not in info


def test_snippet_reader_only_reads_code_nodes(tmp_path):
    root, G = _project(tmp_path)
    G.add_node("doc", label="README", file_type="document", source_file="src/api.ts", source_location="L1")
    reader = SnippetReader(root / "graphify-out" / "graph.json")
    assert reader.snippet(G, "doc", "document") is None
    assert reader.snippet(G, "load", "function") is not None
