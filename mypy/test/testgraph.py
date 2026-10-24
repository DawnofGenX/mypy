"""Test cases for graph processing code in build.py."""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Set as AbstractSet

from mypy.build import BuildManager, BuildSourceSet, State, order_ascc, sorted_components
from mypy.errors import Errors
from mypy.fscache import FileSystemCache
from mypy.graph_utils import strongly_connected_components, topsort
from mypy.modulefinder import SearchPaths
from mypy.options import Options
from mypy.plugin import Plugin
from mypy.report import Reports
from mypy.test.helpers import Suite, assert_equal
from mypy.version import __version__


class GraphSuite(Suite):
    def test_topsort_empty(self) -> None:
        data: dict[AbstractSet[str], set[AbstractSet[str]]] = {}
        assert_equal(list(topsort(data)), [])

    def test_topsort(self) -> None:
        a = frozenset({"A"})
        b = frozenset({"B"})
        c = frozenset({"C"})
        d = frozenset({"D"})
        data: dict[AbstractSet[str], set[AbstractSet[str]]] = {a: {b, c}, b: {d}, c: {d}}
        res = list(topsort(data))
        assert_equal(res, [{d}, {b, c}, {a}])

    def test_topsort_orphan(self) -> None:
        a = frozenset({"A"})
        b = frozenset({"B"})
        data: dict[AbstractSet[str], set[AbstractSet[str]]] = {a: {b}}
        res = list(topsort(data))
        assert_equal(res, [{b}, {a}])

    def test_topsort_independent(self) -> None:
        a = frozenset({"A"})
        b = frozenset({"B"})
        c = frozenset({"C"})
        data: dict[AbstractSet[str], set[AbstractSet[str]]] = {a: set(), b: set(), c: set()}
        res = list(topsort(data))
        assert_equal(res, [{a, b, c}])

    def test_topsort_linear_chain(self) -> None:
        a = frozenset({"A"})
        b = frozenset({"B"})
        c = frozenset({"C"})
        d = frozenset({"D"})
        data: dict[AbstractSet[str], set[AbstractSet[str]]] = {a: {b}, b: {c}, c: {d}, d: set()}
        res = list(topsort(data))
        assert_equal(res, [{d}, {c}, {b}, {a}])

    def test_topsort_self_dependency(self) -> None:
        a = frozenset({"A"})
        b = frozenset({"B"})
        data: dict[AbstractSet[str], set[AbstractSet[str]]] = {a: {a, b}, b: set()}
        res = list(topsort(data))
        assert_equal(res, [{b}, {a}])

    def test_topsort_orphan_diamond(self) -> None:
        a = frozenset({"A"})
        b = frozenset({"B"})
        c = frozenset({"C"})
        # B and C are orphans -- they appear only in values, not as keys.
        data: dict[AbstractSet[str], set[AbstractSet[str]]] = {a: {b, c}}
        res = list(topsort(data))
        assert_equal(res, [{b, c}, {a}])

    def test_topsort_cycle(self) -> None:
        a = frozenset({"A"})
        b = frozenset({"B"})
        data: dict[AbstractSet[str], set[AbstractSet[str]]] = {a: {b}, b: {a}}
        with self.assertRaises(AssertionError):
            list(topsort(data))

    def test_scc(self) -> None:
        vertices = {"A", "B", "C", "D"}
        edges: dict[str, list[str]] = {"A": ["B", "C"], "B": ["C"], "C": ["B", "D"], "D": []}
        sccs = {frozenset(x) for x in strongly_connected_components(vertices, edges)}
        assert_equal(sccs, {frozenset({"A"}), frozenset({"B", "C"}), frozenset({"D"})})

    def _make_manager(
        self, parallel_worker: bool = False, cache_dir: str | None = None
    ) -> BuildManager:
        options = Options()
        options.use_builtins_fixtures = True
        if cache_dir is not None:
            options.cache_dir = cache_dir
        errors = Errors(options)
        fscache = FileSystemCache()
        search_paths = SearchPaths((), (), (), ())
        manager = BuildManager(
            data_dir="",
            search_paths=search_paths,
            ignore_prefix="",
            source_set=BuildSourceSet([]),
            reports=Reports("", {}),
            options=options,
            version_id=__version__,
            plugin=Plugin(options),
            plugins_snapshot={},
            errors=errors,
            flush_errors=lambda filename, msgs, serious: None,
            fscache=fscache,
            stdout=sys.stdout,
            stderr=sys.stderr,
            parallel_worker=parallel_worker,
        )
        return manager

    def test_sorted_components(self) -> None:
        manager = self._make_manager()
        graph = {
            "a": State.new_state("a", None, "import b, c", manager),
            "d": State.new_state("d", None, "pass", manager),
            "b": State.new_state("b", None, "import c", manager),
            "c": State.new_state("c", None, "import b, d", manager),
            "builtins": State.new_state("builtins", None, "", manager),
        }
        manager.parse_all(list(graph.values()))
        res = [scc.mod_ids for scc in sorted_components(graph)]
        assert_equal(res, [{"builtins"}, {"d"}, {"c", "b"}, {"a"}])

    def test_order_ascc(self) -> None:
        manager = self._make_manager()
        graph = {
            "a": State.new_state("a", None, "import b, c", manager),
            "d": State.new_state("d", None, "def f(): import a", manager),
            "b": State.new_state("b", None, "import c", manager),
            "c": State.new_state("c", None, "import b, d", manager),
            "builtins": State.new_state("builtins", None, "", manager),
        }
        manager.parse_all(list(graph.values()))
        res = [scc.mod_ids for scc in sorted_components(graph)]
        assert_equal(res, [{"builtins"}, {"a", "d", "c", "b"}])
        ascc = res[1]
        scc = order_ascc(graph, ascc)
        assert_equal(scc, ["d", "c", "b", "a"])

    def test_reload_meta_missing_cache_is_noop(self) -> None:
        # Regression test for #22241: a parallel worker must not assert when the
        # cache meta is missing, e.g. because a cache write was dropped while the
        # sqlite cache was locked by a concurrent mypy run. It should keep the
        # interface hash it already has instead of crashing the worker.
        with tempfile.TemporaryDirectory() as tmp:
            manager = self._make_manager(parallel_worker=True, cache_dir=tmp)
            state = State.new_state("a", os.path.join(tmp, "a.py"), "pass", manager)
            state.interface_hash = b"existing"
            state.reload_meta()  # there is no cache meta for "a": must not raise
            assert_equal(state.interface_hash, b"existing")

    def test_load_tree_parses_when_cache_data_is_missing(self) -> None:
        # Regression test for #22241: the same dropped write can leave the tree
        # cache file missing in a parallel worker. Leaving state.tree as None
        # crashes later in fix_cross_refs(), so load_tree() must treat the missing
        # file as a cache miss and parse the source.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.py")
            with open(path, "w") as f:
                f.write("x: int = 1\n")
            manager = self._make_manager(
                parallel_worker=True, cache_dir=os.path.join(tmp, "cache")
            )
            state = State.new_state("a", path, None, manager)
            state.load_tree()
            assert state.tree is not None, "load_tree() must not leave the tree unset"
            assert_equal(state.tree.fullname, "a")
            assert_equal(len(state.tree.defs), 1)
