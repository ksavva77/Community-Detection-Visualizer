"""Tests that check the from-scratch Girvan–Newman code against NetworkX."""

import os
import sys

import networkx as nx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import girvan_newman as gn  # noqa: E402


@pytest.fixture
def karate():
    graph = nx.karate_club_graph()
    for _, _, data in graph.edges(data=True):
        data.pop("weight", None)
    return graph


def test_edge_betweenness_matches_networkx(karate):
    mine = gn.edge_betweenness(gn.to_adjacency(karate))
    ref = nx.edge_betweenness_centrality(karate, normalized=False)
    for (u, v), score in ref.items():
        assert mine[gn.edge_key(u, v)] == pytest.approx(score)


def test_modularity_matches_networkx(karate):
    parts = [set(range(17)), set(range(17, 34))]
    assert gn.modularity(gn.to_adjacency(karate), parts) == pytest.approx(
        nx.community.modularity(karate, parts)
    )


def test_splits_match_networkx_girvan_newman(karate):
    result = gn.girvan_newman(karate)
    reference = nx.community.girvan_newman(karate)
    for level, communities in zip(result.levels[1:5], reference):
        assert sorted(map(sorted, level["communities"])) == sorted(map(sorted, communities))


def test_karate_best_modularity(karate):
    result = gn.girvan_newman(karate)
    assert result.best_modularity == pytest.approx(0.4013, abs=1e-4)
    assert len(result.communities) == 5


def test_two_triangles_split_at_bridge():
    graph = nx.Graph([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)])
    result = gn.girvan_newman(graph)
    assert result.removed_edges[0][:2] == (2, 3)
    assert sorted(map(sorted, result.communities)) == [[0, 1, 2], [3, 4, 5]]


def test_connected_components():
    adjacency = {0: {1}, 1: {0}, 2: set(), 3: {4}, 4: {3}}
    assert gn.connected_components(adjacency) == [{0, 1}, {3, 4}, {2}]
