"""Girvan–Newman community detection, implemented from scratch.

The Girvan–Newman algorithm finds communities by *removing* edges rather
than grouping nodes. Edges that connect two communities carry many
shortest paths between them (high *edge betweenness*), so removing them
one at a time slowly splits the graph apart:

1. Compute the edge betweenness of every edge.
2. Remove the edge with the highest betweenness.
3. Recompute betweenness and repeat until no edges are left.

Each time the graph splits into more pieces, the pieces form a candidate
partition. The partition with the highest *modularity* (Q) is returned
as the answer.

Everything in this module (breadth-first search, connected components,
Brandes' edge-betweenness algorithm and modularity) is written with plain
Python data structures. NetworkX is used only to read the input graph.
The one speed-up: after an edge is removed, betweenness is recomputed
only for the component that contained it, because shortest paths in every
other component are unchanged.
"""

import time
from collections import deque


def to_adjacency(graph):
    """Convert a graph into a plain adjacency dictionary.

    Parameters
    ----------
    graph : networkx.Graph or dict
        An undirected NetworkX graph, or a dictionary that maps each node
        to an iterable of its neighbors.

    Returns
    -------
    dict of node to set
        Maps every node to the set of its neighbors. Self-loops are
        dropped.
    """
    if hasattr(graph, "adj"):
        items = ((node, graph.adj[node]) for node in graph.nodes())
    else:
        items = graph.items()
    adjacency = {}
    for node, neighbors in items:
        adjacency[node] = {n for n in neighbors if n != node}
    for node, neighbors in list(adjacency.items()):
        for n in neighbors:
            adjacency.setdefault(n, set()).add(node)
    return adjacency


def edge_key(u, v):
    """Return one fixed key for the undirected edge between ``u`` and ``v``.

    Parameters
    ----------
    u, v : hashable
        The two endpoints.

    Returns
    -------
    tuple
        ``(u, v)`` or ``(v, u)``, always in the same order, so the edge has
        one key no matter which way round it is given.
    """
    return (u, v) if repr(u) <= repr(v) else (v, u)


def count_edges(adjacency):
    """Count the undirected edges in an adjacency dictionary.

    Parameters
    ----------
    adjacency : dict of node to set
        Adjacency dictionary, as built by :func:`to_adjacency`.

    Returns
    -------
    int
        The number of edges.
    """
    return sum(len(neighbors) for neighbors in adjacency.values()) // 2


def bfs_component(adjacency, start):
    """Find every node reachable from ``start`` with breadth-first search.

    Parameters
    ----------
    adjacency : dict of node to set
        Adjacency dictionary.
    start : hashable
        Node to start the search from.

    Returns
    -------
    set
        All nodes in the same connected component as ``start``.
    """
    seen = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen


def connected_components(adjacency):
    """Split a graph into its connected components.

    Parameters
    ----------
    adjacency : dict of node to set
        Adjacency dictionary.

    Returns
    -------
    list of set
        One set of nodes per component. Components are sorted largest
        first, so the result is the same on every run.
    """
    seen = set()
    components = []
    for node in adjacency:
        if node not in seen:
            component = bfs_component(adjacency, node)
            seen |= component
            components.append(component)
    components.sort(key=lambda c: (-len(c), min(repr(n) for n in c)))
    return components


def edge_betweenness(adjacency, nodes=None):
    """Compute edge betweenness with Brandes' algorithm.

    The edge betweenness of an edge is the number of shortest paths,
    between all pairs of nodes, that pass through it. When there are
    several equally short paths between two nodes, each one counts as a
    fraction. For an unweighted graph, Brandes' algorithm runs one
    breadth-first search per source node, in O(nm) time overall.

    Parameters
    ----------
    adjacency : dict of node to set
        Adjacency dictionary.
    nodes : iterable, optional
        Only use these nodes as sources. Pass one connected component to
        score just that component's edges. Defaults to every node.

    Returns
    -------
    dict of tuple to float
        Maps each edge key (see :func:`edge_key`) to its betweenness. Each
        unordered pair of nodes is counted once.
    """
    sources = list(adjacency) if nodes is None else list(nodes)
    scores = {}
    for source in sources:
        # Phase 1: BFS from the source, counting shortest paths (sigma)
        # and recording each node's predecessors on those paths.
        order = []
        predecessors = {source: []}
        sigma = {source: 1.0}
        distance = {source: 0}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in distance:
                    distance[neighbor] = distance[node] + 1
                    sigma[neighbor] = 0.0
                    predecessors[neighbor] = []
                    queue.append(neighbor)
                if distance[neighbor] == distance[node] + 1:
                    sigma[neighbor] += sigma[node]
                    predecessors[neighbor].append(node)

        # Phase 2: walk back from the farthest nodes, passing each node's
        # share of the path count to the edges it came through.
        delta = dict.fromkeys(order, 0.0)
        for node in reversed(order):
            for pred in predecessors[node]:
                share = sigma[pred] / sigma[node] * (1.0 + delta[node])
                key = edge_key(pred, node)
                scores[key] = scores.get(key, 0.0) + share
                delta[pred] += share

    # Every pair (s, t) was counted twice: once from s and once from t.
    for key in scores:
        scores[key] /= 2.0
    return scores


def modularity(adjacency, communities):
    """Compute the modularity (Q) of a partition, from scratch.

    Modularity compares the number of edges inside each community with
    the number expected if the edges were placed at random::

        Q = sum over communities c of [ L_c / m  -  (d_c / 2m)^2 ]

    where ``m`` is the total number of edges, ``L_c`` is the number of
    edges inside community ``c``, and ``d_c`` is the sum of the degrees of
    its nodes. Q is close to 0 for a random split. Values above about 0.3
    mean the graph has clear community structure.

    Parameters
    ----------
    adjacency : dict of node to set
        Adjacency dictionary of the *original* graph (before any edges
        were removed).
    communities : iterable of set
        The partition: every node in exactly one set.

    Returns
    -------
    float
        The modularity, between -0.5 and 1.
    """
    m = count_edges(adjacency)
    if m == 0:
        return 0.0
    label = {}
    for index, community in enumerate(communities):
        for node in community:
            label[node] = index
    inside = {}
    degree_sum = {}
    for node, neighbors in adjacency.items():
        c = label[node]
        degree_sum[c] = degree_sum.get(c, 0) + len(neighbors)
        for neighbor in neighbors:
            if label[neighbor] == c:
                inside[c] = inside.get(c, 0) + 1
    q = 0.0
    for c, degrees in degree_sum.items():
        q += inside.get(c, 0) / 2 / m - (degrees / (2 * m)) ** 2
    return q


class GirvanNewmanResult:
    """The result of one Girvan–Newman run.

    Parameters
    ----------
    levels : list of dict
        One entry each time the graph split into more pieces. Each entry
        has ``"edges_removed"``, ``"communities"`` (a list of sets) and
        ``"modularity"``.
    removed_edges : list of tuple
        Every removed edge, in order, as ``(u, v, betweenness)``.
    modularity_history : list of float
        Modularity after each edge removal. Index 0 is the original graph.
    runtime : float
        Wall-clock time of the run, in seconds.

    Attributes
    ----------
    levels : list of dict
        See the parameter above.
    removed_edges : list of tuple
        See the parameter above.
    modularity_history : list of float
        See the parameter above.
    runtime : float
        See the parameter above.
    best : dict
        The level with the highest modularity.
    """

    def __init__(self, levels, removed_edges, modularity_history, runtime):
        self.levels = levels
        self.removed_edges = removed_edges
        self.modularity_history = modularity_history
        self.runtime = runtime
        self.best = max(levels, key=lambda level: level["modularity"])

    @property
    def communities(self):
        """list of set: the communities of the best (highest-Q) partition."""
        return self.best["communities"]

    @property
    def best_modularity(self):
        """float: the highest modularity found."""
        return self.best["modularity"]

    def partition_with(self, k):
        """Return the first partition that has at least ``k`` communities.

        Parameters
        ----------
        k : int
            The number of communities wanted.

        Returns
        -------
        list of set or None
            The communities, or ``None`` if the graph never split that far.
        """
        for level in self.levels:
            if len(level["communities"]) >= k:
                return level["communities"]
        return None

    def bridge_edges(self, count=3):
        """Return the first edges whose removal split the graph.

        These are the "bridges" between communities: the removed edges
        that each increased the number of components.

        Parameters
        ----------
        count : int, optional
            How many bridges to return. Defaults to 3.

        Returns
        -------
        list of tuple
            Up to ``count`` edges as ``(u, v)``.
        """
        bridges = []
        for level in self.levels[1:]:
            u, v, _ = self.removed_edges[level["edges_removed"] - 1]
            bridges.append((u, v))
            if len(bridges) == count:
                break
        return bridges


def girvan_newman(graph, progress=False):
    """Run the Girvan–Newman algorithm until every edge has been removed.

    Parameters
    ----------
    graph : networkx.Graph or dict
        Undirected, unweighted graph, as a NetworkX graph or an adjacency
        dictionary.
    progress : bool, optional
        If ``True``, print a line each time the graph splits. Defaults to
        ``False``.

    Returns
    -------
    GirvanNewmanResult
        Every partition found, the order edges were removed, the
        modularity after each removal, and the runtime. The answer is
        ``result.communities``, the partition with the highest modularity.

    Notes
    -----
    Ties between edges with the same betweenness are broken by edge key,
    so the result is the same on every run.
    """
    start = time.perf_counter()
    original = to_adjacency(graph)
    working = {node: set(neighbors) for node, neighbors in original.items()}

    components = connected_components(working)
    scores = {}
    for component in components:
        scores.update(edge_betweenness(working, component))

    q = modularity(original, components)
    levels = [{"edges_removed": 0, "communities": components, "modularity": q}]
    history = [q]
    removed = []

    while scores:
        # Highest betweenness wins; round so float noise cannot decide ties.
        u, v = max(scores, key=lambda e: (round(scores[e], 9), e))
        removed.append((u, v, scores[(u, v)]))
        working[u].discard(v)
        working[v].discard(u)

        # Only the component that held (u, v) has changed. Drop its old
        # scores and recompute them for the one or two new pieces.
        piece_u = bfs_component(working, u)
        pieces = [piece_u]
        if v not in piece_u:
            pieces.append(bfs_component(working, v))
        changed = set().union(*pieces)
        for key in [k for k in scores if k[0] in changed]:
            del scores[key]
        for piece in pieces:
            scores.update(edge_betweenness(working, piece))

        if len(pieces) == 2:
            components = connected_components(working)
            q = modularity(original, components)
            levels.append({
                "edges_removed": len(removed),
                "communities": components,
                "modularity": q,
            })
            if progress:
                print(f"  {len(removed):5d} edges removed -> "
                      f"{len(components):3d} communities, Q = {q:.4f}")
        history.append(q)

    runtime = time.perf_counter() - start
    return GirvanNewmanResult(levels, removed, history, runtime)
