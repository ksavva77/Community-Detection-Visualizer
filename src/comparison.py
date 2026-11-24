"""Compare Girvan–Newman with Louvain and draw the results.

This module runs both algorithms on every dataset and produces the
project's outputs:

* a console table of communities, modularity (Q), runtime and the top
  bridge edges for each dataset;
* color-coded community drawings next to the raw graph;
* modularity curves showing how Q changes as Girvan–Newman removes edges;
* a runtime-scaling chart across several Facebook ego networks;
* an accuracy check on the Karate Club against the club's real split.
"""

import os
import time

import matplotlib

matplotlib.use("Agg")

import community as community_louvain  # noqa: E402  (python-louvain)
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402

import community_detection as cd  # noqa: E402
import girvan_newman as gn  # noqa: E402

#: Facebook ego networks used for the runtime-scaling test, smallest first.
SCALING_EGO_NODES = [3980, 698, 414, 0, 348]

#: Colors for the largest communities; smaller ones reuse them in order.
PALETTE = [
    "#2a78b5", "#e07b39", "#3a9e5b", "#c8463e", "#8a63b8", "#8c5a46",
    "#d672b8", "#7f7f7f", "#b5b532", "#2bb0c2", "#9cc5e6", "#f4b77e",
]

#: Color for communities with a single node.
SINGLETON_COLOR = "#cfd4d9"


def run_louvain(graph, seed=42):
    """Run the Louvain method from the ``python-louvain`` library.

    Parameters
    ----------
    graph : networkx.Graph
        Undirected graph. Edge weights are ignored (every edge counts as 1),
        to match Girvan–Newman.
    seed : int, optional
        Random seed, because Louvain visits nodes in random order.
        Defaults to 42.

    Returns
    -------
    communities : list of set
        The communities found, largest first.
    runtime : float
        Wall-clock time in seconds.
    """
    start = time.perf_counter()
    labels = community_louvain.best_partition(graph, weight="unweighted", random_state=seed)
    runtime = time.perf_counter() - start
    groups = {}
    for node, label in labels.items():
        groups.setdefault(label, set()).add(node)
    communities = sorted(groups.values(), key=lambda c: (-len(c), min(c)))
    return communities, runtime


def majority_accuracy(communities, truth):
    """Score a partition against known group labels.

    Each detected community is labeled with the most common true group
    among its members. Accuracy is the fraction of nodes whose community
    label matches their true group.

    Parameters
    ----------
    communities : list of set
        The detected communities.
    truth : dict
        Maps each node to its true group label.

    Returns
    -------
    accuracy : float
        Fraction of nodes labeled correctly, from 0 to 1.
    wrong : list
        The nodes that were labeled incorrectly.
    """
    wrong = []
    for community in communities:
        counts = {}
        for node in community:
            counts[truth[node]] = counts.get(truth[node], 0) + 1
        majority = max(counts, key=counts.get)
        wrong.extend(n for n in community if truth[n] != majority)
    total = sum(len(c) for c in communities)
    return 1 - len(wrong) / total, sorted(wrong)


def analyze(graph, name, progress=False):
    """Run both algorithms on one graph and collect the results.

    Parameters
    ----------
    graph : networkx.Graph
        The graph to analyze.
    name : str
        Display name of the dataset.
    progress : bool, optional
        Print Girvan–Newman's progress. Defaults to ``False``.

    Returns
    -------
    dict
        Keys ``name``, ``graph``, ``gn`` (a
        :class:`girvan_newman.GirvanNewmanResult`), ``louvain`` (list of
        sets), ``louvain_q`` and ``louvain_time``.
    """
    print(f"Running Girvan–Newman on {name} ...")
    gn_result = gn.girvan_newman(graph, progress=progress)
    louvain, louvain_time = run_louvain(graph)
    adjacency = gn.to_adjacency(graph)
    return {
        "name": name,
        "graph": graph,
        "gn": gn_result,
        "louvain": louvain,
        "louvain_q": gn.modularity(adjacency, louvain),
        "louvain_time": louvain_time,
    }


def results_table(results):
    """Build the comparison table printed to the console.

    Parameters
    ----------
    results : list of dict
        Results from :func:`analyze`.

    Returns
    -------
    pandas.DataFrame
        One row per dataset with node and edge counts, communities,
        modularity and runtime for each algorithm, and the first three
        bridge edges Girvan–Newman removed.
    """
    rows = []
    for r in results:
        g = r["graph"]
        rows.append({
            "Dataset": r["name"],
            "Nodes": g.number_of_nodes(),
            "Edges": g.number_of_edges(),
            "GN k": len(r["gn"].communities),
            "GN Q": round(r["gn"].best_modularity, 4),
            "GN time (s)": round(r["gn"].runtime, 3),
            "Louvain k": len(r["louvain"]),
            "Louvain Q": round(r["louvain_q"], 4),
            "Louvain time (s)": round(r["louvain_time"], 4),
            "Top bridges": ", ".join(f"{u}-{v}" for u, v in r["gn"].bridge_edges(3)),
        })
    return pd.DataFrame(rows)


def community_colors(graph, communities):
    """Give every node the color of its community.

    Parameters
    ----------
    graph : networkx.Graph
        The graph being drawn.
    communities : list of set
        The communities, largest first.

    Returns
    -------
    list of str
        One color per node, in ``graph.nodes()`` order.
    """
    color = {}
    index = 0
    for community in communities:
        if len(community) == 1:
            shade = SINGLETON_COLOR
        else:
            shade = PALETTE[index % len(PALETTE)]
            index += 1
        for node in community:
            color[node] = shade
    return [color[n] for n in graph.nodes()]


def draw_communities(graph, communities, pos, ax, title):
    """Draw a graph with nodes colored by community.

    Edges inside a community are drawn darker than edges between
    communities, so the structure is easy to see.

    Parameters
    ----------
    graph : networkx.Graph
        The graph to draw.
    communities : list of set
        The communities to color.
    pos : dict
        Node positions, shared with the other panels of the figure.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    title : str
        Panel title.
    """
    label = {n: i for i, c in enumerate(communities) for n in c}
    inside = [(u, v) for u, v in graph.edges() if label[u] == label[v]]
    between = [(u, v) for u, v in graph.edges() if label[u] != label[v]]
    nx.draw_networkx_edges(graph, pos, edgelist=between, ax=ax, alpha=0.12, width=0.5)
    nx.draw_networkx_edges(graph, pos, edgelist=inside, ax=ax, alpha=0.35, width=0.7)
    size = 160 if graph.number_of_nodes() < 100 else 45
    nx.draw_networkx_nodes(
        graph, pos, ax=ax, node_color=community_colors(graph, communities),
        node_size=size, edgecolors="white", linewidths=0.5,
    )
    ax.set_title(title)
    ax.axis("off")


def plot_community_comparison(result, path, seed=42):
    """Save the raw graph, Girvan–Newman and Louvain results side by side.

    All three panels use the same node positions.

    Parameters
    ----------
    result : dict
        One result from :func:`analyze`.
    path : str
        File path for the PNG image.
    seed : int, optional
        Random seed for the spring layout. Defaults to 42.

    Returns
    -------
    str
        The path the image was saved to.
    """
    graph = result["graph"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    pos = cd.draw_raw_graph(graph, f"{result['name']}: raw", axes[0], seed=seed)
    gn_res = result["gn"]
    draw_communities(
        graph, gn_res.communities, pos, axes[1],
        f"Girvan–Newman: {len(gn_res.communities)} communities, "
        f"Q = {gn_res.best_modularity:.3f}",
    )
    draw_communities(
        graph, result["louvain"], pos, axes[2],
        f"Louvain: {len(result['louvain'])} communities, Q = {result['louvain_q']:.3f}",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_modularity_curves(results, path):
    """Plot modularity against the number of edges Girvan–Newman removed.

    The peak of each curve (the partition Girvan–Newman returns) is marked,
    and Louvain's modularity is drawn as a dashed line for comparison.

    Parameters
    ----------
    results : list of dict
        Results from :func:`analyze`.
    path : str
        File path for the PNG image.

    Returns
    -------
    str
        The path the image was saved to.
    """
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4.5))
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        history = r["gn"].modularity_history
        best = r["gn"].best
        ax.plot(range(len(history)), history, color="#2a78b5", lw=1.8,
                label="Girvan–Newman")
        ax.axhline(r["louvain_q"], color="#e07b39", ls="--", lw=1.5,
                   label=f"Louvain (Q = {r['louvain_q']:.3f})")
        ax.scatter([best["edges_removed"]], [best["modularity"]], color="#2a78b5",
                   zorder=3, s=40)
        ax.annotate(
            f"peak Q = {best['modularity']:.3f}\n{len(best['communities'])} communities",
            (best["edges_removed"], best["modularity"]), textcoords="offset points",
            xytext=(10, -28), fontsize=9,
        )
        ax.set_title(r["name"])
        ax.set_xlabel("Edges removed")
        ax.set_ylabel("Modularity (Q)")
        ax.set_ylim(bottom=min(0, min(history)) - 0.02)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_runtime_scaling(scaling, path):
    """Plot runtime against graph size for both algorithms.

    Parameters
    ----------
    scaling : pandas.DataFrame
        Columns ``Dataset``, ``Edges``, ``GN time (s)`` and
        ``Louvain time (s)``, as built by :func:`run_scaling_test`.
    path : str
        File path for the PNG image.

    Returns
    -------
    str
        The path the image was saved to.
    """
    data = scaling.sort_values("Edges")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data["Edges"], data["GN time (s)"], "o-", color="#2a78b5",
            label="Girvan–Newman (from scratch)")
    ax.plot(data["Edges"], data["Louvain time (s)"], "s-", color="#e07b39",
            label="Louvain (python-louvain)")
    for _, row in data.iterrows():
        ax.annotate(row["Dataset"], (row["Edges"], row["GN time (s)"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Edges (log scale)")
    ax.set_ylabel("Runtime in seconds (log scale)")
    ax.set_title("Runtime as graphs grow")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_karate_accuracy(graph, communities, wrong, path, seed=42):
    """Draw a two-community Karate split, circling misclassified members.

    Parameters
    ----------
    graph : networkx.Graph
        The Karate Club graph.
    communities : list of set
        The two detected communities.
    wrong : list
        Nodes placed with the wrong club.
    path : str
        File path for the PNG image.
    seed : int, optional
        Random seed for the spring layout. Defaults to 42.

    Returns
    -------
    str
        The path the image was saved to.
    """
    pos = nx.spring_layout(graph, seed=seed)
    fig, ax = plt.subplots(figsize=(7, 6))
    draw_communities(graph, communities, pos, ax, "")
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=7, font_color="white")
    if wrong:
        nx.draw_networkx_nodes(graph, pos, nodelist=wrong, ax=ax, node_size=420,
                               node_color="none", edgecolors="#c8463e", linewidths=2)
    accuracy = 1 - len(wrong) / graph.number_of_nodes()
    ax.set_title(f"Girvan–Newman first split vs. real clubs: {accuracy:.0%} correct\n"
                 f"(red circle = placed in the wrong club)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_scaling_test(ego_nodes=SCALING_EGO_NODES, extra_results=()):
    """Time both algorithms on Facebook ego networks of different sizes.

    Parameters
    ----------
    ego_nodes : list of int, optional
        Ego nodes to test. Defaults to :data:`SCALING_EGO_NODES`.
    extra_results : iterable of dict, optional
        Results already computed with :func:`analyze` to reuse instead
        of running again (and to add the Karate and Dolphin points).

    Returns
    -------
    pandas.DataFrame
        One row per graph with ``Dataset``, ``Nodes``, ``Edges``,
        ``GN time (s)`` and ``Louvain time (s)``.
    """
    known = {r["name"]: r for r in extra_results}
    rows = []
    for r in extra_results:
        rows.append(_timing_row(r))
    for ego in ego_nodes:
        name = f"Facebook (ego {ego})"
        if name in known:
            continue
        rows.append(_timing_row(analyze(cd.load_facebook_ego(ego), name)))
    return pd.DataFrame(rows).sort_values("Edges").reset_index(drop=True)


def _timing_row(result):
    """Return one row of the scaling table for an :func:`analyze` result."""
    g = result["graph"]
    return {
        "Dataset": result["name"],
        "Nodes": g.number_of_nodes(),
        "Edges": g.number_of_edges(),
        "GN time (s)": round(result["gn"].runtime, 3),
        "Louvain time (s)": round(result["louvain_time"], 4),
    }


def slug(name):
    """Turn a dataset name into a file-name-friendly string.

    Parameters
    ----------
    name : str
        Dataset name, such as ``"Facebook (ego 414)"``.

    Returns
    -------
    str
        Lower-case name with only letters, digits and underscores, such as
        ``"facebook_ego_414"``.
    """
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in name)
    return "_".join(part for part in cleaned.split("_") if part)


def main():
    """Run the full comparison and save every table and image.

    Returns
    -------
    dict
        ``"results"`` (table from :func:`results_table`), ``"scaling"``
        (table from :func:`run_scaling_test`) and ``"karate_accuracy"``.
    """
    out = cd.OUTPUT_DIR
    os.makedirs(out, exist_ok=True)
    datasets = cd.load_all_datasets()
    results = [analyze(graph, name) for name, graph in datasets.items()]

    table = results_table(results)
    print("\nCommunity detection results")
    print("=" * 27)
    print(table.to_string(index=False))

    karate = results[0]
    truth = nx.get_node_attributes(karate["graph"], "club")
    split = karate["gn"].partition_with(2)
    acc_split, wrong_split = majority_accuracy(split, truth)
    acc_gn, _ = majority_accuracy(karate["gn"].communities, truth)
    acc_lv, _ = majority_accuracy(karate["louvain"], truth)
    print("\nKarate Club accuracy against the real club split")
    print(f"  Girvan–Newman, first split (2 communities): {acc_split:.1%}"
          f"  misplaced: {wrong_split}")
    print(f"  Girvan–Newman, best Q ({len(karate['gn'].communities)} communities): "
          f"{acc_gn:.1%}")
    print(f"  Louvain ({len(karate['louvain'])} communities): {acc_lv:.1%}")

    for r in results:
        plot_community_comparison(r, os.path.join(out, f"communities_{slug(r['name'])}.png"))
    plot_modularity_curves(results, os.path.join(out, "modularity_curves.png"))
    plot_karate_accuracy(karate["graph"], split, wrong_split,
                         os.path.join(out, "karate_accuracy.png"))

    print("\nRunning the scaling test on more Facebook ego networks ...")
    scaling = run_scaling_test(extra_results=results)
    print(scaling.to_string(index=False))
    plot_runtime_scaling(scaling, os.path.join(out, "runtime_scaling.png"))
    print(f"\nImages saved in {out}")
    return {
        "results": table,
        "scaling": scaling,
        "karate_accuracy": {"gn_split": acc_split, "gn_best": acc_gn, "louvain": acc_lv},
    }


if __name__ == "__main__":
    main()
