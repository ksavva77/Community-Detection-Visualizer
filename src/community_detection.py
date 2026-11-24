"""Community Detection Visualizer: dataset loading and raw-graph visualization.

This module is the first milestone of the CSCI 310 final project
*Community Detection and Visualization in Real-World Networks*. It loads
three real-world networks, prints a summary table of their basic
properties, and draws each network before any community detection is
run, so the later results have a baseline to compare against.

The three datasets are:

* **Zachary's Karate Club**, built into NetworkX (34 nodes, 78 edges).
* **Dolphin Social Network**, from the Network Repository (62 nodes,
  159 edges).
* **Facebook ego network**, one person's friend circle taken from the
  Stanford SNAP ``ego-Facebook`` dataset.

The dolphin and Facebook files are downloaded the first time they are
needed and kept in the ``data/`` folder, so later runs work offline.

Run the module from the repository root::

    python src/community_detection.py
"""

import gzip
import os
import urllib.request
import zipfile

import matplotlib

matplotlib.use("Agg")  # draw to files only; no window is needed

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.io import mmread  # noqa: E402

#: Repository root folder (the folder above ``src/``).
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Folder where downloaded datasets are stored.
DATA_DIR = os.path.join(ROOT_DIR, "data")

#: Folder where generated images are saved.
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

#: Download link for the Dolphin Social Network (Network Repository).
DOLPHINS_URL = "https://nrvis.com/download/data/soc/soc-dolphins.zip"

#: Download link for the combined SNAP Facebook graph.
FACEBOOK_URL = "https://snap.stanford.edu/data/facebook_combined.txt.gz"

#: Default ego node for the Facebook subgraph. Its friend circle has
#: 160 nodes and 1,857 edges: big enough to test scaling, but small
#: enough for Girvan-Newman to finish.
DEFAULT_EGO_NODE = 414


def download_dataset(url, filename, data_dir=DATA_DIR):
    """Download a dataset file unless a copy is already saved.

    Parameters
    ----------
    url : str
        Web address of the file to download.
    filename : str
        Name to save the file under inside ``data_dir``.
    data_dir : str, optional
        Folder for downloaded data. Defaults to :data:`DATA_DIR`.

    Returns
    -------
    str
        Full path to the saved file.
    """
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, path)
    return path


def load_karate_club():
    """Load Zachary's Karate Club network.

    Each node has a ``club`` attribute (``"Mr. Hi"`` or ``"Officer"``)
    recording which side the member joined after the club split. This
    real split is the ground truth used later to measure how accurate
    the detected communities are.

    NetworkX stores a ``weight`` on each edge (how many activities two
    members shared). The weights are removed so that every dataset is
    treated as unweighted, the same way Girvan–Newman treats it.

    Returns
    -------
    networkx.Graph
        Unweighted undirected graph with 34 nodes and 78 edges.
    """
    graph = nx.karate_club_graph()
    for _, _, data in graph.edges(data=True):
        data.pop("weight", None)
    return graph


def load_dolphins(data_dir=DATA_DIR):
    """Load the Dolphin Social Network from the Network Repository.

    The file is in Matrix Market (``.mtx``) format inside a zip file. The
    nodes are renumbered from 0 so they match the other datasets.

    Parameters
    ----------
    data_dir : str, optional
        Folder for downloaded data. Defaults to :data:`DATA_DIR`.

    Returns
    -------
    networkx.Graph
        Undirected graph with 62 nodes (dolphins) and 159 edges (frequent
        associations).
    """
    path = download_dataset(DOLPHINS_URL, "soc-dolphins.zip", data_dir)
    with zipfile.ZipFile(path) as archive:
        with archive.open("soc-dolphins.mtx") as mtx_file:
            matrix = mmread(mtx_file)
    graph = nx.from_scipy_sparse_array(matrix)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def load_facebook_full(data_dir=DATA_DIR):
    """Load the whole SNAP Facebook graph (all ten ego networks combined).

    Parameters
    ----------
    data_dir : str, optional
        Folder for downloaded data. Defaults to :data:`DATA_DIR`.

    Returns
    -------
    networkx.Graph
        Undirected graph with 4,039 nodes and 88,234 edges.
    """
    path = download_dataset(FACEBOOK_URL, "facebook_combined.txt.gz", data_dir)
    with gzip.open(path, "rt") as edge_file:
        return nx.read_edgelist(edge_file, nodetype=int)


def load_facebook_ego(ego_node=DEFAULT_EGO_NODE, data_dir=DATA_DIR):
    """Load one person's Facebook friend circle (an ego network).

    The full Facebook graph is too large for Girvan-Newman, which takes
    roughly O(m^2 n) time, so the project works with one ego network: the
    chosen person, all of their friends, and the friendships among them.

    Parameters
    ----------
    ego_node : int, optional
        ID of the person at the center. The SNAP ego nodes are 0, 107,
        348, 414, 686, 698, 1684, 1912, 3437 and 3980. Defaults to
        :data:`DEFAULT_EGO_NODE`.
    data_dir : str, optional
        Folder for downloaded data. Defaults to :data:`DATA_DIR`.

    Returns
    -------
    networkx.Graph
        The ego network, relabeled with nodes numbered from 0.

    Raises
    ------
    ValueError
        If ``ego_node`` is not in the Facebook graph.
    """
    full_graph = load_facebook_full(data_dir)
    if ego_node not in full_graph:
        raise ValueError(f"Node {ego_node} is not in the Facebook graph.")
    ego = nx.ego_graph(full_graph, ego_node)
    return nx.convert_node_labels_to_integers(ego, label_attribute="original_id")


def load_all_datasets(ego_node=DEFAULT_EGO_NODE, data_dir=DATA_DIR):
    """Load all three project datasets.

    Parameters
    ----------
    ego_node : int, optional
        Ego node for the Facebook subgraph. Defaults to
        :data:`DEFAULT_EGO_NODE`.
    data_dir : str, optional
        Folder for downloaded data. Defaults to :data:`DATA_DIR`.

    Returns
    -------
    dict of str to networkx.Graph
        Maps each dataset's display name to its graph, in order from
        smallest to largest.
    """
    return {
        "Karate Club": load_karate_club(),
        "Dolphins": load_dolphins(data_dir),
        f"Facebook (ego {ego_node})": load_facebook_ego(ego_node, data_dir),
    }


def graph_summary(graph, name):
    """Compute basic statistics for one graph.

    Parameters
    ----------
    graph : networkx.Graph
        The graph to describe.
    name : str
        Display name of the dataset.

    Returns
    -------
    dict
        The dataset name, node count, edge count, average degree, density,
        number of connected components and average clustering coefficient.
    """
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    return {
        "Dataset": name,
        "Nodes": nodes,
        "Edges": edges,
        "Avg Degree": round(2 * edges / nodes, 2) if nodes else 0.0,
        "Density": round(nx.density(graph), 4),
        "Components": nx.number_connected_components(graph),
        "Avg Clustering": round(nx.average_clustering(graph), 4),
    }


def summary_table(datasets):
    """Build a summary table for several graphs.

    Parameters
    ----------
    datasets : dict of str to networkx.Graph
        Dataset names mapped to their graphs, as returned by
        :func:`load_all_datasets`.

    Returns
    -------
    pandas.DataFrame
        One row per dataset, with the columns from :func:`graph_summary`.
    """
    rows = [graph_summary(graph, name) for name, graph in datasets.items()]
    return pd.DataFrame(rows)


def draw_raw_graph(graph, name, ax, seed=42):
    """Draw one graph before any communities are detected.

    All nodes are the same color. Node size grows with degree (scaled to
    the graph's highest degree, so dense graphs stay readable), so
    well-connected nodes stand out.

    Parameters
    ----------
    graph : networkx.Graph
        The graph to draw.
    name : str
        Title shown above the drawing.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    seed : int, optional
        Random seed for the spring layout, so the picture is the same on
        every run. Defaults to 42.

    Returns
    -------
    dict
        Node positions from the spring layout, so later plots of the same
        graph can reuse them for side-by-side comparison.
    """
    pos = nx.spring_layout(graph, seed=seed)
    degrees = dict(graph.degree())
    max_degree = max(degrees.values()) or 1
    sizes = [15 + 220 * degrees[n] / max_degree for n in graph.nodes()]
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.25, width=0.6)
    nx.draw_networkx_nodes(
        graph, pos, ax=ax, node_size=sizes, node_color="#7a8794",
        edgecolors="white", linewidths=0.5,
    )
    ax.set_title(
        f"{name}\n{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
    )
    ax.axis("off")
    return pos


def plot_raw_graphs(datasets, path, seed=42):
    """Save a side-by-side image of every raw (unclustered) graph.

    Parameters
    ----------
    datasets : dict of str to networkx.Graph
        Dataset names mapped to their graphs.
    path : str
        File path for the PNG image.
    seed : int, optional
        Random seed for the spring layouts. Defaults to 42.

    Returns
    -------
    str
        The path the image was saved to.
    """
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 6))
    if len(datasets) == 1:
        axes = [axes]
    for ax, (name, graph) in zip(axes, datasets.items()):
        draw_raw_graph(graph, name, ax, seed=seed)
    fig.suptitle("Raw networks before community detection", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_karate_ground_truth(graph, path, seed=42):
    """Save the Karate Club network colored by the real club split.

    This is the known correct answer the detected communities will be
    compared with in the next milestone.

    Parameters
    ----------
    graph : networkx.Graph
        The Karate Club graph from :func:`load_karate_club`.
    path : str
        File path for the PNG image.
    seed : int, optional
        Random seed for the spring layout. Defaults to 42.

    Returns
    -------
    str
        The path the image was saved to.
    """
    colors = {"Mr. Hi": "#2a78b5", "Officer": "#e07b39"}
    pos = nx.spring_layout(graph, seed=seed)
    fig, ax = plt.subplots(figsize=(7, 6))
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.3)
    for club, color in colors.items():
        members = [n for n, d in graph.nodes(data=True) if d["club"] == club]
        nx.draw_networkx_nodes(
            graph, pos, nodelist=members, ax=ax, node_color=color,
            node_size=260, edgecolors="white", label=f"{club} ({len(members)})",
        )
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8, font_color="white")
    ax.set_title("Karate Club: the real split (ground truth)")
    ax.legend(loc="lower left", frameon=False)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    """Load every dataset, print the summary table and save the images.

    Returns
    -------
    pandas.DataFrame
        The summary table that was printed.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    datasets = load_all_datasets()

    table = summary_table(datasets)
    print("\nDataset summary")
    print("=" * 15)
    print(table.to_string(index=False))

    raw_path = plot_raw_graphs(datasets, os.path.join(OUTPUT_DIR, "raw_networks.png"))
    truth_path = plot_karate_ground_truth(
        datasets["Karate Club"], os.path.join(OUTPUT_DIR, "karate_ground_truth.png")
    )
    print(f"\nSaved {raw_path}\nSaved {truth_path}")
    return table


if __name__ == "__main__":
    main()
