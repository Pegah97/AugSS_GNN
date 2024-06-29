from embeddings.data.dataset_loader import DatasetLoader
import logging
import pandas as pd
import itertools
from embeddings.core.graph import MyGraph

log = logging.getLogger(__name__)


def _load_cora_or_citeseer(
    dataset,
    directed,
    largest_connected_component_only,
    subject_as_feature,
    edge_weights,
    nodes_dtype,
):

    assert isinstance(dataset, (Cora, CiteSeer))

    if nodes_dtype is None:
        nodes_dtype = dataset._NODES_DTYPE

    dataset.download()

    # expected_files should be in this order
    cites, content = [dataset._resolve_path(name) for name in dataset.expected_files]

    feature_names = ["w_{}".format(ii) for ii in range(dataset._NUM_FEATURES)]
    subject = "subject"
    if subject_as_feature:
        feature_names.append(subject)
        column_names = feature_names
    else:
        column_names = feature_names + [subject]

    node_data = pd.read_csv(
        content, sep="\t", header=None, names=column_names, dtype={0: nodes_dtype}
    )

    edgelist = pd.read_csv(
        cites, sep="\t", header=None, names=["target", "source"], dtype=nodes_dtype
    )

    valid_source = node_data.index.get_indexer(edgelist.source) >= 0
    valid_target = node_data.index.get_indexer(edgelist.target) >= 0
    edgelist = edgelist[valid_source & valid_target]

    subjects = node_data[subject]

    cls = MyGraph


    features = node_data[feature_names]

    if subject_as_feature:
        # one-hot encode the subjects
        features = pd.get_dummies(features, columns=[subject])

    #graph = cls({"paper": features}, {"cites": edgelist}, subjects)
    graph = cls({"paper": features}, {"cites": edgelist}, subjects)


    if edge_weights is not None:
        edgelist["weight"] = edge_weights(graph, subjects, edgelist)
        graph = cls({"paper": node_data[feature_names]}, {"cites": edgelist}, subjects)

    """
    if largest_connected_component_only:
        cc_ids = next(graph.connected_components())
        return graph.subgraph(cc_ids), subjects[cc_ids]
    """

    return graph


class Cora(
    DatasetLoader,
    name="Cora",
    directory_name="cora",
    url="https://linqs-data.soe.ucsc.edu/public/datasets/cora/cora.tar.gz",
    url_archive_format="gztar",
    expected_files=["cora.cites", "cora.content"],
    description="The Cora dataset consists of 2708 scientific publications classified into one of seven classes. "
    "The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued word vector "
    "indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words.",
    source="https://linqs.soe.ucsc.edu/data",
):

    _NUM_FEATURES = 1433

    def load(
        self,
        directed=False,
        largest_connected_component_only=False,
        subject_as_feature=False,
        edge_weights=None,
        str_node_ids=False,
    ):
        nodes_dtype = str if str_node_ids else int

        return _load_cora_or_citeseer(
            self,
            directed,
            largest_connected_component_only,
            subject_as_feature,
            edge_weights,
            nodes_dtype,
        )


class CiteSeer(
    DatasetLoader,
    name="CiteSeer",
    directory_name="citeseer-doc-classification",
    url="https://linqs-data.soe.ucsc.edu/public/datasets/citeseer-doc-classification/citeseer-doc-classification.tar.gz",
    url_archive_format="gztar",
    expected_files=["citeseer.cites", "citeseer.content"],
    description="The CiteSeer dataset consists of 3312 scientific publications classified into one of six classes. "
    "The citation network consists of 4732 links, although 17 of these have a source or target publication that isn't in the dataset and only 4715 are included in the graph. "
    "Each publication in the dataset is described by a 0/1-valued word vector "
    "indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 3703 unique words.",
    source="https://linqs.soe.ucsc.edu/data",
):
    _NUM_FEATURES = 3703

    def load(self, largest_connected_component_only=False):
        # some node IDs are integers like 100157 and some are strings like
        # bradshaw97introduction. Pandas can get confused, so it's best to explicitly force them all
        # to be treated as strings.
        nodes_dtype = str

        return _load_cora_or_citeseer(
            self, False, largest_connected_component_only, False, None, nodes_dtype
        )


class PubMedDiabetes(
    DatasetLoader,
    name="PubMed Diabetes",
    directory_name="pubmed-diabetes",
    url="https://linqs-data.soe.ucsc.edu/public/datasets/pubmed-diabetes/pubmed-diabetes.tar.gz",
    url_archive_format="gztar",
    expected_files=[
        "data/Pubmed-Diabetes.DIRECTED.cites.tab",
        "data/Pubmed-Diabetes.GRAPH.pubmed.tab",
        "data/Pubmed-Diabetes.NODE.paper.tab",
    ],
    description="The PubMed Diabetes dataset consists of 19717 scientific publications from PubMed database "
                "pertaining to diabetes classified into one of three classes. The citation network consists of 44338 links. "
                "Each publication in the dataset is described by a TF/IDF weighted word vector from a dictionary which consists of 500 unique words.",
    source="https://linqs.soe.ucsc.edu/data",
    data_subdirectory_name="data",
):
    def load(self):
        """
        Load this graph into an undirected homogeneous graph, downloading it if required.

        Returns:
            A tuple where the first element is a :class:`.StellarGraph` instance containing the graph
            data and features, and the second element is a pandas Series of node class labels.
        """
        self.download()

        directed, _graph, node = [self._resolve_path(f) for f in self.expected_files]
        edgelist = pd.read_csv(
            directed,
            sep="\t",
            skiprows=2,
            header=None,
            names=["id", "source", "pipe", "target"],
            usecols=["source", "target"],
        )
        edgelist.source = edgelist.source.str.lstrip("paper:").astype(int)
        edgelist.target = edgelist.target.str.lstrip("paper:").astype(int)

        def parse_feature(feat):
            name, value = feat.split("=")
            return name, float(value)

        def parse_line(line):
            pid, raw_label, *raw_features, _summary = line.split("\t")
            features = dict(parse_feature(feat) for feat in raw_features)
            features["pid"] = int(pid)
            features["label"] = int(parse_feature(raw_label)[1])
            return features

        with open(node) as fp:
            node_data = pd.DataFrame(
                parse_line(line) for line in itertools.islice(fp, 2, None)
            )

        node_data.fillna(0, inplace=True)
        node_data.set_index("pid", inplace=True)

        labels = node_data["label"]

        nodes = node_data.drop(columns="label")
        graph = MyGraph({"paper": nodes}, {"cites": edgelist}, labels)

        return graph



