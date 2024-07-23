import time
from itertools import combinations
from typing import Dict, Tuple, List, Union, Set
import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from scipy.special import digamma


class MetricsBetweenTwoGraphs:
    def __init__(self, G_target: nx.DiGraph, G_pred: nx.DiGraph, df_target: pd.DataFrame, df_pred: pd.DataFrame):
        self.G_target = G_target
        self.G_pred = G_pred

        self.df_target = df_target
        self.df_pred = df_pred

        self.result = {}

    def compare(self):
        self._compute_metrics_between_graphs()
        self._generate_result()

        return self.result

    def _compute_metrics_between_graphs(self):
        adj_matrix_target = get_adjacency_matrix(self.G_target)
        adj_matrix_pred = get_adjacency_matrix(self.G_pred)

        self.shd = get_StructuralHammingDistance(self.G_target, self.G_pred)
        self.sid = get_StructuralInterventionDistance(self.G_target, self.G_pred)
        self.js_ = 1 - get_JaccardSimilarity(self.G_target, self.G_pred)
        self.frobenius_norm = get_FrobeniusNorm(adj_matrix_target, adj_matrix_pred)
        self.dds = get_DegreeDistributionSimilarity(self.G_target, self.G_pred)
        self.ccs = get_ClusteringCoefficientSimilarity(self.G_target, self.G_pred)

        obs_dist_target, int_dist_target = get_obs_and_int_distribution(self.G_target, self.df_target)
        obs_dist_pred, int_dist_pred = get_obs_and_int_distribution(self.G_pred, self.df_pred)

        self.kl_obs = get_KL_Divergence(obs_dist_target, obs_dist_pred)
        self.tvd_obs = get_TotalVariationDistance(obs_dist_target, obs_dist_pred)
        self.cmi_obs = get_CapacitatedMutualInformation(obs_dist_target, obs_dist_pred)

        self.kl_int = get_KL_Divergence(int_dist_target, int_dist_pred)
        self.tvd_int = get_TotalVariationDistance(int_dist_target, int_dist_pred)
        self.cmi_int = get_CapacitatedMutualInformation(int_dist_target, int_dist_pred)

    def _generate_result(self) -> Dict:
        result = {
            'shd': self.shd,
            'sid': self.sid,
            'js_': self.js_,
            'frobenius_norm': self.frobenius_norm,
            'dds': self.dds,
            'ccs': self.ccs,
            'kl_obs': self.kl_obs,
            'tvd_obs': self.tvd_obs,
            'cmi_obs': self.cmi_obs,
            'kl_int': self.kl_int,
            'tvd_int': self.tvd_int,
            'cmi_int': self.cmi_int,
        }

        return result


class MetricsOnTheGraph:
    def __init__(self, df: pd.DataFrame, causal_graph: nx.DiGraph):
        self.df = df
        self.G = causal_graph
        self.result = None
        self.computation_time_metrics = 0

    def assessment(self):
        self._compute_graph_metrics()
        self._generate_result()

        return self.result

    def _compute_graph_metrics(self):
        start_time = time.time()
        self.n_nodes = self.G.number_of_nodes()
        self.n_edges = self.G.number_of_edges()
        self.observational_distribution, self.interventional_distribution = get_obs_and_int_distribution(
            self.G, self.df)
        self.causal_strengths = get_causal_strengths_all_edges(self.G, self.df)
        self.adjacency_matrix = get_adjacency_matrix(self.G)
        self.mec = get_markov_equivalence_class(self.G)
        self.markov_blanket_all_nodes = {str(node): list(get_markov_blanket_node(self.G, node)) for node in
                                         self.G.nodes}
        self.computation_time_metrics = time.time() - start_time

    def _generate_result(self):
        self.result = {
            'computation_time_metrics': self.computation_time_metrics,
            'n_edges': self.n_edges,
            'n_nodes': self.n_nodes,
            'observational_distribution': {k: v.tolist() for k, v in self.observational_distribution.items()},
            'interventional_distribution': convert_arrays_to_lists(self.interventional_distribution),
            'causal_strengths_all_nodes': self.causal_strengths,
            'adjacency_matrix': list(self.adjacency_matrix),
            'markov_equivalence_class': mec_to_serializable(self.mec),
            'markov_blanket_all_nodes': self.markov_blanket_all_nodes
        }
        print(f' Time to compute graph-metrics: {round(self.computation_time_metrics, 2)} secs')


def convert_arrays_to_lists(d):
    for k, v in d.items():
        if isinstance(v, dict):
            convert_arrays_to_lists(v)
        elif isinstance(v, np.ndarray):
            d[k] = v.tolist()
    return d


def mec_to_serializable(mec: Set[nx.DiGraph]) -> List[List[Tuple[str, str]]]:
    return [list(graph.edges()) for graph in mec]


class CausalStrengthsComputationClass:
    def __init__(self, data: pd.DataFrame, dag_structure: nx.DiGraph):
        self.data = data.copy()
        self.dag = BayesianNetwork(dag_structure)
        self.encoders = self._encode_categorical_data()
        self.is_continuous = all(self.data.dtypes == 'float64')
        self.model = None

    def _encode_categorical_data(self) -> Dict[str, LabelEncoder]:
        encoders = {col: LabelEncoder().fit(self.data[col]) for col in
                    self.data.select_dtypes(include=['object']).columns}
        for col, encoder in encoders.items():
            self.data[col] = encoder.transform(self.data[col])
        return encoders

    def fit_model(self) -> None:
        self.model = self.dag.fit(self.data, estimator=MaximumLikelihoodEstimator)

    def _get_joint_distribution(self) -> np.ndarray:
        return self.data.value_counts(normalize=True).values

    def _get_interventional_distribution_or_covariance(self, removed_edge: Tuple[str, str]) -> Union[
        np.ndarray, np.ndarray]:
        source, target = removed_edge
        interventional_data = self.data.copy()
        interventional_data[target] = np.random.permutation(interventional_data[target].values)
        if self.is_continuous:
            return get_covariance_matrix(interventional_data)
        else:
            return self._get_joint_distribution()

    def compute_causal_strengths(self) -> Dict[str, float]:
        causal_strengths = {}
        for edge in self.dag.edges():
            if self.is_continuous:
                original_cov = get_covariance_matrix(self.data)
                original_mean = self.data.mean().values
                interventional_cov = self._get_interventional_distribution_or_covariance(edge)
                interventional_mean = self.data.mean().values
                cs = get_kl_divergence_gaussian_continuous(original_mean, original_cov, interventional_mean,
                                                           interventional_cov)
            else:
                original_dist = self._get_joint_distribution()
                interventional_dist = self._get_interventional_distribution_or_covariance(edge)
                cs = get_kl_divergence_discrete(original_dist, interventional_dist)
            causal_strengths[str(edge)] = cs
        return causal_strengths


class MarkovEquivalenceClass:
    def __init__(self, dag: nx.DiGraph):
        self.dag = dag

    @staticmethod
    def skeleton(graph: nx.DiGraph) -> nx.Graph:
        return nx.Graph(graph)

    @staticmethod
    def find_v_structures(graph: nx.DiGraph) -> List[Tuple[int, int, int]]:
        v_structures = []
        for node in graph.nodes:
            predecessors = list(graph.predecessors(node))
            if len(predecessors) < 2:
                continue
            for pair in combinations(predecessors, 2):
                if not graph.has_edge(pair[0], pair[1]) and not graph.has_edge(pair[1], pair[0]):
                    v_structures.append((pair[0], node, pair[1]))
        return v_structures

    @staticmethod
    def markov_equivalent(graph1: nx.DiGraph, graph2: nx.DiGraph) -> bool:
        return (MarkovEquivalenceClass.skeleton(graph1).edges == MarkovEquivalenceClass.skeleton(graph2).edges and
                set(MarkovEquivalenceClass.find_v_structures(graph1)) == set(
                    MarkovEquivalenceClass.find_v_structures(graph2)))

    @staticmethod
    def covered_edge_flips(graph: nx.DiGraph) -> List[nx.DiGraph]:
        flips = []
        for edge in graph.edges:
            u, v = edge
            if set(graph.predecessors(u)) == set(graph.predecessors(v)).difference({u}):
                flipped = graph.copy()
                flipped.remove_edge(u, v)
                flipped.add_edge(v, u)
                flips.append(flipped)
        return flips

    def return_equivalence_class(self) -> Set[nx.DiGraph]:
        mec = set()
        to_visit = [self.dag]
        while to_visit:
            current = to_visit.pop()
            if any(self.markov_equivalent(current, g) for g in mec):
                continue
            mec.add(current)
            to_visit.extend(self.covered_edge_flips(current))
        return mec


def get_covariance_matrix(data: pd.DataFrame) -> np.ndarray:
    return np.cov(data, rowvar=False)


def get_kl_divergence_gaussian_continuous(mu1: np.ndarray, cov1: np.ndarray, mu2: np.ndarray,
                                          cov2: np.ndarray) -> float:
    if not _is_positive_definite(cov2):
        epsilon = 1e-10
        cov2 += epsilon * np.eye(cov2.shape[0])
    dim = len(mu1)
    cov2_inv = np.linalg.inv(cov2)
    term1 = np.trace(cov2_inv @ cov1)
    term2 = (mu2 - mu1).T @ cov2_inv @ (mu2 - mu1)
    term3 = np.log(np.linalg.det(cov2) / np.linalg.det(cov1))
    return 0.5 * (term1 + term2 - dim + term3)


def get_kl_divergence_discrete(p: np.ndarray, q: np.ndarray) -> float:
    return np.sum(p * np.log(p / q))


def _is_positive_definite(matrix: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def get_obs_and_int_distribution(G: nx.DiGraph, df: pd.DataFrame):
    model = BayesianNetwork(G.edges())
    model.fit(df, estimator=MaximumLikelihoodEstimator)
    observational_distributions = _compute_observational_distributions(model)
    interventional_distributions = _compute_all_interventional_distributions(model, df)
    return observational_distributions, interventional_distributions


def _compute_observational_distributions(model):
    infer = VariableElimination(model)
    observational_distributions = {var: infer.query([var]).values for var in model.nodes()}
    return observational_distributions


def _compute_all_interventional_distributions(model, df: pd.DataFrame):
    infer = VariableElimination(model)
    interventional_distributions = {}

    for intervention_var in model.nodes():
        interventional_distributions[intervention_var] = {}
        for intervention_value in df[intervention_var].unique():
            intervention = {intervention_var: intervention_value}
            distributions = {var: infer.query([var], intervention).values for var in model.nodes() if
                             var != intervention_var}
            interventional_distributions[intervention_var][intervention_value] = distributions

    return interventional_distributions


def get_causal_strengths_all_edges(G: nx.DiGraph, df: pd.DataFrame):
    causal_model = CausalStrengthsComputationClass(df, G)
    causal_model.fit_model()
    causal_strengths = causal_model.compute_causal_strengths()
    return causal_strengths


def get_adjacency_matrix(graph, order_nodes=None, weight=False) -> np.array:
    """Retrieve the adjacency matrix from the nx.DiGraph or numpy array."""
    if isinstance(graph, np.ndarray):
        return graph
    elif isinstance(graph, nx.DiGraph):
        if order_nodes is None:
            order_nodes = graph.nodes()
        if not weight:
            return np.array(nx.adjacency_matrix(graph, order_nodes, weight=None).todense())
        else:
            return np.array(nx.adjacency_matrix(graph, order_nodes).todense())
    else:
        raise TypeError("Only networkx.DiGraph and np.ndarray (adjacency matrixes) are supported.")


def get_markov_equivalence_class(G: nx.DiGraph):
    mec_computer = MarkovEquivalenceClass(G)
    return mec_computer.return_equivalence_class()


def get_markov_blanket_node(G: nx.DiGraph, node: str) -> set:
    parents = list(G.predecessors(node))
    children = list(G.successors(node))
    co_parents = {pred for child in children for pred in G.predecessors(child)}
    co_parents.discard(node)
    return set(parents + children + list(co_parents))


def get_StructuralHammingDistance(target, pred, double_for_anticausal=True) -> float:
    r"""Compute the Structural Hamming Distance.

    The Structural Hamming Distance (SHD) is a standard distance to compare
    graphs by their adjacency matrix. It consists in computing the difference
    between the two (binary) adjacency matrixes: every edge that is either
    missing or not in the target graph is counted as a mistake. Note that
    for directed graph, two mistakes can be counted as the edge in the wrong
    direction is false and the edge in the good direction is missing ; the
    `double_for_anticausal` argument accounts for this remark. Setting it to
    `False` will count this as a single mistake.

    Args:
        target (numpy.ndarray or networkx.DiGraph): Target graph, must be of
            ones and zeros.
        pred (numpy.ndarray or networkx.DiGraph): Prediction made by the
            algorithm to evaluate.
        double_for_anticausal (bool): Count the badly oriented edges as two
            mistakes. Default: True

    Returns:
        int: Structural Hamming Distance (int).

            The value tends to zero as the graphs tend to be identical."""

    edges1 = set(target.edges())
    edges2 = set(pred.edges())

    # Edge additions and deletions
    additions = edges2 - edges1
    deletions = edges1 - edges2

    # Edge reversals
    reversals = set((v, u) for (u, v) in edges1).intersection(edges2)

    shd_value = len(additions) + len(deletions) + len(reversals)
    print('shd1: ', shd_value)

    true_labels = get_adjacency_matrix(target)
    predictions = get_adjacency_matrix(pred, target.nodes()
    if isinstance(target, nx.DiGraph) else None)

    diff = np.abs(true_labels - predictions)
    if double_for_anticausal:
        return np.sum(diff)
    else:
        diff = diff + diff.transpose()
        diff[diff > 1] = 1  # Ignoring the double edges.
        print('shd2: ', np.sum(diff) / 2)
        return np.sum(diff) / 2


def get_StructuralInterventionDistance(target, pred) -> int:
    def find_intervention_distances(graph):
        distances = {}
        nodes = list(graph.nodes)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                distances[(nodes[i], nodes[j])] = nx.shortest_path_length(graph, nodes[i], nodes[j])
                distances[(nodes[j], nodes[i])] = nx.shortest_path_length(graph, nodes[j], nodes[i])
        return distances

    true_distances = find_intervention_distances(target)
    estimated_distances = find_intervention_distances(pred)

    sid = 0
    for key in true_distances.keys():
        if true_distances[key] != estimated_distances[key]:
            sid += 1

    return sid


def get_JaccardSimilarity(target, pred) -> float:
    def _intersection(a, b):
        return list(set(a) & set(b))

    def _union(a, b):
        return list(set(a) | set(b))

    nodes_target = set(target.nodes)
    nodes_pred = set(pred.nodes)
    all_nodes = nodes_target | nodes_pred
    similarity = 0
    num_pairs = 0

    for node1 in all_nodes:
        neighbors1 = list(target.successors(node1)) if node1 in target else []
        for node2 in all_nodes:
            neighbors2 = list(pred.successors(node2)) if node2 in pred else []

            intersection_size = len(_intersection(neighbors1, neighbors2))
            union_size = len(_union(neighbors1, neighbors2))

            if union_size > 0:
                similarity += intersection_size / union_size
                num_pairs += 1

    return similarity / num_pairs if num_pairs > 0 else 0


def get_FrobeniusNorm(target, pred) -> float:
    """
    Compute the Frobenius norm between two matrices.

    Parameters:
    target (np.ndarray): First matrix.
    pred (np.ndarray): Second matrix.

    Returns:
    float: Frobenius norm between matrix1 and matrix2.
    """
    return np.linalg.norm(target - pred, 'fro')


def get_KL_Divergence(target, pred) -> float:
    """
    Compute the Kullback-Leibler (KL) divergence between two probability distributions.

    Parameters:
    target (np.ndarray): First probability distribution.
    pred (np.ndarray): Second probability distribution.

    Returns:
    float: KL divergence between target and pred.
    """
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    # Add a small value to avoid division by zero
    target += 1e-10
    pred += 1e-10

    return np.sum(target * np.log(target / pred))


def get_TotalVariationDistance(target, pred) -> float:
    """
    Compute the total variation distance between two probability distributions.

    Parameters:
    target (np.ndarray): First probability distribution.
    pred (np.ndarray): Second probability distribution.

    Returns:
    float: Total variation distance between target and pred.
    """
    return 0.5 * np.sum(np.abs(target - pred))


def get_DegreeDistributionSimilarity(target, pred) -> float:
    # Function to calculate Degree Distribution Similarity (Kolmogorov-Smirnov)
    degrees1 = [d for n, d in target.degree()]
    degrees2 = [d for n, d in pred.degree()]
    return ks_2samp(degrees1, degrees2).statistic


def get_ClusteringCoefficientSimilarity(target, pred) -> float:
    # Function to calculate Clustering Coefficient Similarity
    cc1 = nx.average_clustering(target)
    cc2 = nx.average_clustering(pred)
    return abs(cc1 - cc2)


def get_CapacitatedMutualInformation(target, pred, k=5) -> float:
    def _compute_knn_distances(data):
        """
        Compute k-nearest neighbor distances for each point in the dataset.
        """
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(data)
        distances, indices = nbrs.kneighbors(data)
        return distances[:, 1:], indices[:, 1:]

    def _estimate_mutual_information(X, Y):
        """
        Estimate the mutual information I(X; Y) using the k-nearest neighbor approach.
        """
        N, dx = X.shape
        dy = Y.shape[1]

        data = np.concatenate((X, Y), axis=1)
        dXY = _compute_knn_distances(data)[0]
        dX = _compute_knn_distances(X)[0]
        dY = _compute_knn_distances(Y)[0]

        knn_distances = np.max(np.concatenate((dX, dY), axis=1), axis=1)
        nx = np.sum(dX < knn_distances[:, None], axis=1)
        ny = np.sum(dY < knn_distances[:, None], axis=1)

        mi_est = digamma(k) + digamma(N) - np.mean(digamma(nx)) - np.mean(digamma(ny))
        return mi_est

    def importance_sampling_weights(X, kernel_density_estimate):
        """
        Compute importance sampling weights for transforming the distribution of X.
        """
        weights = 1.0 / kernel_density_estimate
        weights /= np.sum(weights)
        return weights

    def kernel_density_estimation(X, bandwidth='scott'):
        """
        Estimate the kernel density of X using a Gaussian kernel.
        """
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(X.T, bw_method=bandwidth)
        return kde.evaluate(X.T)

    """
    Estimate the Capacitated Mutual Information (CMI) between target and pred.
    """
    N = target.shape[0]
    kde = kernel_density_estimation(target)
    weights = importance_sampling_weights(target, kde)

    # Define the objective function for optimization
    def objective(weights):
        weights = np.maximum(weights, 1e-10)
        weights /= np.sum(weights)
        weighted_X = np.sqrt(weights[:, None]) * target
        return -_estimate_mutual_information(weighted_X, pred, k)

    # Optimize the weights to maximize mutual information
    from scipy.optimize import minimize
    initial_weights = np.ones(N) / N
    result = minimize(objective, initial_weights, method='L-BFGS-B', bounds=[(1e-10, 1)] * N)

    optimal_weights = np.maximum(result.x, 1e-10)
    optimal_weights /= np.sum(optimal_weights)
    weighted_X = np.sqrt(optimal_weights[:, None]) * target

    cmi_est = _estimate_mutual_information(weighted_X, pred)
    return cmi_est
