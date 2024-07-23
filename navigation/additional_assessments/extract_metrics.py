import time
from itertools import combinations
from typing import Dict, Tuple, List, Union, Set
import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from sklearn.preprocessing import LabelEncoder


class CausalDiscoveryPipeline:
    def __init__(self, df: pd.DataFrame, causal_graph: nx.DiGraph):
        self.df = df
        self.causal_graph = causal_graph
        self.result = None
        self.computation_time_metrics = 0

    def run(self):
        self._compute_graph_metrics()
        self._generate_result()

        return self.result

    def _compute_graph_metrics(self):
        start_time = time.time()
        self.n_nodes = self.causal_graph.number_of_nodes()
        self.n_edges = self.causal_graph.number_of_edges()
        self.observational_distribution, self.interventional_distribution = get_obs_and_int_distribution(
            self.causal_graph, self.df)
        self.causal_strengths = get_causal_strengths_all_edges(self.causal_graph, self.df)
        self.adjacency_matrix = get_adjacency_matrix(self.causal_graph)
        self.mec = get_markov_equivalence_class(self.causal_graph)
        self.markov_blanket_all_nodes = {str(node): list(get_markov_blanket_node(self.causal_graph, node)) for node in
                                         self.causal_graph.nodes}
        self.computation_time_metrics = time.time() - start_time

    def _generate_result(self):
        self.result = {
            'computation_time_metrics': self.computation_time_metrics,
            'n_edges': self.n_edges,
            'n_nodes': self.n_nodes,
            'observational_distribution': {k: v.tolist() for k, v in self.observational_distribution.items()},
            'interventional_distribution': self._convert_arrays_to_lists(self.interventional_distribution),
            'causal_strengths_all_nodes': self.causal_strengths,
            'adjacency_matrix': self.adjacency_matrix,
            'markov_equivalence_class': self._mec_to_serializable(self.mec),
            'markov_blanket_all_nodes': self.markov_blanket_all_nodes
        }
        print(f' Time to compute graph-metrics: {round(self.computation_time_metrics, 2)} secs')

    def _convert_arrays_to_lists(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                self._convert_arrays_to_lists(v)
            elif isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d

    @staticmethod
    def _mec_to_serializable(mec: Set[nx.DiGraph]) -> List[List[Tuple[str, str]]]:
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


def get_kl_divergence_gaussian_continuous(mu1: np.ndarray, cov1: np.ndarray, mu2: np.ndarray, cov2: np.ndarray) -> float:
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


def get_obs_and_int_distribution(causal_graph: nx.DiGraph, df: pd.DataFrame):
    model = BayesianNetwork(causal_graph.edges())
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


def get_causal_strengths_all_edges(causal_graph: nx.DiGraph, df: pd.DataFrame):
    causal_model = CausalStrengthsComputationClass(df, causal_graph)
    causal_model.fit_model()
    causal_strengths = causal_model.compute_causal_strengths()
    return causal_strengths


def get_adjacency_matrix(causal_graph: nx.DiGraph):
    return nx.adjacency_matrix(causal_graph).todense().tolist()


def get_markov_equivalence_class(causal_graph: nx.DiGraph):
    mec_computer = MarkovEquivalenceClass(causal_graph)
    return mec_computer.return_equivalence_class()


def get_markov_blanket_node(causal_graph: nx.DiGraph, node: str) -> set:
    parents = list(causal_graph.predecessors(node))
    children = list(causal_graph.successors(node))
    co_parents = {pred for child in children for pred in causal_graph.predecessors(child)}
    co_parents.discard(node)
    return set(parents + children + list(co_parents))
