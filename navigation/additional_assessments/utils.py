import time
from collections import Counter
from itertools import combinations
from typing import Dict, Tuple, List, Union, Set
import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from scipy.stats import ks_2samp
from sklearn.preprocessing import LabelEncoder

from navigation.causality_algos import CausalDiscovery


def run_causal_discovery(df: pd.DataFrame, cd_algo: str = 'pc', show_progress_cd: bool = False):
    initial_time = time.time()

    cd = CausalDiscovery(df)
    cd.training(cd_algo, show_progress=show_progress_cd)
    causal_graph = cd.return_causal_graph()

    # Remove edges between "sensor" variables
    edges_to_remove = [(u, v) for u, v in causal_graph.edges() if 'sensor' in u and 'sensor' in v]
    causal_graph.remove_edges_from(edges_to_remove)

    computation_time1 = time.time() - initial_time  # seconds
    initial_time2 = time.time()

    n_nodes = causal_graph.number_of_nodes()
    n_edges = causal_graph.number_of_edges()

    observational_distribution, interventional_distribution = get_obs_and_int_distribution(causal_graph, df)

    causal_strengths = get_causal_strengths_all_edges(causal_graph, df)

    adjacency_matrix = get_adjacency_matrix(causal_graph)

    mec = get_markov_equivalence_class(causal_graph)

    markov_blanket_all_nodes = {str(node): list(markov_blanket(causal_graph, node)) for node in causal_graph.nodes}

    computation_time2 = time.time() - initial_time2  # seconds

    result = {
        'causal_graph': [(x[0], x[1]) for x in causal_graph.edges],
        'computation_time1_sec': computation_time1,
        'computation_time2_sec': computation_time2,
        'n_edges': n_edges,
        'n_nodes': n_nodes,
        'observational_distribution': {k: v.tolist() for k, v in observational_distribution.items()},
        'interventional_distribution': convert_arrays_to_lists(interventional_distribution),
        'causal_strengths_all_nodes': causal_strengths,
        'adjacency_matrix': adjacency_matrix,
        'markov_equivalence_class': mec_to_serializable(mec),
        'markov_blanket_all_nodes': markov_blanket_all_nodes
    }

    print(f'Time to perform CD: {computation_time1} secs and Time to compute graph-metrics: {computation_time2} secs')

    return result


# Function to convert the Markov equivalence class to a serializable format
def mec_to_serializable(mec: Set[nx.DiGraph]) -> List[List[Tuple[str, str]]]:
    return [convert_graph_to_edgelist(g) for g in mec]


# Function to convert the serializable format back to the Markov equivalence class
def serializable_to_mec(serialized_mec: List[List[Tuple[str, str]]]) -> Set[nx.DiGraph]:
    return {convert_edgelist_to_graph(edgelist) for edgelist in serialized_mec}


def convert_graph_to_edgelist(graph: nx.DiGraph) -> List[Tuple[str, str]]:
    return list(graph.edges())


def convert_edgelist_to_graph(edgelist: List[Tuple[str, str]]) -> nx.DiGraph:
    return nx.DiGraph(edgelist)


# Convert NumPy arrays to lists
def convert_arrays_to_lists(d):
    for k, v in d.items():
        if isinstance(v, dict):
            convert_arrays_to_lists(v)
        elif isinstance(v, np.ndarray):
            d[k] = v.tolist()


# Function to calculate Structural Hamming Distance (SHD)
def shd(G1: nx.DiGraph, G2: nx.DiGraph):
    edges1 = set(G1.edges())
    edges2 = set(G2.edges())

    # Edge additions and deletions
    additions = edges2 - edges1
    deletions = edges1 - edges2

    # Edge reversals
    reversals = set((v, u) for (u, v) in edges1).intersection(edges2)

    shd_value = len(additions) + len(deletions) + len(reversals)
    return shd_value


# Placeholder for Structural Intervention Distance (SID)
# Implementation of SID requires causal inference capabilities, not included in standard libraries
def sid(G1: nx.DiGraph, G2: nx.DiGraph):
    raise NotImplementedError("SID computation is not implemented")


# Function to calculate Jaccard Similarity
def jaccard_similarity(G1: nx.DiGraph, G2: nx.DiGraph):
    edges1 = set(G1.edges())
    edges2 = set(G2.edges())
    intersection = edges1.intersection(edges2)
    union = edges1.union(edges2)
    return len(intersection) / len(union)


# Function to calculate Degree Distribution Similarity (Kolmogorov-Smirnov)
def degree_distribution_similarity(G1: nx.DiGraph, G2: nx.DiGraph):
    degrees1 = [d for n, d in G1.degree()]
    degrees2 = [d for n, d in G2.degree()]
    return ks_2samp(degrees1, degrees2, method='asymp').statistic


# Function to calculate Clustering Coefficient Similarity
def clustering_coefficient_similarity(G1: nx.DiGraph, G2: nx.DiGraph):
    cc1 = nx.average_clustering(G1)
    cc2 = nx.average_clustering(G2)
    return abs(cc1 - cc2)


# Function to calculate Graphlet Degree Distribution Agreement (GDDA)
def gdda(G1: nx.DiGraph, G2: nx.DiGraph, k=3):
    def count_graphlets(graph, k):
        graphlets = [graph.subgraph(nodes).copy() for nodes in combinations(graph.nodes(), k)]
        graphlet_degrees = [nx.degree_histogram(graphlet) for graphlet in graphlets]
        return Counter(tuple(map(tuple, graphlet_degrees)))

    count1 = count_graphlets(G1, k)
    count2 = count_graphlets(G2, k)

    all_graphlets = set(count1.keys()).union(set(count2.keys()))
    gdda_value = sum(abs(count1[g] - count2[g]) for g in all_graphlets)
    return gdda_value


# Function to compute markov blanket for a graph
def markov_blanket(graph: nx.DiGraph, node: str) -> set:
    # Markov blanket of a node is its parents, children, and co-parents (parents of its children)
    parents = list(graph.predecessors(node))
    children = list(graph.successors(node))
    co_parents = set()
    for child in children:
        co_parents.update(graph.predecessors(child))
    co_parents.discard(node)
    markov_blanket = set(parents + children + list(co_parents))
    return markov_blanket


def get_adjacency_matrix(graph: nx.DiGraph):
    return nx.adjacency_matrix(graph).todense().tolist()


def get_obs_and_int_distribution(dag: nx.DiGraph, data: pd.DataFrame):
    # Convert the DAG to a Bayesian Network
    model = BayesianNetwork(dag.edges())

    # Step 2: Estimate the observational distribution
    model.fit(data, estimator=MaximumLikelihoodEstimator)

    # Function to compute all observational distributions
    def compute_observational_distributions(model):
        infer = VariableElimination(model)
        observational_distributions = {}

        for var in model.nodes():
            result = infer.query([var])
            observational_distributions[var] = result.values

        return observational_distributions

    # Function to compute all interventional distributions
    def compute_all_interventional_distributions(model, data):
        infer = VariableElimination(model)
        interventional_distributions = {}

        for intervention_var in model.nodes():
            interventional_distributions[intervention_var] = {}

            for intervention_value in data[intervention_var].unique():
                intervention = {intervention_var: intervention_value}
                distributions = {}

                for var in model.nodes():
                    if var != intervention_var:
                        result = infer.query([var], intervention)
                        distributions[var] = result.values

                interventional_distributions[intervention_var][intervention_value] = distributions

        return interventional_distributions

    # Compute observational distributions
    observational_distributions = compute_observational_distributions(model)
    # Compute all interventional distributions
    interventional_distributions = compute_all_interventional_distributions(model, data)

    return observational_distributions, interventional_distributions


def get_causal_strengths_all_edges(dag: nx.DiGraph, data: pd.DataFrame):
    # Initialize the CausalModel class
    causal_model = Obs_Int_Distributions(data, dag)

    # Fit the model
    causal_model.fit_model()

    # Compute causal strengths for all edges
    causal_strengths = causal_model.compute_causal_strengths()

    return causal_strengths


def is_positive_definite(matrix: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


def kl_divergence_gaussian_continuous(mu1: np.ndarray, cov1: np.ndarray, mu2: np.ndarray, cov2: np.ndarray) -> float:
    # Check if cov2 is positive definite
    if not is_positive_definite(cov2):
        # Apply regularization if cov2 is not positive definite
        epsilon = 1e-10
        cov2 += epsilon * np.eye(cov2.shape[0])

    # KL Divergence for two Gaussian distributions
    dim = len(mu1)
    cov2_inv = np.linalg.inv(cov2)
    term1 = np.trace(cov2_inv @ cov1)
    term2 = (mu2 - mu1).T @ cov2_inv @ (mu2 - mu1)
    term3 = np.log(np.linalg.det(cov2) / np.linalg.det(cov1))
    return 0.5 * (term1 + term2 - dim + term3)


def kl_divergence_discrete(p: np.ndarray, q: np.ndarray) -> float:
    # KL Divergence for two discrete distributions
    return np.sum(p * np.log(p / q))


class Obs_Int_Distributions:
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

    def get_joint_distribution(self) -> np.ndarray:
        joint_dist = self.data.value_counts(normalize=True).values
        return joint_dist

    def compute_covariance_matrix(self, data: pd.DataFrame) -> np.ndarray:
        return np.cov(data, rowvar=False)

    def get_interventional_distribution_or_covariance(self, removed_edge: Tuple[str, str]) -> Union[
        np.ndarray, np.ndarray]:
        source, target = removed_edge
        interventional_data = self.data.copy()
        interventional_data[target] = np.random.permutation(interventional_data[target].values)
        if self.is_continuous:
            return self.compute_covariance_matrix(interventional_data)
        else:
            return self.get_joint_distribution()

    def compute_causal_strengths(self) -> Dict[str, float]:
        causal_strengths = {}

        for edge in self.dag.edges():
            if self.is_continuous:
                original_cov = self.compute_covariance_matrix(self.data)
                original_mean = self.data.mean().values

                interventional_cov = self.get_interventional_distribution_or_covariance(edge)
                interventional_mean = self.data.mean().values
                cs = kl_divergence_gaussian_continuous(original_mean, original_cov, interventional_mean,
                                                       interventional_cov)

            else:
                original_dist = self.get_joint_distribution()

                interventional_dist = self.get_interventional_distribution_or_covariance(edge)
                cs = kl_divergence_discrete(original_dist, interventional_dist)

            causal_strengths[str(edge)] = cs

        return causal_strengths


def get_markov_equivalence_class(dag: nx.DiGraph):
    mec_computer = MarkovEquivalenceClass(dag)
    return mec_computer.return_equivalence_class()


class MarkovEquivalenceClass:
    def __init__(self, dag: nx.DiGraph):
        self.dag = dag

    @staticmethod
    def skeleton(graph: nx.DiGraph) -> nx.Graph:
        """ Returns the skeleton of a DAG (i.e., its undirected version). """
        return nx.Graph(graph)

    @staticmethod
    def find_v_structures(graph: nx.DiGraph) -> List[Tuple[int, int, int]]:
        """ Identify v-structures in the graph. """
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
        """ Check if two graphs are Markov equivalent. """
        return (MarkovEquivalenceClass.skeleton(graph1).edges == MarkovEquivalenceClass.skeleton(graph2).edges and
                set(MarkovEquivalenceClass.find_v_structures(graph1)) == set(
                    MarkovEquivalenceClass.find_v_structures(graph2)))

    @staticmethod
    def covered_edge_flips(graph: nx.DiGraph) -> List[nx.DiGraph]:
        """ Generate all graphs from covered edge flips of the given graph. """
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
        """ Generate the Markov equivalence class for the given DAG. """
        mec = set()
        to_visit = [self.dag]
        while to_visit:
            current = to_visit.pop()
            if any(self.markov_equivalent(current, g) for g in mec):
                continue
            mec.add(current)
            to_visit.extend(self.covered_edge_flips(current))

        return mec


"""def evaluate_results_sensitive_analysis_causal_graphs(json_file_folder: str, employed_dataframe: pd.DataFrame,
                                                      x_semidim: float = None, y_semidim: float = None):
    graphs = []
    # Load all graphs from JSON files
    for file_name in os.listdir(json_file_folder):
        if file_name.endswith('.json'):
            with open(os.path.join(json_file_folder, file_name), 'r') as file:
                data = json.load(file)
                graph = data['causal_graph']
                file_name_to_save = file_name.replace('causal_graph_', '').replace('.json', '')
                graphs.append((file_name_to_save, nx.DiGraph(graph)))

    results = []

    # Compare each graph with every other graph
    pbar = tqdm(range(len(graphs)), 'Evaluation...')
    for i in pbar:
        for j in range(i, len(graphs)):
            file_name_1, G1 = graphs[i]
            file_name_2, G2 = graphs[j]

            n_bins1, n_sensors1 = _get_n_bins_and_n_sensors(file_name_1)
            n_bins2, n_sensors2 = _get_n_bins_and_n_sensors(file_name_2)

            single_row_results = compare_two_graphs(G1, G2, employed_dataframe, n_bins1, n_sensors1, n_bins2,
                                                    n_sensors2, x_semidim, y_semidim)

            single_row_results['n_bins_g1'] = n_bins1
            single_row_results['n_sensors_g1'] = n_sensors1
            single_row_results['n_bins_g2'] = n_bins2
            single_row_results['n_sensors_g2'] = n_sensors2

            results.append(single_row_results)

    results_df = pd.DataFrame(results)

    # get_df_boundaries(results_df)

    metrics = list(metrics_to_compare_causal_graphs().keys())
    for metric in metrics:
        plot_relationships(results_df, metric)

    results_df = group_as_similarity_score(results_df)

    return results_df


def group_as_similarity_score(data: pd.DataFrame):
    thresholds = {
        'shd': 5,
        'js': 0.8,
        'dds': 0.1,
        'ccs': 0.1,
        'kld': 0.1
    }

    def similarity_score(row, thresholds):
        score = 0
        score += (row['shd'] <= thresholds['shd'])
        score += (row['js'] >= thresholds['js'])
        score += (row['dds'] <= thresholds['dds'])
        score += (row['ccs'] <= thresholds['ccs'])
        score += (row['kld'] <= thresholds['kld'])
        return score

    # Generate all possible combinations of n_bins and n_sensors
    bins_sensors_combinations = list(
        itertools.product(data['n_bins_g1'].unique(), data['n_sensors_g1'].unique()))

    # Create a DataFrame to store the similarity scores for each combination
    results = []

    # Iterate over all combinations of bins and sensors
    for (bins1, sensors1), (bins2, sensors2) in itertools.product(bins_sensors_combinations, repeat=2):
        if bins1 == bins2 and sensors1 == sensors2:
            continue  # Skip comparing the same configuration

        subset1 = data[(data['n_bins_g1'] == bins1) & (data['n_sensors_g1'] == sensors1)]
        subset2 = data[(data['n_bins_g2'] == bins2) & (data['n_sensors_g2'] == sensors2)]

        if subset1.empty or subset2.empty:
            continue

        # Calculate the mean similarity score between the two subsets
        mean_score = (subset1.apply(similarity_score, axis=1, thresholds=thresholds).mean() +
                      subset2.apply(similarity_score, axis=1, thresholds=thresholds).mean()) / 2

        results.append({'n_bins1': bins1, 'n_sensors1': sensors1, 'n_bins2': bins2, 'n_sensors2': sensors2,
                        'mean_similarity_score': mean_score})

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Aggregate by averaging the mean_similarity_score for each unique combination
    aggregated_results = results_df.groupby(['n_bins1', 'n_sensors1']).agg(
        {'mean_similarity_score': 'mean'}).reset_index()

    # Sort by similarity scores to identify the best combinations
    best_combinations = aggregated_results.sort_values(by='mean_similarity_score', ascending=False)

    print(best_combinations.head(5))

    # Plot the best combinations for visual analysis
    plt.figure(figsize=(12, 8), dpi=1000)
    sns.heatmap(best_combinations.pivot(index='n_bins1', columns='n_sensors1', values='mean_similarity_score'),
                annot=True,
                fmt='.2f', cmap='Blues')
    plt.title('#bins and #sensors vs. similarity score')
    plt.xlabel('#sensors')
    plt.ylabel('#bins')
    plt.savefig('scores.png')


def metrics_to_compare_causal_graphs() -> Dict:
    metrics = {'shd': shd,
               # "sid": sid(G1, G2),
               # "ged": ged(G1, G2),
               "js": jaccard_similarity,
               "dds": degree_distribution_similarity,
               "ccs": clustering_coefficient_similarity,
               "kld": kullback_leibler_divergence,
               "gdda": gdda
               }
    return metrics


def compare_two_graphs(G1: nx.DiGraph, G2: nx.DiGraph, data: pd.DataFrame, n_bins1: int, n_sensors1: int, n_bins2: int,
                       n_sensors2: int, x_semidim: float = None, y_semidim: float = None):
    metrics = metrics_to_compare_causal_graphs()

    new_data1 = discretize_df(data, n_bins1, n_sensors1, x_semidim, y_semidim)
    new_data2 = discretize_df(data, n_bins2, n_sensors2, x_semidim, y_semidim)
    markov_blanket_and_equivalence_class(G1, G2, new_data1, new_data2)

    comparison_results = {}
    for metric_name, metric_func in metrics.items():
        try:
            comparison_results[metric_name] = metric_func(G1, G2)
        except Exception as e:
            comparison_results[metric_name] = str(e)

    return comparison_results


def _get_n_bins_and_n_sensors(input_string: str) -> Tuple[int, int]:
    # Define the regex pattern
    pattern = r"(\d+)bins_(\d+)sensors"

    # Search the string for the pattern
    match = re.search(pattern, input_string)

    if match:
        bins = int(match.group(1))
        sensors = int(match.group(2))
        # print(f"Input: {input_string}, Bins: {bins}, Sensors: {sensors}")
        return bins, sensors
    else:
        print("No match found")
        return None, None


def plot_relationships(data: pd.DataFrame, metric: str):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=500)

    # Plot metric vs. number of bins
    axes[0].scatter(data['n_bins_g1'], data[metric], label='g1', s=60)
    axes[0].scatter(data['n_bins_g2'], data[metric], label='g2', alpha=0.7, s=30)
    axes[0].set_title(f'{metric} vs. #bins')
    axes[0].set_xlabel('#bins')
    axes[0].set_ylabel(metric)
    axes[0].legend()

    # Plot metric vs. number of sensors
    axes[1].scatter(data['n_sensors_g1'], data[metric], label='g1', s=60)
    axes[1].scatter(data['n_sensors_g2'], data[metric], label='g2', alpha=0.7, s=30)
    axes[1].set_title(f'{metric} vs. #sensors')
    axes[1].set_xlabel('#sensors')
    axes[1].set_ylabel(metric)
    axes[1].legend()

    plt.tight_layout()
    plt.show()"""
