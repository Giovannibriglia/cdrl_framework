import itertools
import random
import re
import warnings
from multiprocessing import Pool
from typing import Dict, Tuple, List
import causalnex
import networkx as nx
import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causalnex.inference import InferenceEngine
from causalnex.structure.pytorch import from_pandas
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference.CausalInference import CausalInference
from pgmpy.models.BayesianNetwork import BayesianNetwork
from tqdm import tqdm

from causality_vmas import LABEL_reward_action_values, LABEL_discrete_intervals, LABEL_grouped_features
from causality_vmas.utils import dict_to_bn, get_process_and_threads, check_values_in_states, get_df_size

warnings.filterwarnings("ignore", category=RuntimeWarning, module="pgmpy.factors.discrete")


class CausalDiscovery:
    def __init__(self, df: pd.DataFrame = None):

        self.cd_algo = None
        self.features_names = None
        self.causal_graph = None
        self.notears_graph = None
        self.df = None

        random.seed(42)
        np.random.seed(42)

        self.add_data(df)

    def add_data(self, df: pd.DataFrame):
        if self.df is None:
            self.df = df
        else:
            self.df = pd.concat([self.df, df]).reset_index(drop=True)

        self.features_names = self.df.columns.to_list()

        for col in self.df.columns:
            self.df[str(col)] = self.df[str(col)].astype(str).str.replace(',', '').astype(float)

    def training(self, cd_algo: str = 'PC', show_progress: bool = False):
        self.cd_algo = cd_algo
        if cd_algo == 'PC':
            self.causal_graph = self.use_PC(show_progress)

        elif cd_algo == 'NOTEARS':
            self.causal_graph = self.use_NOTEARS()

        elif cd_algo == 'MY':
            notears_graph = self.use_NOTEARS()
            largest_component = max(nx.weakly_connected_components(notears_graph), key=len)
            notears_graph = notears_graph.subgraph(largest_component).copy()
            self.causal_graph = self.use_MY(notears_graph, show_progress)

    def return_causal_graph(self) -> nx.DiGraph:
        structure_to_return_list = [(x[0], x[1]) for x in self.causal_graph.edges]
        structure_to_return = nx.DiGraph(structure_to_return_list)
        return structure_to_return

    def return_df(self):
        return self.df

    def use_PC(self, show_progress) -> nx.DiGraph:

        # Clean the data
        data = self.df.apply(lambda x: x.fillna(x.mean()), axis=0)
        data = data.apply(lambda x: x.fillna(x.mean()), axis=0)
        # Calculate the maximum finite value for each column
        finite_max = data.apply(lambda x: np.max(x[np.isfinite(x)]), axis=0)
        # Replace infinite values in each column with the corresponding maximum finite value
        data = data.apply(lambda x: x.replace([np.inf, -np.inf], finite_max[x.name]), axis=0)
        # Ensure no zero standard deviation
        stddev = data.std()
        if (stddev == 0).any():
            stddev[stddev == 0] = np.inf
        # Convert the cleaned data to a numpy array
        data_array = data.values

        labels = [f'{col}' for i, col in enumerate(self.features_names)]
        cg = pc(data_array, show_progress=show_progress)

        G = nx.DiGraph()
        dict_nodes_number = {}
        # Add nodes
        for node in cg.G.get_nodes():
            node_name = node.get_name()
            node_index = int(node_name[1:]) - 1
            node_label = labels[node_index]
            G.add_node(node_label)
            dict_nodes_number[node_name] = node_label

        # Add edges
        for edge in cg.G.get_graph_edges():
            source_name = edge.get_node1().get_name()
            source_label = dict_nodes_number[source_name]

            target_name = edge.get_node2().get_name()
            target_label = dict_nodes_number[target_name]

            G.add_edge(source_label, target_label)

        return G

    def use_NOTEARS(self) -> nx.DiGraph:
        structure_model = from_pandas(self.df, max_iter=5000, use_gpu=True)
        structure_model.remove_edges_below_threshold(0.2)

        # Convert CausalNex DAG to NetworkX DiGraph
        G = nx.DiGraph()
        # Add nodes
        G.add_nodes_from(structure_model.nodes)
        # Add edges
        G.add_edges_from(structure_model.edges)

        return G

    def use_MY(self, graph: nx.DiGraph, show_progress: bool = False) -> nx.DiGraph:
        print('***************** RIFAREEEEEEEEEEEEE ******************')
        df = self.df.copy()

        print('Bayesian network definition...')
        bn = causalnex.network.BayesianNetwork(graph.edges)
        print('Bayesian network fitting...')
        bn = bn.fit_node_states_and_cpds(df)

        bad_nodes = [node for node in bn.nodes if not re.match("^[0-9a-zA-Z_]+$", node)]
        if bad_nodes:
            print('Bad nodes: ', bad_nodes)
        print('Inference engine definition...')
        ie = InferenceEngine(bn)

        # Initial assumption: all nodes are independent until proven dependent
        independent_vars = set(graph.nodes)
        dependent_vars = set()
        causal_relationships = []

        # Precompute unique values for all nodes
        unique_values = {node: df[node].unique() for node in graph.nodes}

        # Initial query to get the baseline distributions
        before = ie.query()
        print('Start causality assessment...')

        if show_progress:
            pbar = tqdm(graph.nodes, desc=f'nodes')
        else:
            pbar = graph.nodes
        for node in pbar:
            if node in dependent_vars:
                continue  # Skip nodes already marked as dependent

            connected_nodes = list(self.notears_graph.neighbors(node))
            change_detected = False

            # Perform interventions on the node and observe changes in all connected nodes
            for value in unique_values[node]:
                dict_set_probs = {}
                if change_detected:
                    continue

                for key in unique_values[node]:
                    dict_set_probs[key] = 1.0 if key == value else 0.0

                try:
                    ie.do_intervention(str(node), dict_set_probs)
                    after = ie.query()

                    # Check each connected node for changes in their distribution
                    for conn_node in connected_nodes:
                        best_key_before, max_value_before = max(before[conn_node].items(), key=lambda x: x[1])
                        best_key_after, max_value_after = max(after[conn_node].items(), key=lambda x: x[1])
                        uniform_probability_value = round(1 / len(after[conn_node]), 8)

                        if max_value_after > uniform_probability_value and best_key_after != best_key_before:
                            dependent_vars.add(conn_node)  # Mark as dependent
                            if conn_node in independent_vars:
                                independent_vars.remove(conn_node)  # Remove from independents
                                print(f"Link removed: {node} -> {conn_node}")
                            change_detected = True
                            causal_relationships.append((node, conn_node))  # Ensure this is a 2-tuple

                    # Also check the intervened node itself
                    best_key_before, max_value_before = max(before[node].items(), key=lambda x: x[1])
                    best_key_after, max_value_after = max(after[node].items(), key=lambda x: x[1])
                    uniform_probability_value = round(1 / len(after[node]), 8)

                    if max_value_after > uniform_probability_value and best_key_after != best_key_before:
                        change_detected = True

                    ie.reset_do(str(node))
                except Exception as e:
                    print(f'{node} = {value}, {e}')

            if change_detected:
                dependent_vars.add(node)  # Mark as dependent
                independent_vars.discard(node)  # Remove from independents

        G = nx.DiGraph()
        # Add edges from the list
        G.add_edges_from(causal_relationships)

        return G

class SingleCausalInference:
    def __init__(self, df: pd.DataFrame, causal_graph: nx.DiGraph, dict_init_cbn: Dict = None):
        self.df = df
        self.causal_graph = causal_graph
        self.features = self.causal_graph.nodes

        if dict_init_cbn is None and (df is None and causal_graph is None):
            raise ImportError('dataframe - causal graph - bayesian network are None')

        if dict_init_cbn is None:
            self.cbn = BayesianNetwork()
            self.cbn.add_edges_from(ebunch=self.causal_graph.edges())
            self.cbn.fit(self.df, estimator=MaximumLikelihoodEstimator)
        else:
            self.cbn = dict_to_bn(dict_init_cbn)
        del dict_init_cbn
        assert self.cbn.check_model()

        self.ci = CausalInference(self.cbn)

    def return_cbn(self) -> BayesianNetwork:
        return self.cbn

    def return_discrete_intervals_bn(self):
        intervals_dict = {}
        for node in self.cbn.nodes():
            cpd = self.cbn.get_cpds(node)
            if cpd:
                # Assuming discrete nodes with states
                intervals_dict[node] = cpd.state_names[node]
        return intervals_dict

    def infer(self, input_dict_do: Dict, target_variable: str, evidence=None, adjustment_set=None) -> Dict:
        # print(f'infer: {input_dict_do} - {target_variable}')

        # Ensure the target variable is not in the evidence
        input_dict_do_ok = {k: v for k, v in input_dict_do.items() if k != target_variable}

        # print(f'Cleaned input (evidence): {input_dict_do_ok}')
        # print(f'Target variable: {target_variable}')

        if adjustment_set is None:
            # Compute an adjustment set if not provided
            do_vars = [var for var, state in input_dict_do_ok.items()]
            adjustment_set = set(
                itertools.chain(*[self.causal_graph.predecessors(var) for var in do_vars])
            )
            # print(f'Computed adjustment set: {adjustment_set}')
        else:
            # print(f'Provided adjustment set: {adjustment_set}')
            pass

        # Ensure target variable is not part of the adjustment set
        adjustment_set.discard(target_variable)
        query_result = self.ci.query(
            variables=[target_variable],
            do=input_dict_do_ok,
            evidence=input_dict_do_ok if evidence is None else evidence,
            adjustment_set=adjustment_set,
            show_progress=False
        )
        # print(f'Query result: {query_result}')

        # Convert DiscreteFactor to a dictionary
        result_dict = {str(state): float(query_result.values[idx]) for idx, state in
                       enumerate(query_result.state_names[target_variable])}
        # print(f'Result distributions: {result_dict}')

        return result_dict

    @staticmethod
    def _check_states(input_dict_do, evidence, adjustment_set):
        def is_numeric(value):
            try:
                # Convert value to a numpy array and check for NaN or infinite values
                value = np.array(value, dtype=float)
                return np.isnan(value).any() or np.isinf(value).any()
            except (ValueError, TypeError):
                return False

        def check_for_nan_or_infinite(data):
            if isinstance(data, dict):
                for key, value in data.items():
                    if is_numeric(value):
                        print(f"Warning: {key} contains NaN or infinite values.")
                        return True
            elif isinstance(data, set):
                for value in data:
                    if is_numeric(value):
                        print(f"Warning: Set contains NaN or infinite values: {value}")
                        return True
            else:
                print(f"Unsupported data type: {type(data)}")
                return True  # Return True to signal an issue if data type is unsupported
            return False

        # Check the input data
        if check_for_nan_or_infinite(input_dict_do):
            raise ValueError("Input data contains NaN or infinite values.")
        if evidence is not None and check_for_nan_or_infinite(evidence):
            raise ValueError("Evidence data contains NaN or infinite values.")
        if adjustment_set is not None and check_for_nan_or_infinite(adjustment_set):
            raise ValueError("Adjustment set data contains NaN or infinite values.")

class CausalInferenceForRL:
    def __init__(self, online: bool, df_train: pd.DataFrame, causal_graph: nx.DiGraph,
                 bn_dict: Dict = None, causal_table: pd.DataFrame = None,
                 obs_train_to_test=None, grouped_features: Tuple = None):

        self.online = online

        self.df_train = df_train
        self.causal_graph = causal_graph
        self.bn_dict = bn_dict

        self.obs_train_to_test = obs_train_to_test

        del df_train, causal_graph, bn_dict

        self.ci = SingleCausalInference(self.df_train, self.causal_graph, self.bn_dict)

        self.discrete_intervals_bn = self.ci.return_discrete_intervals_bn()

        self.grouped_features = grouped_features

        self.reward_variable = [s for s in self.df_train.columns.to_list() if 'reward' in s][0]
        self.reward_values = self.df_train[self.reward_variable].unique().tolist()
        self.action_variable = [s for s in self.df_train.columns.to_list() if 'action' in s][0]
        self.action_values = self.df_train[self.action_variable].unique().tolist()

        self.causal_table = causal_table

        self.n_threads, self.n_processes = get_process_and_threads()

    def _single_query(self, obs: Dict) -> Dict:
        def get_action_distribution(reward_value):
            # Create a new evidence dictionary by merging obs with the current reward_value
            evidence = {**obs, f'{self.reward_variable}': reward_value}
            # Check the values in the states
            check_values_in_states(self.ci.cbn.states, obs, evidence)
            # Infer the action distribution based on the evidence
            return self.ci.infer(obs, self.action_variable, evidence)

        # Construct the result dictionary using a dictionary comprehension
        reward_actions_values = {
            reward_value: get_action_distribution(reward_value)
            for reward_value in self.reward_values
        }

        return reward_actions_values

    def _compute_reward_action_values(self, input_obs: Dict) -> Dict:
        if self.obs_train_to_test is not None:
            kwargs = {}
            kwargs[LABEL_discrete_intervals] = self.discrete_intervals_bn
            kwargs[LABEL_grouped_features] = self.grouped_features

            obs = self.obs_train_to_test(input_obs, **kwargs)
        else:
            obs = input_obs

        reward_action_values = self._single_query(obs)

        row_result = obs.copy()
        row_result[f'{LABEL_reward_action_values}'] = reward_action_values

        return row_result

    def return_reward_action_values(self, input_obs: Dict) -> Dict:
        if self.online:
            dict_input_and_rav = self._compute_reward_action_values(input_obs)
            reward_action_values = dict_input_and_rav[LABEL_reward_action_values]
        else:
            if self.causal_table is None:
                self.causal_table = self.create_causal_table(show_progress=True)

            if self.obs_train_to_test is not None:
                kwargs = {LABEL_discrete_intervals: self.discrete_intervals_bn,
                          LABEL_grouped_features: self.grouped_features}

                obs = self.obs_train_to_test(input_obs, **kwargs)
            else:
                obs = input_obs

            filtered_df = self.causal_table.copy()
            # TODO: try to speed-up with smart search
            for feature, value in obs.items():
                filtered_df = filtered_df[filtered_df[feature] == value]

            reward_action_values = filtered_df[LABEL_reward_action_values].to_dict()

            if reward_action_values == {}:
                reward_action_values = {reward_value: {action_value: np.nan for action_value in self.action_values}
                                        for reward_value in self.reward_values}

        return reward_action_values

    def _process_combination(self, combination: Dict) -> Dict:
        # initial_time = time.time()
        try:
            reward_action_values = self._single_query(combination)
        except Exception as e:
            raise e

        return reward_action_values

    @staticmethod
    def _create_dataframe_chunk(combinations_chunk: List, show_progress: bool) -> pd.DataFrame:
        if show_progress:
            return pd.DataFrame(tqdm(combinations_chunk, desc="Processing single combination..."))
        else:
            return pd.DataFrame(combinations_chunk)

    def _update_causal_table_chunk(self, chunk: pd.DataFrame, show_progress: bool, parallel: bool) -> pd.DataFrame:
        rows_as_tuples = chunk.to_dict(orient='records')

        if parallel:
            with Pool(self.n_processes) as pool:
                if show_progress:
                    results = list(pool.map(self._process_combination, tqdm(rows_as_tuples,
                                                                            total=len(rows_as_tuples),
                                                                           desc='Processing chunk in parallel...')))
                else:
                    results = list(pool.map(self._process_combination, rows_as_tuples))
        else:
            if show_progress:
                results = list(map(self._process_combination, tqdm(rows_as_tuples,
                                                                   total=len(rows_as_tuples),
                                                                   desc='Processing chunk...')))
            else:
                results = list(map(self._process_combination, rows_as_tuples))

        chunk[LABEL_reward_action_values] = [
            {str(k): v for k, v in d.items()} if isinstance(d, dict) else d for d in results
        ]
        return chunk

    def _process_chunk(self, chunk: List, show_progress: bool, parallel: bool) -> pd.DataFrame:
        df_chunk = self._create_dataframe_chunk(chunk, show_progress)
        df_chunk = self._update_causal_table_chunk(df_chunk, show_progress, parallel)
        return df_chunk

    def create_causal_table(self, show_progress: bool = True, parallel: bool = True):
        model = self.ci.return_cbn()

        variables = model.nodes()
        excluded_variables = {"agent_0_reward", "agent_0_action_0", "agent_0_action_1"}
        state_names = {variable: model.get_cpds(variable).state_names[variable] for variable in variables if
                       variable not in excluded_variables}
        print(state_names)
        all_combinations = list(itertools.product(*state_names.values()))
        total_combinations = len(all_combinations)

        if show_progress:
            combinations_dicts = [dict(zip(state_names.keys(), combination)) for
                                  combination in tqdm(all_combinations, total=total_combinations,
                                                      desc='Generating all possible state-value combinations...')]
        else:
            combinations_dicts = [dict(zip(state_names.keys(), combination)) for combination in all_combinations]

        del all_combinations

        return self._process_chunk(combinations_dicts, show_progress, parallel)


