import itertools
import random
import re
from typing import Dict

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
from tqdm.auto import tqdm

from causality_vmas.utils import dict_to_bn

COL_REWARD_ACTION_VALUES = 'reward_action_values'


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
            pbar = tqdm(graph.nodes, desc=f'{self.env_name} nodes')
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


"""class CausalInferenceForRL:
    def __init__(self, df: pd.DataFrame = None, causal_graph: StructureModel = None, causal_table: pd.DataFrame = None,
                 dir_name: str = None, env_name: str = None):

        self.unique_values_df = None
        self.action_space_size = None
        self.action_column = None
        self.reward_column = None
        random.seed(42)
        np.random.seed(42)
        self.df = None
        self.possible_reward_values = None
        self.causal_table = causal_table
        self.ie = None
        self.bn = None
        self.causal_graph = None

        self.dir_saving = f'{dir_name}/{env_name}'
        os.makedirs(self.dir_saving, exist_ok=True)

        if causal_graph is not None and df is not None:
            if isinstance(causal_graph, list):
                sm = StructureModel()
                sm.add_edges_from([(node1, node2) for node1, node2 in causal_graph])
                self.add_data(df, sm)
            elif isinstance(causal_graph, StructureModel):
                self.add_data(df, causal_graph)
        else:
            self.action_space_size = 9

    def add_data(self, new_df: pd.DataFrame, new_graph: StructureModel):
        if self.df is None:
            self.df = new_df
        else:
            self.df = pd.concat([self.df, new_df]).reset_index(drop=True)

        self.reward_column = [s for s in self.df.columns.to_list() if 'reward' in s][0]
        self.action_column = [s for s in self.df.columns.to_list() if 'action' in s][0]

        self.unique_values_df = {col: self.df[col].unique().tolist() for col in self.df.columns}
        self.action_space_size = int(max(self.df[self.action_column].unique()))
        self.possible_reward_values = self.df[self.reward_column].unique()

        for col in self.df.columns:
            self.df[str(col)] = self.df[str(col)].astype(str).str.replace(',', '').astype(float)

        self.causal_graph = new_graph

        try:
            # print('bayesian network definition...')
            self.bn = causalnex.network.BayesianNetwork(self.causal_graph)
            # print('bayesian network fitting...')
            self.bn = self.bn.fit_node_states_and_cpds(self.df)

            bad_nodes = [node for node in self.bn.nodes if not re.match("^[0-9a-zA-Z_]+$", node)]
            if bad_nodes:
                print('Bad nodes: ', bad_nodes)
            # print('inference engine definition...')
            self.ie = InferenceEngine(self.bn)
        except:
            self.bn = None
            self.ie = None

    def get_actions_rewards_values(self, observation: Dict, online: bool) -> dict:

        def _compute_averaged_mean(input_dict):

            def _rescale_reward(past_value: float) -> float:
                old_max = max(self.unique_values_df[self.reward_column])
                old_min = min(self.unique_values_df[self.reward_column])
                new_max = 1
                new_min = 0

                new_value = new_min + ((past_value - old_min) / (old_max - old_min)) * (new_max - new_min)
                return new_value

            averaged_mean_dict = {}
            for key_action, dict_rewards in input_dict.items():
                reward_values = list(dict_rewards.keys())
                reward_values_prob = list(dict_rewards.values())

                average_weighted_reward = sum(
                    [reward_values[n] * reward_values_prob[n] for n in range(len(dict_rewards))])
                averaged_mean_dict[key_action] = round(average_weighted_reward, 3)

            return averaged_mean_dict

        if online:  # online causal inference
            if self.bn is not None and self.ie is not None:
                # print('online bn')
                action_reward_values = inference_function(observation, self.ie, self.reward_column, self.action_column,
                                                          self.unique_values_df)
                weighted_average_act_rew_values = _compute_averaged_mean(action_reward_values)
                return weighted_average_act_rew_values
            else:
                uniform_prob = 1 / self.action_space_size
                return {key: uniform_prob for key in range(self.action_space_size)}
        else:  # offline causal inference
            if self.causal_table is not None:
                filtered_df = self.causal_table.copy()
                for feature, value in observation.items():
                    filtered_df = filtered_df[filtered_df[feature] == value]
                print('**** METTI A POSTO NEL CASO ***')
                if not filtered_df.empty:
                    return filtered_df['reward_action_values'].values[0]
                else:
                    print('No reward-action values for this observation are available')
                    uniform_prob = 1 / self.action_space_size
                    return {key: uniform_prob for key in range(self.action_space_size)}
            else:
                raise ValueError('There is no causal table')

    def create_causal_table(self) -> pd.DataFrame:
        features = self.df.columns.to_list()
        not_observations = [s for s in features if s not in ['reward', 'action']]

        unique_values = [self.df[column].unique() for column in not_observations]
        combinations = list(itertools.product(*unique_values))
        combinations_list = [dict(zip(not_observations, combination)) for combination in combinations]

        num_chunks = multiprocessing.cpu_count()
        chunk_size = len(combinations_list) // num_chunks + 1
        chunks = [combinations_list[i:i + chunk_size] for i in range(0, len(combinations_list), chunk_size)]

        with multiprocessing.Pool(processes=num_chunks) as pool:
            results = pool.starmap(process_chunk, [(chunk, self.df, self.causal_graph) for chunk in chunks])

        all_rows = [row for result in results for row in result]

        self.causal_table = pd.DataFrame(all_rows)

        self.causal_table.to_pickle(f'{self.dir_saving}/causal_table.pkl')

        return self.causal_table


def process_chunk(chunk: Tuple, df: pd.DataFrame, causal_graph: StructureModel):
    ie = InferenceEngine(causalnex.network.BayesianNetwork(causal_graph).fit_node_states_and_cpds(df))
    rows = []
    pbar = tqdm(chunk, desc=f'Preparing causal table', leave=True)
    col_reward = [s for s in df.columns.to_list() if 'reward' in s][0]
    col_action = [s for s in df.columns.to_list() if 'action' in s][0]
    unique_values_df = {col: df[col].unique().tolist() for col in df.columns}
    for comb in pbar:
        reward_action_values = inference_function(comb, ie, col_reward, col_action, unique_values_df)
        new_row = comb.copy()
        new_row[COL_REWARD_ACTION_VALUES] = reward_action_values
        rows.append(new_row)
    return rows


def inference_function(observation: Dict, ie: InferenceEngine, reward_col: str,
                       action_col: str, unique_values_df: Dict):
    def _find_nearest(array, val):
        array = np.asarray(array)
        idx = (np.abs(array - val)).argmin()
        return array[idx]

    def _reverse_dict(d):
        reversed_d = {}
        for key, value in d.items():
            if isinstance(value, dict):
                reversed_value = _reverse_dict(value)
                for subkey, subvalue in reversed_value.items():
                    if subkey not in reversed_d:
                        reversed_d[subkey] = {}
                    reversed_d[subkey][key] = subvalue
            else:
                reversed_d[key] = value
        return reversed_d

    def _create_action_reward_values():
        action_reward_values = {}

        for value_action in unique_values_df[action_col]:
            try:
                action_reward_values[value_action] = ie.query({action_col: value_action})[reward_col]
            except Exception as e_action:
                print(f"Exception occurred while querying by action_col for value {value_action}: {e_action}")

                # Try querying with reward_col if action_col fails
                try:
                    reward_action_values = {}
                    for value_reward in unique_values_df[reward_col]:
                        try:
                            reward_action_values[value_reward] = ie.query({reward_col: value_reward})[action_col]
                        except Exception as e_reward:
                            print(
                                f"Exception occurred while querying by reward_col for value {value_reward}: {e_reward}")

                    # Reverse the dictionary to get action_reward_values if reward_action_values was successful
                    if reward_action_values:
                        action_reward_values = _reverse_dict(reward_action_values)
                except Exception as e:
                    print(f"Exception occurred while creating reward_action_values: {e}")

        return action_reward_values

    for feature, value in observation.items():
        try:
            unique_values_feature = unique_values_df[feature]
            dict_set_probs = {}

            if value in unique_values_feature:
                for key in unique_values_feature:
                    dict_set_probs[key] = 1.0 if key == value else 0.0
            else:
                nearest_key = _find_nearest(unique_values_feature, value)
                for key in unique_values_feature:
                    dict_set_probs[key] = 1.0 if key == nearest_key else 0.0

            ie.do_intervention(feature, dict_set_probs)
            # print(f'do({feature} = {value})')
        except Exception as e:
            # print(f"Error during intervention on {feature} with value {value}: {str(e)}")
            pass

    action_reward_values = _create_action_reward_values()

    for feature, value in observation.items():
        try:
            ie.reset_do(feature)
        except Exception as e:
            print(f"Error during reset intervention on {feature} with value {value}: {str(e)}")

    return action_reward_values"""


class SingleCausalInference:
    def __init__(self, df: pd.DataFrame, causal_graph: nx.DiGraph, dict_ready_cbn: Dict = None):
        self.df = df
        self.causal_graph = causal_graph
        self.features = self.causal_graph.nodes

        if dict_ready_cbn is None:
            self.cbn = BayesianNetwork()
            self.cbn.add_edges_from(ebunch=self.causal_graph.edges())
            self.cbn.fit(self.df, estimator=MaximumLikelihoodEstimator)
        else:
            self.cbn = dict_to_bn(dict_ready_cbn)

        assert self.cbn.check_model()

        self.ci = CausalInference(self.cbn)

    def return_cbn(self) -> BayesianNetwork:
        return self.cbn

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
