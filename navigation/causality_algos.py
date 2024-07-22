import itertools
import os
from typing import Tuple, List, Dict
import re
import networkx as nx
import pandas as pd
import random
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causalnex.inference import InferenceEngine
from causalnex.network import BayesianNetwork
from causalnex.structure import StructureModel
from causalnex.structure.pytorch import from_pandas
import json
import multiprocessing
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

FONT_SIZE_NODE_GRAPH = 7
ARROWS_SIZE_NODE_GRAPH = 30
NODE_SIZE_GRAPH = 1000

COL_REWARD_ACTION_VALUES = 'reward_action_values'


class CausalDiscovery:
    def __init__(self, df: pd.DataFrame = None, dir_name: str = None, env_name: str = None):

        self.features_names = None
        self.causal_graph = None
        self.notears_graph = None
        self.df = None
        self.env_name = env_name
        if dir_name is not None or env_name is not None:
            self.dir_save = f'{dir_name}/{self.env_name}'
            os.makedirs(self.dir_save, exist_ok=True)
        else:
            self.dir_save = None
        random.seed(42)
        np.random.seed(42)

        self.add_data(df)

    def add_data(self, df: pd.DataFrame):
        if self.df is None:
            self.df = df
        else:
            self.df = pd.concat([self.df, df]).reset_index(drop=True)

        reward_col = [s for s in self.df.columns.to_list() if 'reward' in s][0]

        if self.dir_save is not None:
            self.df.to_pickle(f'{self.dir_save}/df_{len(self.df)}.pkl')

        self.features_names = self.df.columns.to_list()

        for col in self.df.columns:
            self.df[str(col)] = self.df[str(col)].astype(str).str.replace(',', '').astype(float)

    def training(self, cd_algo: str = 'mario', show_progress: bool = False):
        if cd_algo == 'pc':
            causal_relationships = self.use_pc(show_progress)
            sm = StructureModel()
            sm.add_edges_from([(node1, node2) for node1, node2 in causal_relationships])
            self.causal_graph = sm

            self._plot_and_save_graph(self.causal_graph, True)

            if_causal_graph_DAG = nx.is_directed_acyclic_graph(self.causal_graph)
            if not if_causal_graph_DAG:
                print('**** Causal graph is not a DAG ****')
        elif cd_algo == 'notears':
            self.causal_graph = from_pandas(self.df, max_iter=5000, use_gpu=True)
            self.causal_graph.remove_edges_below_threshold(0.2)
            largest_component = max(nx.weakly_connected_components(self.causal_graph), key=len)
            self.causal_graph = self.causal_graph.subgraph(largest_component).copy()
            # self.notears_graph = self.generate_random_dag(self.features_names)
            self._plot_and_save_graph(self.causal_graph, True)

            if_causal_graph_DAG = nx.is_directed_acyclic_graph(self.causal_graph)
            if not if_causal_graph_DAG:
                print('**** Causal graph is not a DAG ****')

        else:
            print(f'\n{self.env_name} - structuring model through NOTEARS... {len(self.df)} timesteps')
            self.notears_graph = from_pandas(self.df, max_iter=5000, use_gpu=True)
            self.notears_graph.remove_edges_below_threshold(0.2)
            largest_component = max(nx.weakly_connected_components(self.notears_graph), key=len)
            self.notears_graph = self.notears_graph.subgraph(largest_component).copy()
            # self.notears_graph = self.generate_random_dag(self.features_names)
            self._plot_and_save_graph(self.notears_graph, False)

            if nx.number_weakly_connected_components(self.notears_graph) == 1 and nx.is_directed_acyclic_graph(
                    self.notears_graph):

                # print('do-calculus-1...')
                # assessment of the no-tears graph
                causal_relationships, _, _ = self._causality_assessment(self.notears_graph, self.df, show_progress)

                sm = StructureModel()
                sm.add_edges_from([(node1, node2) for node1, node2 in causal_relationships])
                self.causal_graph = sm

                self._plot_and_save_graph(self.causal_graph, True)

                if_causal_graph_DAG = nx.is_directed_acyclic_graph(self.causal_graph)
                if not if_causal_graph_DAG:
                    print('**** Causal graph is not a DAG ****')

            else:
                self.causal_graph = None
                print(
                    f'Number of graphs: {nx.number_weakly_connected_components(self.notears_graph)},'f' DAG: {nx.is_directed_acyclic_graph(self.notears_graph)}')

    def generate_random_dag(self, nodes):
        # Initialize the StructureModel
        sm = StructureModel()

        # Identify "action" and "reward" nodes
        action_nodes = [node for node in nodes if "action" in node]
        reward_nodes = [node for node in nodes if "reward" in node]
        other_nodes = [node for node in nodes if node not in action_nodes + reward_nodes]

        # Shuffle the non-action and non-reward nodes to create a random topological order
        random.shuffle(other_nodes)

        # Create a sorted list of nodes: action_nodes + other_nodes + reward_nodes
        sorted_nodes = action_nodes + other_nodes + reward_nodes

        # Ensure the graph is fully connected by adding a backbone
        for i in range(len(sorted_nodes) - 1):
            sm.add_edge(sorted_nodes[i], sorted_nodes[i + 1])

        # Connect action nodes to all other nodes
        for action_node in action_nodes:
            for node in sorted_nodes:
                if action_node != node:
                    sm.add_edge(action_node, node)

        # Connect all other nodes to reward nodes
        for node in sorted_nodes:
            if node not in reward_nodes:
                for reward_node in reward_nodes:
                    sm.add_edge(node, reward_node)

        # Add additional random edges to the DAG
        for i in range(len(sorted_nodes)):
            for j in range(i + 2, len(sorted_nodes)):  # Skip immediate next node to avoid redundant backbone edges
                if random.random() < 0.5:  # Randomly decide to add an edge
                    sm.add_edge(sorted_nodes[i], sorted_nodes[j])

        return sm

    def _causality_assessment(self, graph: StructureModel, df: pd.DataFrame, show_progress: bool = False) -> Tuple[List[Tuple], List, List]:
        print('Bayesian network definition...')
        bn = BayesianNetwork(graph)
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

        return causal_relationships, list(independent_vars), list(dependent_vars)

    def return_causal_graph(self) -> nx.DiGraph:
        if self.causal_graph is not None:
            structure_to_return_list = [(x[0], x[1]) for x in self.causal_graph.edges]
            structure_to_return = nx.DiGraph(structure_to_return_list)
            return structure_to_return
        else:
            structure_to_return_list = [(x[0], x[1]) for x in self.notears_graph.edges]
            structure_to_return = nx.DiGraph(structure_to_return_list)
            return structure_to_return

    def return_df(self):
        return self.df

    def _plot_and_save_graph(self, sm: StructureModel, if_causal: bool):

        import warnings
        warnings.filterwarnings("ignore")

        fig = plt.figure(dpi=1000)
        if if_causal:
            plt.title(f'Causal graph - {len(sm)} nodes - {len(sm.edges)} edges', fontsize=16)
        else:
            plt.title(f'NOTEARS graph - {len(sm)} nodes - {len(sm.edges)} edges', fontsize=16)

        nx.draw(sm, with_labels=True, font_size=FONT_SIZE_NODE_GRAPH,
                arrowsize=ARROWS_SIZE_NODE_GRAPH if if_causal else 0,
                arrows=if_causal,
                edge_color='orange', node_size=NODE_SIZE_GRAPH, font_weight='bold', node_color='skyblue',
                pos=nx.circular_layout(sm))

        structure_to_save = [(x[0], x[1]) for x in sm.edges]

        if self.dir_save is not None:
            if if_causal:
                plt.savefig(f'{self.dir_save}/causal_graph{len(self.df)}.png')

                with open(f'{self.dir_save}/causal_structure{len(self.df)}.json', 'w') as json_file:
                    json.dump(structure_to_save, json_file)
            else:
                plt.savefig(f'{self.dir_save}/notears_graph.png')

                with open(f'{self.dir_save}/notears_structure.json', 'w') as json_file:
                    json.dump(structure_to_save, json_file)

        # plt.show()
        plt.close(fig)

    def use_pc(self, show_progress: bool = False):

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
            stddev[stddev == 0] = np.inf  # Adjust as needed
        # Convert the cleaned data to a numpy array
        data_array = data.values

        labels = [f'{col}' for i, col in enumerate(self.features_names)]
        cg = pc(data_array, show_progress=show_progress)

        # Create a NetworkX graph
        G = nx.DiGraph()
        causal_relationships = []

        # Add nodes with proper labels
        for node in cg.G.get_nodes():
            node_name = node.get_name()
            node_index = int(node_name[1:]) - 1  # Assuming node names are in the form 'X1', 'X2', ...
            G.add_node(node_name, label=labels[node_index])

        # Add edges and collect causal relationships
        for edge in cg.G.get_graph_edges():
            src = edge.get_node1().get_name()
            dst = edge.get_node2().get_name()
            G.add_edge(src, dst)
            causal_relationships.append((labels[int(src[1:]) - 1], labels[int(dst[1:]) - 1]))

        return causal_relationships


class CausalInferenceForRL:
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
            self.bn = BayesianNetwork(self.causal_graph)
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
    ie = InferenceEngine(BayesianNetwork(causal_graph).fit_node_states_and_cpds(df))
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

    return action_reward_values


def plot_reward(values, name):
    fig = plt.figure(dpi=500)
    plt.title(f'{name} - {len(values)}')
    x = np.arange(0, len(values), 1)
    plt.plot(x, values)
    plt.show()
