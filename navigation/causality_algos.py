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
        self.dir_save = f'{dir_name}/{self.env_name}'
        os.makedirs(self.dir_save, exist_ok=True)
        random.seed(42)
        np.random.seed(42)

        self.add_data(df)

    def add_data(self, df: pd.DataFrame):
        if self.df is None:
            self.df = df
        else:
            self.df = pd.concat([self.df, df]).reset_index(drop=True)

        reward_col = [s for s in self.df.columns.to_list() if 'reward' in s][0]

        self.df.to_pickle(f'{self.dir_save}/df_{len(self.df)}.pkl')
        # plot_reward(self.df[reward_col], reward_col)
        self.features_names = self.df.columns.to_list()

        for col in self.df.columns:
            self.df[str(col)] = self.df[str(col)].astype(str).str.replace(',', '').astype(float)

    def training(self, cd_algo: str = 'mario'):
        if cd_algo == 'pc':
            causal_relationships = self.use_pc()
            sm = StructureModel()
            sm.add_edges_from([(node1, node2) for node1, node2 in causal_relationships])
            self.causal_graph = sm

            self._plot_and_save_graph(self.causal_graph, True)

            if_causal_graph_DAG = nx.is_directed_acyclic_graph(self.causal_graph)
            if not if_causal_graph_DAG:
                print('**** Causal graph is not a DAG ****')
        else:
            print(f'\n{self.env_name} - structuring model through NOTEARS... {len(self.df)} timesteps')
            self.notears_graph = from_pandas(self.df, max_iter=2000, use_gpu=True)
            self.notears_graph.remove_edges_below_threshold(0.2)
            largest_component = max(nx.weakly_connected_components(self.notears_graph), key=len)
            self.notears_graph = self.notears_graph.subgraph(largest_component).copy()
            self._plot_and_save_graph(self.notears_graph, False)

            if nx.number_weakly_connected_components(self.notears_graph) == 1 and nx.is_directed_acyclic_graph(
                    self.notears_graph):

                # print('do-calculus-1...')
                # assessment of the no-tears graph
                causal_relationships, _, _ = self._causality_assessment(self.notears_graph, self.df)

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

    def _causality_assessment(self, graph: StructureModel, df: pd.DataFrame) -> Tuple[List[Tuple], List, List]:
        """ Given an edge, do-calculus on each direction once each other (not simultaneously) """
        # print('bayesian network definition...')
        bn = BayesianNetwork(graph)
        bn = bn.fit_node_states_and_cpds(df)

        bad_nodes = [node for node in bn.nodes if not re.match("^[0-9a-zA-Z_]+$", node)]
        if bad_nodes:
            print('Bad nodes: ', bad_nodes)

        ie = InferenceEngine(bn)

        # Initial assumption: all nodes are independent until proven dependent
        independent_vars = set(graph.nodes)
        dependent_vars = set()
        causal_relationships = []

        # Precompute unique values for all nodes
        unique_values = {node: df[node].unique() for node in graph.nodes}

        # Initial query to get the baseline distributions
        before = ie.query()

        # Iterate over each node in the graph
        pbar = tqdm(graph.nodes, desc=f'{self.env_name} nodes')
        for node in pbar:
            # for node in graph.nodes:
            connected_nodes = list(self.notears_graph.neighbors(node))
            change_detected = False

            # Perform interventions on the node and observe changes in all connected nodes
            for value in unique_values[node]:
                try:
                    ie.do_intervention(node, int(value))
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
                                # print(f"Link removed: {node} -> {conn_node}")
                            change_detected = True
                            causal_relationships.append((node, conn_node))  # Ensure this is a 2-tuple

                    # Also check the intervened node itself
                    best_key_before, max_value_before = max(before[node].items(), key=lambda x: x[1])
                    best_key_after, max_value_after = max(after[node].items(), key=lambda x: x[1])
                    uniform_probability_value = round(1 / len(after[node]), 8)

                    if max_value_after > uniform_probability_value and best_key_after != best_key_before:
                        change_detected = True

                except Exception as e:
                    # Log the error
                    # print(f"Error during intervention on {node} with value {value}: {str(e)}")
                    pass

            if change_detected:
                dependent_vars.add(node)  # Mark as dependent
                if node in independent_vars:
                    independent_vars.remove(node)  # Remove from independents
                    # print(f"Link removed: {node}")

        return causal_relationships, list(independent_vars), list(dependent_vars)

    def return_causal_graph(self):
        if self.causal_graph is not None:
            structure_to_return = [(x[0], x[1]) for x in self.causal_graph.edges]
            return structure_to_return
        else:
            return self.notears_graph

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

    def use_pc(self):
        data = self.df.to_numpy()
        labels = [f'{col}' for i, col in enumerate(self.features_names)]
        cg = pc(data, show_progress=False)

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
    def __init__(self, df: pd.DataFrame, causal_graph: StructureModel, causal_table: pd.DataFrame = None):

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

        self.add_data(df, causal_graph)

    def add_data(self, new_df: pd.DataFrame, new_graph: StructureModel):
        if self.df is None:
            self.df = new_df
        else:
            self.df = pd.concat([self.df, new_df]).reset_index(drop=True)

        self.reward_column = [s for s in self.df.columns.to_list() if 'reward' in s][0]
        self.action_column = [s for s in self.df.columns.to_list() if 'action' in s][0]

        self.action_space_size = int(max(self.df[self.action_column].unique()))
        self.possible_reward_values = self.df[self.reward_column].unique()

        for col in self.df.columns:
            self.df[str(col)] = self.df[str(col)].astype(str).str.replace(',', '').astype(float)

        self.causal_graph = new_graph
        try:
            self.bn = BayesianNetwork(self.causal_graph)
            self.bn = self.bn.fit_node_states_and_cpds(self.df)

            bad_nodes = [node for node in self.bn.nodes if not re.match("^[0-9a-zA-Z_]+$", node)]
            if bad_nodes:
                print('Bad nodes: ', bad_nodes)

            self.ie = InferenceEngine(self.bn)
        except:
            self.bn = None
            self.ie = None

    def get_rewards_actions_values(self, observation: Dict, online: bool) -> dict:
        print(observation)
        if self.bn is not None and self.ie is not None:
            print('online bn')
            if online:
                reward_action_values = inference_function(observation, self.ie, self.possible_reward_values,
                                                          self.reward_column, self.action_column)

                return reward_action_values
            else:
                filtered_df = self.causal_table.copy()
                for feature, value in observation.items():
                    filtered_df = filtered_df[filtered_df[feature] == value]

                if not filtered_df.empty:
                    reward_action_values = filtered_df['reward_action_values'].values[0]
                else:
                    print('No reward-action values for this observation are available')
                    reward_action_values = {}

                return reward_action_values
        else:
            uniform_prob = round(1 / self.action_space_size, 5)
            return {key: uniform_prob for key in range(self.action_space_size)}

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
    possible_reward_values = df[col_reward].unique()
    for comb in pbar:
        reward_action_values = inference_function(comb, ie, possible_reward_values, col_reward, col_action)
        new_row = comb.copy()
        new_row[COL_REWARD_ACTION_VALUES] = reward_action_values
        rows.append(new_row)
    return rows


def inference_function(observation: Dict, ie: InferenceEngine, possible_reward_values: List, reward_col: str,
                       action_col: str):
    reward_feature = reward_col
    action_feature = action_col
    reward_action_values = {}

    for feature, value in observation.items():
        try:
            ie.do_intervention(feature, int(value))
        except Exception as e:
            pass

    for value_reward in possible_reward_values:
        probabilities_action_feature = ie.query({f'{reward_feature}': value_reward})[action_feature]
        # best_action_value, best_action_prob = max(probabilities_action_feature.items(), key=lambda x: x[1])

        reward_action_values[value_reward] = probabilities_action_feature  # {reward_value: dict_action{value:prob}, ..}

    for feature, value in observation.items():
        try:
            ie.reset_do(feature)
        except:
            pass

    return reward_action_values


def plot_reward(values, name):
    fig = plt.figure(dpi=500)
    plt.title(f'{name} - {len(values)}')
    x = np.arange(0, len(values), 1)
    plt.plot(x, values)
    plt.show()
