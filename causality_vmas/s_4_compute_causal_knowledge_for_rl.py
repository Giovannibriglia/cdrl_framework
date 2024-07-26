from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict
import networkx as nx
import pandas as pd
from tqdm import tqdm
import json

from causality_algos import SingleCausalInference
from causality_vmas import LABEL_bn_dict, LABEL_reward_action_values, LABEL_dict_causality, LABEL_causal_graph
from causality_vmas.utils import list_to_graph


class OfflineCausalInferenceForRL:
    def __init__(self, df: pd.DataFrame, causal_graph: nx.DiGraph, bn_info: Dict = None):
        self.df = df
        self.causal_graph = causal_graph

        self.ci = SingleCausalInference(df, causal_graph, bn_info)

        self.reward_variable = [s for s in df.columns.to_list() if 'reward' in s][0]
        self.reward_values = self.df[self.reward_variable].unique().tolist()
        self.action_variable = [s for s in df.columns.to_list() if 'action' in s][0]

    def single_query(self, obs: Dict) -> Dict:
        reward_actions_values = {}

        # Query the distribution of actions for each reward
        for reward_value in self.reward_values:
            evidence = obs.copy()
            evidence.update({f'{self.reward_variable}': reward_value})
            # do "obs" | evidence "reward"+"obs" | look at "action"
            self.check_values_in_states(self.ci.cbn.states, obs, evidence)
            action_distribution = self.ci.infer(obs, self.action_variable, evidence)
            reward_actions_values[reward_value] = action_distribution

        return reward_actions_values

    @staticmethod
    def check_values_in_states(known_states, observation, evidence):
        not_in_observation = {}
        not_in_evidence = {}

        for state, values in known_states.items():
            obs_value = observation.get(state, None)
            evid_value = evidence.get(state, None)

            if obs_value is not None and obs_value not in values:
                print(state)
                print(values)
                print(obs_value)
                not_in_observation[state] = obs_value

            if evid_value is not None and evid_value not in values:
                print(state)
                print(values)
                print(evid_value)
                not_in_evidence[state] = evid_value

        if not_in_observation != {}:
            print("Values not in observation: ", not_in_observation)

        if not_in_evidence != {}:
            print("\nValues not in evidence: ", not_in_evidence)

    @staticmethod
    def _process_row(args):
        index, row, selected_columns, single_query = args
        input_ci = row[selected_columns].to_dict()
        reward_action_values = single_query(input_ci)

        row_result = input_ci.copy()
        row_result[f'{LABEL_reward_action_values}'] = reward_action_values
        return row_result

    def create_causal_table(self, show_progress: bool = False) -> pd.DataFrame:
        rows_causal_table = []
        print('only 1000 rows')
        self.df = self.df.loc[:1000, :]

        selected_columns = [s for s in self.df.columns.to_list() if
                            s != self.reward_variable and s != self.action_variable]

        tasks = [(index, row, selected_columns, self.single_query) for index, row in self.df.iterrows()]

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._process_row, task) for task in tasks]
            if show_progress:
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc=f'Inferring causal knowledge...'):
                    rows_causal_table.append(future.result())
            else:
                for future in as_completed(futures):
                    rows_causal_table.append(future.result())

        causal_table = pd.DataFrame(rows_causal_table)

        return causal_table


if __name__ == '__main__':
    path_file = './df_approx_and_info_navigation'

    dataframe = pd.read_pickle(f'{path_file}/df_4.pkl')

    file_info_json = f'{path_file}/info_4.json'

    with open(file_info_json, 'r') as file:
        all_info = json.load(file)

    graph_list = all_info[LABEL_dict_causality][LABEL_causal_graph]
    graph = list_to_graph(graph_list)
    bn_dict = all_info[LABEL_dict_causality][LABEL_bn_dict]

    offline_ci = OfflineCausalInferenceForRL(dataframe, graph, bn_dict)
    ct = offline_ci.create_causal_table(show_progress=True)
    ct.to_excel('mario.xlsx')
