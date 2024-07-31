import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict
import networkx as nx
import pandas as pd
from tqdm import tqdm

from causality_algos import SingleCausalInference
from causality_vmas import LABEL_reward_action_values, LABEL_dir_storing_dict_and_info
from causality_vmas.utils import list_to_graph, is_folder_empty


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
        """print('only 1000 rows')
        self.df = self.df.loc[:1000, :]"""

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


def main(task: str = 'navigation'):
    path_file = f'{LABEL_dir_storing_dict_and_info}_{task}/best'

    if not is_folder_empty(path_file):
        dataframe = pd.read_pickle(f'{path_file}/best_df.pkl')

        with open(f'{path_file}/best_causal_graph.json', 'r') as file:
            graph_list = json.load(file)
        causal_graph = list_to_graph(graph_list)

        with open(f'{path_file}/best_bn_params.json', 'r') as file:
            bn_dict = json.load(file)

        if (dataframe is not None and causal_graph is not None) or bn_dict is not None:
            offline_ci = OfflineCausalInferenceForRL(dataframe, causal_graph, bn_dict)
            ct = offline_ci.create_causal_table(show_progress=True)
            ct.to_pickle(f'{path_file}/causal_table.pkl')
            ct.to_excel('mario.xlsx')
        else:
            print('some files are empty')
    else:
        print('there are no best')


if __name__ == '__main__':
    main()
