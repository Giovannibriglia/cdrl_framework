from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict
import networkx as nx
import pandas as pd
from tqdm import tqdm

from causality_algos import SingleCausalInference
from causality_vmas import LABEL_bn_dict, LABEL_reward_action_values


# causal table store in "causal_knowledge for rl
class OfflineCausalInferenceForRL:
    def __init__(self, df: pd.DataFrame, causal_graph: nx.DiGraph, bn_info: Dict = None):
        self.df = df
        self.causal_graph = causal_graph

        self.ci = SingleCausalInference(df, causal_graph, bn_info)

        self.reward_variable = [s for s in df.columns.to_list() if 'reward' in s][0]
        self.reward_values = self.df[self.reward_variable].value_counts().to_list()
        self.action_variable = [s for s in df.columns.to_list() if 'action' in s][0]

    def single_query(self, obs: Dict) -> Dict:
        reward_actions_values = {}

        # Query the distribution of actions for each reward
        for reward_value in self.reward_values:
            evidence = obs.copy()
            evidence.update({f'{self.reward_variable}': reward_value})
            # do "obs" | evidence "reward"+"obs" | look at "action"
            action_distribution = self.ci.infer(obs, self.action_variable, evidence)
            print(f"Distribution of actions given observation {obs} and reward {reward_value}: {action_distribution}")
            reward_actions_values[reward_value] = action_distribution

        return reward_actions_values

    """    
    def create_causal_table(self, show_progress: bool = False) -> pd.DataFrame:
        rows_causal_table = []

        selected_columns = [s for s in self.df.columns.to_list() if s != self.reward_variable and s != self.action_variable]

        for_cycle = tqdm(self.df.iterrows(), f'Inferring on {len(self.df)} for causal table') if show_progress else self.df.iterrows()
        for index, row in for_cycle:
            input_ci = row[selected_columns].to_dict()
            try:
                reward_action_values = self.single_query(input_ci)
            except Exception as e:
                print(e)
                reward_action_values = {}

            row = input_ci.copy()
            row[f'{LABEL_reward_action_values}'] = reward_action_values

            rows_causal_table.append(row)

        causal_table = pd.DataFrame(rows_causal_table)
        
        return causal_table"""

    @staticmethod
    def _process_row(args):
        index, row, selected_columns, single_query = args
        input_ci = row[selected_columns].to_dict()
        try:
            reward_action_values = single_query(input_ci)
        except Exception as e:
            print(e)
            reward_action_values = {}

        row_result = input_ci.copy()
        row_result[f'{LABEL_reward_action_values}'] = reward_action_values
        return row_result

    def create_causal_table(self, show_progress: bool = False) -> pd.DataFrame:
        rows_causal_table = []

        selected_columns = [s for s in self.df.columns.to_list() if
                            s != self.reward_variable and s != self.action_variable]

        tasks = [(index, row, selected_columns, self.single_query) for index, row in self.df.iterrows()]

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._process_row, task) for task in tasks]
            if show_progress:
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc=f'Inferring on {len(self.df)} for causal table'):
                    rows_causal_table.append(future.result())
            else:
                for future in as_completed(futures):
                    rows_causal_table.append(future.result())

        causal_table = pd.DataFrame(rows_causal_table)

        return causal_table


"""def inference_function(observation: Dict, ie: InferenceEngine, reward_col: str,
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

if __name__ == '__main__':
    dataframe = None
    graph = None
    bn_dict = None

    offline_ci = OfflineCausalInferenceForRL(dataframe, graph, bn_dict)
    ct = offline_ci.create_causal_table()
    print(ct)
