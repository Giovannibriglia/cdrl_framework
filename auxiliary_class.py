from typing import Dict, List

import numpy as np
import pandas as pd
import json

from gymnasium.spaces import Discrete, Box

from causality_vmas import LABEL_dir_storing_dict_and_info, LABEL_dict_causality, LABEL_causal_graph, LABEL_bn_dict, \
    LABEL_reward_action_values
from causality_vmas.causality_algos import SingleCausalInference
from causality_vmas.utils import list_to_graph


class CausalityDrivenActionSpaceFilter:
    def __init__(self, action_space_type, online: bool, path_previous_knowledge: bool):
        self.original_action_space_type = action_space_type
        self.dynamic_action_space = DynamicActionSpace(action_space_type)
        self.online = online

        if path_previous_knowledge:
            self.path_knowledge = path_previous_knowledge
            self.df_causality = pd.read_pickle(f'{self.path_knowledge}/best_df.pkl')

            if self.online:
                with open(f'{self.path_knowledge}(best_info.json', 'r') as f:
                    info_best = json.load(f)

                causal_graph_list = info_best[f'{LABEL_dict_causality}'][f'{LABEL_causal_graph}']
                causal_graph = list_to_graph(causal_graph_list)
                bn_info = info_best[f'{LABEL_dict_causality}'][f'{LABEL_bn_dict}']

                self.ci = SingleCausalInference(self.df_causality, causal_graph, bn_info)
            else:
                self.causal_table = pd.read_pickle(f'{self.path_knowledge}/causal_table.pkl')

        else:
            raise NotImplementedError('have no previous knowledge is not still implemented')

    def action_space_filter(self, obs: Dict):
        reward_action_values = self._get_reward_action_values(obs)
        action_reward_scores = self.weighted_action_filter(reward_action_values)
        actions_to_discard = self.select_actions_to_remove(action_reward_scores)

        self.dynamic_action_space.constraint(actions_to_discard)
        # TODO: quale outuput?
        return self.dynamic_action_space

    def _get_reward_action_values(self, obs: Dict) -> Dict:
        if self.online:
            raise NotImplementedError
        else:
            filtered_df = self.causal_table.copy()
            for feature, value in obs.items():
                filtered_df = filtered_df[filtered_df[feature] == value]

            reward_action_values = filtered_df[LABEL_reward_action_values]

        return reward_action_values

    @staticmethod
    def weighted_action_filter(input_dict: Dict) -> Dict:
        def rescale_reward(past_value: float, old_min: float, old_max: float, new_min=0, new_max=1):
            new_value = new_min + ((past_value - old_min) / (old_max - old_min)) * (new_max - new_min)
            return new_value

        # Extract all reward values to find min and max for rescaling
        all_rewards = [reward for rewards_dict in input_dict.values() for reward in rewards_dict.keys()]
        old_min = float(min(all_rewards))
        old_max = float(max(all_rewards))

        averaged_mean_dict = {}
        for key_action in input_dict.values():
            for action, prob in key_action.items():
                if action not in averaged_mean_dict:
                    averaged_mean_dict[action] = 0

                # Here we assume that the reward_value is the key and we are calculating weighted sum
                reward_value = float(action)
                averaged_mean_dict[action] += prob * rescale_reward(reward_value, old_min, old_max)

        # Averaging the summed rewards based on the number of reward entries
        num_reward_entries = len(input_dict)
        for action in averaged_mean_dict:
            averaged_mean_dict[action] = round(averaged_mean_dict[action] / num_reward_entries, 3)

        return averaged_mean_dict

    @staticmethod
    def select_actions_to_remove(action_reward_scores: Dict) -> List:
        # Extract the values and calculate the 25th percentile
        values = list(action_reward_scores.values())
        percentile_25 = np.percentile(values, 25)

        return [key for key, value in action_reward_scores if value <= percentile_25]
