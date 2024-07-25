from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from causality_vmas.causality_algos import CausalDiscovery, SingleCausalInference
from causality_vmas.utils import get_df_boundaries, constraints_causal_graph, discretize_dataframe, bn_to_dict, \
    graph_to_list, _navigation_approximation
from causality_vmas import abs_path_causality_vmas, LABEL_scores_distance, LABEL_scores_binary, LABEL_causal_graph, \
    LABEL_bn_dict

show_progress_cd = True


class CausalityInformativenessQuantification:
    def __init__(self, df: pd.DataFrame, target_feature: str = 'action', cd_algo: str = 'PC'):
        self.df = df
        self.cd_algo = cd_algo

        self.causal_graph = None

        self.action_col_name = str([s for s in df.columns.to_list() if 'action' in s][0])
        if not self.action_col_name:
            raise ValueError('action column did not find')
        self.reward_col_name = str([s for s in df.columns.to_list() if 'reward' in s][0])
        if not self.reward_col_name:
            raise ValueError('reward column did not find')

        if target_feature == 'action':
            self.target_feature = self.action_col_name
        elif target_feature == 'reward':
            self.target_feature = self.reward_col_name
        else:
            raise ValueError('target feature is not correct')

    def evaluate(self) -> Tuple[Dict, Dict]:
        self.cd_process()
        res_score, res_causal = self.ci_assessment(show_progress=True)

        return res_score, res_causal

    def cd_process(self):
        cd = CausalDiscovery(self.df)
        cd.training(cd_algo=self.cd_algo, show_progress=show_progress_cd)
        self.causal_graph = cd.return_causal_graph()
        self.causal_graph = constraints_causal_graph(self.causal_graph)

    def ci_assessment(self, show_progress: bool = False) -> Tuple[Dict, Dict]:
        single_ci = SingleCausalInference(self.df, self.causal_graph)

        cbn = single_ci.return_cbn()
        cbn_in_dict = bn_to_dict(cbn)

        causality_dict = {LABEL_causal_graph: graph_to_list(self.causal_graph), LABEL_bn_dict: cbn_in_dict}
        res_score_dict = {LABEL_scores_distance: [], LABEL_scores_binary: []}

        selected_columns = [s for s in self.df.columns.to_list() if s != self.target_feature]

        for_cycle = tqdm(self.df.iterrows(), f'Inferring on {len(self.df)}') if show_progress else self.df.iterrows()
        for index, row in for_cycle:
            input_ci = row[selected_columns].to_dict()
            try:
                output_ci = single_ci.infer(input_ci, self.target_feature)

                value_target = row[self.target_feature]
                value_pred = max(output_ci, key=output_ci.get)
                value_pred = type(value_target)(value_pred)
                res_score_dict[LABEL_scores_distance].append(value_pred - value_target)
                res_score_dict[LABEL_scores_binary].append(1 if value_target == value_pred else 0)
            except Exception as e:
                print(e)
                res_score_dict[LABEL_scores_distance].append(-np.inf)
                res_score_dict[LABEL_scores_binary].append(0)

        return res_score_dict, causality_dict


if __name__ == '__main__':
    df = pd.read_pickle(f'{abs_path_causality_vmas}/causality_vmas/dataframes/navigation_mdp.pkl')

    agent0_columns = [col for col in df.columns if 'agent_0' in col]
    df = df.loc[:200001, agent0_columns]
    dict_approx = _navigation_approximation((df, 10, 2, 100000))
    df = dict_approx['new_df']
    # get_df_boundaries(df)

    ciq = CausalityInformativenessQuantification(df)
    print(ciq.evaluate())
