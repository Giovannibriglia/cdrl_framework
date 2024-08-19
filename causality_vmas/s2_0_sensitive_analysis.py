import logging
from typing import List, Dict
import json
import pandas as pd

from causality_vmas import LABEL_approximation_parameters, LABEL_dataframe_approximated, \
    LABEL_ciq_scores, LABEL_dir_storing_dict_and_info, LABEL_discrete_intervals, LABEL_target_feature_analysis, \
    LABEL_info_task, LABEL_grouped_features
from causality_vmas.s2_1_causality_informativeness_quantification import CausalityInformativenessQuantification
from causality_vmas.utils import my_approximation, save_file_incrementally, save_json_incrementally, graph_to_list


class SensitiveAnalysis:
    def __init__(self, df: pd.DataFrame, task_name: str, info_task: Dict, target_feature: str = 'agent_0_reward'):
        self.df_original = df
        self.task_name = task_name
        self.info_task = info_task
        self.target_feature = target_feature
        self.dir_save = f'./{LABEL_dir_storing_dict_and_info}_{self.task_name}'
        self.results = None

    def _compute_df_approximations(self) -> List[Dict]:
        logging.info('Computing approximations')
        list_dict_approx = my_approximation(self.df_original, self.task_name)
        return list_dict_approx

    def _compute_and_save_single_ciq(self, single_dict_approx):
        df_approx = single_dict_approx[LABEL_dataframe_approximated]
        params_approximation = single_dict_approx[LABEL_approximation_parameters]
        discrete_intervals = single_dict_approx[LABEL_discrete_intervals]
        grouped_features = single_dict_approx[LABEL_grouped_features]

        kwargs = {LABEL_target_feature_analysis: self.target_feature,
                  LABEL_grouped_features: grouped_features}

        save_file_incrementally(df_approx, self.dir_save, 'df_', 'pkl')

        ciq = CausalityInformativenessQuantification(self.task_name, df_approx, self.df_original, **kwargs)
        dict_scores, causal_graph, dict_bn_info = ciq.evaluate()

        if not (dict_scores is None or causal_graph is None or dict_bn_info is None):
            list_causal_graph = graph_to_list(causal_graph)
            save_json_incrementally(list_causal_graph, self.dir_save, "causal_graph_")
            save_json_incrementally(dict_bn_info, self.dir_save, "bn_params_")
            save_json_incrementally(params_approximation, self.dir_save, "approx_params_")

            others = {LABEL_discrete_intervals: discrete_intervals,
                      LABEL_info_task: self.info_task,
                      LABEL_grouped_features: grouped_features
                      }
            save_json_incrementally(others, self.dir_save, 'others_')

            dict_scores_evaluation = {LABEL_target_feature_analysis: self.target_feature,
                                      LABEL_ciq_scores: dict_scores}
            save_json_incrementally(dict_scores_evaluation, self.dir_save, 'scores_')

            logging.info(f'Results computed and saved for {params_approximation} approximation')
        else:
            logging.error(f'No items to save')

    def computing_CIQs(self) -> str:
        list_dict_approx = self._compute_df_approximations()

        for dict_approx in list_dict_approx:
            self._compute_and_save_single_ciq(dict_approx)

        return self.dir_save


def main(task_name: str):

    df = pd.read_pickle(f'./dataframes/df_{task_name}_pomdp_discrete_actions_0.pkl')
    agent0_columns = [col for col in df.columns if 'agent_0' in col]
    df = df.loc[:10001, agent0_columns]

    with open(f'./dataframes/info_{task_name}_pomdp_discrete_actions_0.json', 'r') as file:
        info_task = json.load(file)

    sensitive_analysis = SensitiveAnalysis(df, task_name, info_task)
    path_results = sensitive_analysis.computing_CIQs()


if __name__ == '__main__':
    main('navigation')
