from typing import List, Dict
import pandas as pd
import logging

from causality_vmas import LABEL_approximation_parameters, LABEL_dataframe_approximated, \
    LABEL_ciq_scores, LABEL_dir_storing_dict_and_info, LABEL_discrete_intervals, LABEL_target_feature_analysis
from causality_vmas.s2_1_causality_informativeness_quantification import CausalityInformativenessQuantification
from causality_vmas.utils import my_approximation, save_file_incrementally, save_json_incrementally, graph_to_list

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(processName)s - %(message)s')


class SensitiveAnalysis:
    def __init__(self, df: pd.DataFrame, task_name: str, target_feature: str = 'agent_0_reward'):
        self.df_original = df
        self.task_name = task_name
        self.target_feature = target_feature
        self.dir_save = f'./{LABEL_dir_storing_dict_and_info}_{self.task_name}'
        self.results = None

    def _compute_df_approximations(self) -> List[Dict]:
        logging.info('Computing approximations')
        list_dict_approx = my_approximation(self.df_original, self.task_name)
        logging.info('Approximations done')
        return list_dict_approx

    def _compute_and_save_single_ciq(self, single_dict_approx):
        df_approx = single_dict_approx[LABEL_dataframe_approximated]
        params_approximation = single_dict_approx[LABEL_approximation_parameters]
        discrete_intervals = single_dict_approx[LABEL_discrete_intervals]

        ciq = CausalityInformativenessQuantification(df_approx, self.target_feature)
        dict_scores, causal_graph, dict_bn_info = ciq.evaluate()

        if not (dict_scores is None or causal_graph is None or dict_bn_info is None):
            save_file_incrementally(df_approx, self.dir_save, 'df_', 'pkl')

            list_causal_graph = graph_to_list(causal_graph)
            save_json_incrementally(list_causal_graph, self.dir_save, "causal_graph_")
            save_json_incrementally(dict_bn_info, self.dir_save, "bn_params_")
            save_json_incrementally(params_approximation, self.dir_save, "approx_params_")

            others = {LABEL_discrete_intervals: discrete_intervals}
            save_json_incrementally(others, self.dir_save, 'others_')

            dict_scores_evaluation = {LABEL_target_feature_analysis: self.target_feature,
                                      LABEL_ciq_scores: dict_scores}
            save_json_incrementally(dict_scores_evaluation, self.dir_save, 'scores_')

            logging.info(f'Results computed and saved for {params_approximation} approximation')
        else:
            logging.error(f'No best found')

    """def computing_CIQs(self) -> Tuple[List[Dict], str]:
        MEMORY_USAGE = 0.5
    
        list_dict_approx = self._compute_df_approximations()

        total_memory = psutil.virtual_memory().total
        available_memory = psutil.virtual_memory().available
        memory_for_tasks = total_memory * MEMORY_USAGE
        num_tasks = len(list_dict_approx)
        estimated_memory_per_task = memory_for_tasks / num_tasks

        max_workers = min(int(available_memory // estimated_memory_per_task), multiprocessing.cpu_count() // 2)
        max_workers = max(1, max_workers)  # Ensure at least one worker is used

        if len(list_dict_approx) > 1:
            if multiprocessing.get_start_method(allow_none=True) is None:
                multiprocessing.set_start_method('spawn')

            logging.info(f'Starting {len(list_dict_approx)} processes with {max_workers} workers')
            with multiprocessing.Pool(max_workers) as pool:
                self.results = []
                results = []
                for single_dict_approx in list_dict_approx:
                    result = pool.apply_async(self._compute_and_save_single_ciq, args=(single_dict_approx,),
                                              callback=self._collect_result,
                                              error_callback=self._handle_error)
                    results.append(result)

                # Ensure all tasks are completed
                for result in results:
                    result.wait()

                gc.collect()  # Force garbage collection after all tasks are done
        else:
            logging.info(f'Starting 1 process with {max_workers} workers')
            self.results = [self._compute_and_save_single_ciq(list_dict_approx[0])]

        return self.results, self.dir_save
    
    def _handle_error(self, error):
        logging.error(f'Error in process: {error}')"""

    def computing_CIQs(self) -> str:
        list_dict_approx = self._compute_df_approximations()

        for dict_approx in list_dict_approx:
            self._compute_and_save_single_ciq(dict_approx)

        return self.dir_save


def main():
    task_name = 'navigation'

    df = pd.read_pickle(f'./dataframes/df_{task_name}_pomdp_discrete_actions_1.pkl')
    agent0_columns = [col for col in df.columns if 'agent_0' in col]
    df = df.loc[:10001, agent0_columns]

    sensitive_analysis = SensitiveAnalysis(df, task_name)
    path_results = sensitive_analysis.computing_CIQs()


if __name__ == '__main__':
    main()
