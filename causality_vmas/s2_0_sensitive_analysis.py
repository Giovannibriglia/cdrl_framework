import os
from typing import List, Dict, Tuple
import json
import numpy as np
import pandas as pd
import multiprocessing

from causality_vmas import abs_path_causality_vmas, LABEL_approximation_parameters, LABEL_dataframe_approximated, \
    LABEL_ciq_results, LABEL_dir_storing_dict_and_info, LABEL_scores_binary, LABEL_scores_distance, LABEL_dict_causality
from causality_vmas.s_2_1_causality_informativeness_quantification import CausalityInformativenessQuantification
from causality_vmas.utils import my_approximation, save_file_incrementally, save_json_incrementally

from path_repo import GLOBAL_PATH_REPO


class SensitiveAnalysis:
    def __init__(self, df: pd.DataFrame, task_name: str):
        self.df_original = df
        self.task_name = task_name
        self.dir_save = f'./{LABEL_dir_storing_dict_and_info}_{self.task_name}'

        self.results = None

    def _compute_df_approximations(self) -> List[Dict]:
        return my_approximation(self.df_original, self.task_name)

    def _compute_and_save_single_ciq(self, single_dict_approx) -> Dict:
        approximation_dict = {k: v for k, v in single_dict_approx.items() if k != LABEL_dataframe_approximated}
        df_approx = single_dict_approx[LABEL_dataframe_approximated]

        for col in df_approx.columns.to_list():
            if len(df_approx[col].unique()) > single_dict_approx['n_bins']:
                print('*** Discretization problem *** ')
        try:
            ciq = CausalityInformativenessQuantification(df_approx, 'reward')
            res_ciq, res_causality = ciq.evaluate()
        except:
            print(f'{single_dict_approx} not suitable for causality informativeness quantification')
            res_ciq = {}
            res_causality = {}
        single_res = {LABEL_approximation_parameters: approximation_dict,
                      LABEL_ciq_results: res_ciq,
                      LABEL_dict_causality: res_causality}
        self._store_results(single_res, df_approx)
        print(f'results computed for {approximation_dict} approximation')
        return single_res

    def computing_CIQs(self) -> Tuple[List[Dict], str]:
        list_dict_approx = self._compute_df_approximations()
        print('approximations done')
        if len(list_dict_approx) > 1:
            if multiprocessing.get_start_method(allow_none=True) is None:
                multiprocessing.set_start_method('spawn')

            with multiprocessing.Pool(int(multiprocessing.cpu_count()*0.66)) as pool:
                self.results = pool.map(self._compute_and_save_single_ciq, list_dict_approx)
        else:
            self.results = [self._compute_and_save_single_ciq(list_dict_approx[0])]

        return self.results, self.dir_save

    def _store_results(self, dict_to_store: Dict, df_to_store: pd.DataFrame):
        save_file_incrementally(df_to_store, self.dir_save, prefix='df_', extension='pkl')
        save_json_incrementally(dict_to_store, self.dir_save, prefix='info_')


def main():
    task_name = 'navigation'

    df = pd.read_pickle(f'{GLOBAL_PATH_REPO}/causality_vmas/dataframes/{task_name}_mdp.pkl')
    agent0_columns = [col for col in df.columns if 'agent_0' in col]
    df = df.loc[:200001, agent0_columns]

    sensitive_analysis = SensitiveAnalysis(df, task_name)
    results, path_results = sensitive_analysis.computing_CIQs()


if __name__ == '__main__':
    main()
