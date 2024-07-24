import os
from typing import List, Dict
import pandas as pd
import json
import multiprocessing

from causality_vmas import abs_path_causality_vmas, LABEL_approximation_parameters, LABEL_dataframe_approximated, \
    LABEL_ciq_results, LABEL_dir_storing_dict_and_info
from causality_vmas.causality_informativeness_quantification import CausalityInformativenessQuantification
from causality_vmas.utils import my_approximation

from path_repo import GLOBAL_PATH_REPO


class ComputeCIQs:
    def __init__(self, df: pd.DataFrame, task_name: str):
        self.df_original = df
        self.task_name = task_name

        self.results = None

    def _compute_df_approximations(self) -> List[Dict]:
        return my_approximation(self.df_original)

    @staticmethod
    def _compute_single_ciq(single_dict_approx) -> Dict:
        approximation_dict = {k: v for k, v in single_dict_approx.items() if k != LABEL_dataframe_approximated}
        df_approx = single_dict_approx[LABEL_dataframe_approximated]

        ciq = CausalityInformativenessQuantification(df_approx, 'reward')
        res_ciq = ciq.evaluate()

        single_res = {LABEL_approximation_parameters: approximation_dict,
                      LABEL_ciq_results: res_ciq,
                      LABEL_dataframe_approximated: df_approx}
        return single_res

    def computing_CIQs(self):
        list_dict_approx = self._compute_df_approximations()

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            self.results = pool.map(self._compute_single_ciq, list_dict_approx)

        self._store_results()

        return self.results

    def _store_results(self):
        dict_res = self.results

        dir_save_df = f'./{LABEL_dir_storing_dict_and_info}_{self.task_name}'
        os.makedirs(dir_save_df, exist_ok=True)
        for idx, dict_to_store in enumerate(dict_res):
            df_to_store = dict_to_store.pop(LABEL_dataframe_approximated)

            df_to_store.to_pickle(f"{dir_save_df}/df_{idx}.pkl")

            with open(f"{dir_save_df}/info_{idx}.json", 'w') as json_file:
                json.dump(dict_to_store, json_file)


class FindBestSimplification:
    def __init__(self, results: List[Dict]):
        self.all_results = results

    def evaluate(self):
        for result in self.all_results:
            print(result)


def main():
    df = pd.read_pickle(f'{GLOBAL_PATH_REPO}/causality_vmas/dataframes/navigation_mdp.pkl')
    agent0_columns = [col for col in df.columns if 'agent_0' in col]
    df = df.loc[:200001, agent0_columns]

    sensitive_analysis = ComputeCIQs(df, 'navigation')
    sensitive_analysis.computing_CIQs()


if __name__ == '__main__':
    main()
