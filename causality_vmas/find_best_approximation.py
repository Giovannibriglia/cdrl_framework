import os
from typing import List, Dict, Tuple
import json
import numpy as np
import pandas as pd
import multiprocessing

from causality_vmas import abs_path_causality_vmas, LABEL_approximation_parameters, LABEL_dataframe_approximated, \
    LABEL_ciq_results, LABEL_dir_storing_dict_and_info, LABEL_scores_binary, LABEL_scores_distance, LABEL_dict_causality
from causality_vmas.causality_informativeness_quantification import CausalityInformativenessQuantification
from causality_vmas.utils import my_approximation, save_file_incrementally, save_json_incrementally

from path_repo import GLOBAL_PATH_REPO


class ComputeCIQs:
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

        ciq = CausalityInformativenessQuantification(df_approx, 'reward')
        res_ciq, res_causality = ciq.evaluate()

        single_res = {LABEL_approximation_parameters: approximation_dict,
                      LABEL_ciq_results: res_ciq,
                      LABEL_dict_causality: res_causality}
        self._store_results(single_res, df_approx)
        print('results computed')
        return single_res

    def computing_CIQs(self) -> Tuple[List[Dict], str]:
        list_dict_approx = self._compute_df_approximations()
        print('approximations done')
        if multiprocessing.get_start_method(allow_none=True) is None:
            multiprocessing.set_start_method('spawn')

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            self.results = pool.map(self._compute_and_save_single_ciq, list_dict_approx)

        return self.results, self.dir_save

    def _store_results(self, dict_to_store: Dict, df_to_store: pd.DataFrame):
        save_file_incrementally(df_to_store, self.dir_save, prefix='df_', extension='pkl')
        save_json_incrementally(dict_to_store, self.dir_save, prefix='info_')


class BestApprox:
    def __init__(self, path_results: str, results: List[Dict] = None):
        self.path_results = path_results
        if results is None:
            self.all_results = self._extract_files()
        else:
            self.all_results = results

    def _extract_files(self) -> List[Dict]:
        # List all files in the folder
        all_files = os.listdir(self.path_results)

        # Filter for .json files
        json_files = [file for file in all_files if file.endswith('.json')]

        # Read and process each .json file
        json_data = []
        for file in json_files:
            file_path = os.path.join(self.path_results, file)
            with open(file_path, 'r') as f:
                json_data.append(json.load(f))

        return json_data

    def evaluate(self):
        for result in self.all_results:
            scores = result[f'{LABEL_ciq_results}']

            scores_binary = scores[f'{LABEL_scores_binary}']
            scores_distances = scores[f'{LABEL_scores_distance}']

            self._compute_binary_clf_metrics(scores_binary)
            self._compute_distance_metrics(scores_distances)

    def _compute_distance_metrics(self, results: List):
        # Convert the list to a numpy array for easier calculations
        errors = np.array(results)
        accuracy = np.sum(errors == 0) / len(errors)
        mae = np.mean(np.abs(errors))
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        median_absolute_error = np.median(np.abs(errors))
        max_error = np.max(np.abs(errors))
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Mean Absolute Error (MAE): {mae:.4f}')
        print(f'Mean Squared Error (MSE): {mse:.4f}')
        print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
        print(f'Median Absolute Error: {median_absolute_error:.4f}')
        print(f'Max Error: {max_error:.4f}')

    def _compute_binary_clf_metrics(self, results: List):
        mean = np.mean(results)
        std = np.std(results)
        print(f'Mean: {mean} \u00B1 Std {std}')


def main():
    task_name = 'navigation'

    df = pd.read_pickle(f'{GLOBAL_PATH_REPO}/causality_vmas/dataframes/navigation_mdp.pkl')
    agent0_columns = [col for col in df.columns if 'agent_0' in col]
    df = df.loc[:200001, agent0_columns]

    sensitive_analysis = ComputeCIQs(df, task_name)
    results, path_results = sensitive_analysis.computing_CIQs()

    # best_approx = BestApprox(path_results)
    # best_approx.evaluate()


if __name__ == '__main__':
    main()
