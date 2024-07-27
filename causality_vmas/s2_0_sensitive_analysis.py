import gc
from typing import List, Dict, Tuple
import pandas as pd
import multiprocessing
import psutil
import logging

from causality_vmas import abs_path_causality_vmas, LABEL_approximation_parameters, LABEL_dataframe_approximated, \
    LABEL_ciq_results, LABEL_dir_storing_dict_and_info, LABEL_dict_causality
from causality_vmas.s_2_1_causality_informativeness_quantification import CausalityInformativenessQuantification
from causality_vmas.utils import my_approximation, save_file_incrementally, save_json_incrementally

from path_repo import GLOBAL_PATH_REPO

"""class SensitiveAnalysis:
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
        except Exception as e:
            print('\n', e)
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

            with multiprocessing.Pool(int(multiprocessing.cpu_count()*0.5)) as pool:
                self.results = pool.map(self._compute_and_save_single_ciq, list_dict_approx)
        else:
            self.results = [self._compute_and_save_single_ciq(list_dict_approx[0])]

        return self.results, self.dir_save

    def _store_results(self, dict_to_store: Dict, df_to_store: pd.DataFrame):
        save_file_incrementally(df_to_store, self.dir_save, prefix='df_', extension='pkl')
        save_json_incrementally(dict_to_store, self.dir_save, prefix='info_')"""

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(processName)s - %(message)s')


class SensitiveAnalysis:
    def __init__(self, df: pd.DataFrame, task_name: str):
        self.df_original = df
        self.task_name = task_name
        self.dir_save = f'./{LABEL_dir_storing_dict_and_info}_{self.task_name}'
        self.results = None

    def _compute_df_approximations(self) -> List[Dict]:
        logging.info('Computing approximations')
        list_dict_approx = my_approximation(self.df_original, self.task_name)
        logging.info('Approximations done')
        return list_dict_approx

    def _compute_and_save_single_ciq(self, single_dict_approx) -> Dict:
        approximation_dict = {k: v for k, v in single_dict_approx.items() if k != LABEL_dataframe_approximated}
        df_approx = single_dict_approx[LABEL_dataframe_approximated]

        ciq = CausalityInformativenessQuantification(df_approx, 'reward')
        res_ciq, res_causality = ciq.evaluate()

        single_res = {LABEL_approximation_parameters: approximation_dict,
                      LABEL_ciq_results: res_ciq,
                      LABEL_dict_causality: res_causality}

        self._store_results(single_res, df_approx)
        logging.info(f'Results computed and saved for {approximation_dict} approximation')
        gc.collect()  # Force garbage collection

        return single_res

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

        return self.results, self.dir_save"""

    def computing_CIQs(self) -> Tuple[List[Dict], str]:
        list_dict_approx = self._compute_df_approximations()

        self.results = [self._compute_and_save_single_ciq(dict_approx) for dict_approx in list_dict_approx]

        return self.results, self.dir_save

    def _collect_result(self, result):
        self.results.append(result)

    def _handle_error(self, error):
        logging.error(f'Error in process: {error}')

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
