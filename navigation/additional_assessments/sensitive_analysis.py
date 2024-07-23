import itertools
import json
import os
from typing import Dict
import yaml
import time
import pandas as pd
import multiprocessing
from navigation.causality_algos import CausalDiscovery
from navigation.utils import discretize_df, _constraints_causal_graph
from path_repo import GLOBAL_PATH_REPO


class SensitiveAnalysis_Bins_Sensors:
    def __init__(self, config_path: str):
        self.config = {}
        self.load_config(config_path)
        self.x_semidim = self.config['x_semidim']
        self.y_semidim = self.config['y_semidim']
        self.n_rows = self.config['n_rows']
        self.global_path_repo = GLOBAL_PATH_REPO
        self.task = self.config['task']
        self.n_bins_considered = self.config['n_bins_considered']
        self.n_sensors_considered = self.config['n_sensors_considered']
        self.cd_algo = self.config.get('default_cd_algo', 'pc')
        self.dataframe_path = self.config['dataframe_path']

    def load_config(self, config_path: str):
        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

    def perform_cd(self, df: pd.DataFrame, show_progress_bar: bool = False) -> Dict:
        start_time = time.time()
        cd = CausalDiscovery(df)
        cd.training(self.cd_algo, show_progress=show_progress_bar)
        causal_graph = cd.return_causal_graph()
        causal_graph = _constraints_causal_graph(causal_graph)
        computation_time_cd = round(time.time()-start_time, 4)

        causal_graph_format_storage = [(x[0], x[1]) for x in causal_graph.edges]

        result = {'causal_graph': causal_graph_format_storage, 'computation_time_cd': computation_time_cd}
        return result

    def run_single_sim(self, df: pd.DataFrame, n_bins: int, n_sensors: int, cd_algo: str,
                       show_progress_bar: bool = False):
        df_copy = df.copy()
        discretized_df = discretize_df(df_copy, n_bins, n_sensors, self.x_semidim, self.y_semidim)

        dir_saving = f'{self.global_path_repo}/{self.task}/additional_assessments/results_{cd_algo}'
        os.makedirs(dir_saving, exist_ok=True)

        result = self.perform_cd(discretized_df, show_progress_bar)

        result['n_bins'] = n_bins
        result['n_sensors'] = n_sensors
        result['cd_algo'] = self.cd_algo
        result['x_semidim'] = self.x_semidim
        result['y_semidim'] = self.y_semidim
        result['len_df'] = self.n_rows

        discretized_df.to_pickle(f'{dir_saving}/df_discretized_{n_bins}bins_{n_sensors}sensors.pkl')

        with open(f'{dir_saving}/results_{n_bins}bins_{n_sensors}sensors.json', 'w') as json_file:
            json.dump(result, json_file)

    def launcher_ground_truth(self, df: pd.DataFrame, cd_algo: str, show_progress_bar: bool = False):
        n_bins = 100
        df_copy = df.copy()
        discretized_df = discretize_df(df_copy, n_bins, x_semidim=self.x_semidim, y_semidim=self.y_semidim)

        dir_saving = f'{self.global_path_repo}/{self.task}/additional_assessments/results_{cd_algo}'
        os.makedirs(dir_saving, exist_ok=True)

        result = self.perform_cd(discretized_df, show_progress_bar)
        result['n_bins'] = n_bins
        result['cd_algo'] = self.cd_algo

        discretized_df.to_pickle(f'{dir_saving}/ground_truth_df_{n_bins}bins.pkl')

        with open(f'{dir_saving}/results_ground_truth.json', 'w') as json_file:
            json.dump(result, json_file)

    def launcher_sensitivity_analysis(self, df: pd.DataFrame, cd_algo: str):
        combinations = list(itertools.product(self.n_bins_considered, self.n_sensors_considered))

        num_chunks = multiprocessing.cpu_count()
        # chunk_size = len(combinations) // num_chunks + 1
        # chunks = [combinations[i:i + chunk_size] for i in range(0, len(combinations), chunk_size)]

        show_progress_cd = False
        with multiprocessing.Pool(processes=num_chunks) as pool:
            # Include ground truth run in multiprocessing
            pool.starmap(self.run_single_sim,
                         [(df, n_bins, n_sensors, cd_algo, show_progress_cd) for (n_bins, n_sensors) in combinations])
            pool.apply_async(self.launcher_ground_truth, (df, cd_algo, True))

    def start_analysis(self, cd_algo: str = None):
        if multiprocessing.get_start_method(allow_none=True) is None:
            multiprocessing.set_start_method('spawn')

        cd_algo = cd_algo or self.cd_algo

        dataframe = pd.read_pickle(f'{self.global_path_repo}/{self.dataframe_path}')
        agent0_columns = [col for col in dataframe.columns if 'agent_0' in col]
        dataframe = dataframe.loc[:self.n_rows, agent0_columns]

        self.launcher_sensitivity_analysis(dataframe, cd_algo)


if __name__ == '__main__':
    pipeline = SensitiveAnalysis_Bins_Sensors('../../config_simulations/config_sensitive_analysis.yaml')
    pipeline.start_analysis()
