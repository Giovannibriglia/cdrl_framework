import pandas as pd

from navigation.additional_assessments.extract_metrics import MetricsBetweenTwoGraphs
from path_repo import GLOBAL_PATH_REPO
import os
import json


# dict info:
# 'causal_graph', 'computation_time_cd', 'n_bins', 'n_sensors', 'cd_algo', 'x_semidim', 'y_semidim', 'len_df'

class CompareGraphsManager:
    def __init__(self, directory_path):
        self.directory_path = directory_path

    def load_json_files(self):
        json_files = [f for f in os.listdir(self.directory_path) if f.endswith('.json')]
        data = {}
        for json_file in json_files:
            with open(os.path.join(self.directory_path, json_file), 'r') as f:
                data[json_file] = json.load(f)
        return data

    def find_ground_truth(self, data):
        for file_name, content in data.items():
            if 'ground_truth' in file_name:
                return file_name, content
        return None, None

    def load_dataframe(self, n_bins, n_sensors, if_ground_truth: bool):
        if if_ground_truth:
            df_filename = f"ground_truth_df_{n_bins}bins.pkl"
            df_path = os.path.join(self.directory_path, df_filename)
            if os.path.exists(df_path):
                return pd.read_pickle(df_path)
            else:
                print(f"Dataframe file {df_filename} not found.")
                return None
        else:
            df_filename = f"df_discretized_{n_bins}bins_{n_sensors}sensors.pkl"
            df_path = os.path.join(self.directory_path, df_filename)
            if os.path.exists(df_path):
                return pd.read_pickle(df_path)
            else:
                print(f"Dataframe file {df_filename} not found.")
                return None

    def compare_graphs(self):
        data = self.load_json_files()
        ground_truth_file, ground_truth_content = self.find_ground_truth(data)

        if not ground_truth_content:
            print("No ground truth file found.")
            return

        causal_graph_ground_truth = ground_truth_content['causal_graph']
        n_bins_gt = ground_truth_content['n_bins']
        n_sensors_gt = ground_truth_content['n_sensors']
        df_ground_truth = self.load_dataframe(n_bins_gt, n_sensors_gt, True)

        if df_ground_truth is None:
            print("Ground truth dataframe not found.")
            return

        for file_name, content in data.items():
            print('sono qui')
            causal_graph_other = content['causal_graph']
            n_bins_other = content['n_bins']
            n_sensors_other = content['n_sensors']
            df_other = self.load_dataframe(n_bins_other, n_sensors_other, file_name == ground_truth_file)

            if df_other is not None:
                print('sono qui pronto')
                comparator = MetricsBetweenTwoGraphs(causal_graph_ground_truth, causal_graph_other, df_ground_truth,
                                                     df_other)
                result = comparator.compare()


def main():
    path_results = './results_pc'
    assessment_sensitive_analysis = CompareGraphsManager(path_results)
    assessment_sensitive_analysis.compare_graphs()


if __name__ == '__main__':
    main()
