import json
import os
import pickle
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import re

from causality_vmas import LABEL_ciq_results, LABEL_scores_binary, LABEL_scores_distance, \
    LABEL_approximation_parameters, LABEL_binary_metrics, LABEL_distance_metrics, LABEL_target_value, \
    LABEL_predicted_value, LABEL_target_feature
from causality_vmas.utils import values_to_bins


class BestApprox:
    def __init__(self, path_results: str, results: Dict[str, Any] = None,
                 dfs: Dict[str, pd.DataFrame] = None):
        self.df_metrics = None
        self.path_results = path_results
        self.dir_save_best = f'{path_results}/best'
        os.makedirs(self.dir_save_best, exist_ok=True)

        self.dict_metrics = {}

        if results is None:
            self.all_results = self._extract_json_results()
            self.all_dfs = self._extract_pkl_dataframes()
        else:
            self.all_results = results
            self.all_dfs = dfs

    def _extract_json_results(self) -> Dict[str, Any]:
        # List all files in the folder
        all_files = os.listdir(self.path_results)

        # Filter for .json files
        json_files = [file for file in all_files if file.endswith('.json')]

        # Read and process each .json file
        json_data = {}
        for file in json_files:
            file_path = os.path.join(self.path_results, file)
            with open(file_path, 'r') as f:
                json_data[file.replace('.json', '')] = json.load(f)
                # json_data.append(json.load(f))

        return json_data

    def _extract_pkl_dataframes(self) -> Dict[str, pd.DataFrame]:
        # List all files in the folder
        all_files = os.listdir(self.path_results)

        # Filter for .pkl files
        pkl_files = [file for file in all_files if file.endswith('.pkl')]

        # Read and process each .pkl file
        pkl_data = {}
        for file in pkl_files:
            file_path = os.path.join(self.path_results, file)
            with open(file_path, 'rb') as f:
                pkl_data[file.replace('.pkl', '')] = pickle.load(f)

        return pkl_data

    def evaluate(self, if_save: bool = False):
        self._compute_metrics()

        if if_save:
            self.df_metrics.to_pickle(f'{self.path_results}/sensitive_analysis_metrics.pkl')

    def _compute_causal_graph_metrics(self):
        # TODO: causal metrics
        raise NotImplementedError('causal graph metrics not still implemented')

    def _compute_distance_metrics(self, true_labels: List, predicted_labels: List, max_error_possible: int) -> Dict:
        errors = np.array(true_labels) - np.array(predicted_labels)

        mae = np.mean(np.abs(errors))
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        max_error = abs(1 - max(errors))

        # Metrics normalization
        normalized_mae = 1 - (mae / max_error_possible)
        normalized_mse = 1 - (mse / (max_error_possible ** 2))
        normalized_rmse = 1 - (rmse / max_error_possible)
        normalized_max_error = 1 - (max_error / max_error_possible)

        dict_distance_metrics = {
            'mae': normalized_mae,
            'mse': normalized_mse,
            'rmse': normalized_rmse,
            'med_abs_err': normalized_max_error
        }
        """print(f'Mean Absolute Error (MAE): {mae:.4f}')
        print(f'Mean Squared Error (MSE): {mse:.4f}')
        print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
        print(f'Max Error: {max_error:.4f}')"""
        self.dict_metrics[LABEL_distance_metrics] = list(dict_distance_metrics.keys())

        return dict_distance_metrics

    def _compute_binary_clf_metrics(self, true_labels: List, predicted_labels: List) -> Dict:
        true = [1] * len(true_labels)
        pred = [1 if true_labels[n] == predicted_labels[n] else 0 for n in range(len(predicted_labels))]

        # Calculate metrics
        accuracy = accuracy_score(true, pred)
        precision = precision_score(true, pred)
        recall = recall_score(true, pred)
        f1 = f1_score(true, pred)

        dict_binary_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'f1': f1,
            'recall': recall
        }
        """print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")"""
        self.dict_metrics[LABEL_binary_metrics] = list(dict_binary_metrics.keys())

        return dict_binary_metrics

    def _compute_metrics(self) -> pd.DataFrame:
        list_metrics_for_params = []

        for name_result, result in tqdm(self.all_results.items(), desc="Processing Results..."):
            params_info = result[f'{LABEL_approximation_parameters}']
            target_feature = result[f'{LABEL_target_feature}']

            intervals = params_info['discrete_intervals'][f'{target_feature}']
            max_distance_error = len(intervals)

            scores = result[f'{LABEL_ciq_results}']

            target_values = scores[f'{LABEL_target_value}']
            pred_values = scores[f'{LABEL_predicted_value}']

            if len(target_values) > 0 and len(pred_values) > 0:
                target_bins = values_to_bins(target_values, intervals)
                pred_bins = values_to_bins(pred_values, intervals)

                binary_metrics = self._compute_binary_clf_metrics(target_bins, pred_bins)

                distance_metrics = self._compute_distance_metrics(target_bins, pred_bins, max_distance_error)
            else:
                binary_metrics = {key: 0 for key in self.dict_metrics[LABEL_binary_metrics]}
                distance_metrics = {key: 0 for key in self.dict_metrics[LABEL_distance_metrics]}

            dict_resume = {
                **params_info,
                **binary_metrics,
                **distance_metrics
            }

            list_metrics_for_params.append(dict_resume)

        self.df_metrics = pd.DataFrame(list_metrics_for_params)

        for metric_category, metrics_names in self.dict_metrics.items():
            self.df_metrics[f'mean_{metric_category}_metric'] = self.df_metrics[metrics_names].mean(axis=1)

        return self.df_metrics

    def _heatmap_definition(self, metric_category: str, informative_keys: List, title: str = None,
                            if_save: bool = False):

        heatmap_data_binary = self.df_metrics.pivot_table(values=f'mean_{metric_category}_metric',
                                                          index=informative_keys[0],
                                                          columns=informative_keys[1], aggfunc='mean')

        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data_binary, annot=True, cmap='viridis')
        plt.title(f'{title}')
        plt.xlabel(informative_keys[1])
        plt.ylabel(informative_keys[0])
        if if_save:
            plt.savefig(f'{self.path_results}/{metric_category}_metrics.jpg')
        plt.show()

    def plot_results(self, if_save: bool = False):

        # TODO: most informative keys selection
        informative_keys = ['n_bins', 'n_sensors']

        for metric_category, metrics_names in self.dict_metrics.items():
            title = f'{metric_category}-metrics category (bigger -> better)'
            self._heatmap_definition(metric_category, informative_keys, title, if_save)

    def get_best(self, if_save: bool = False) -> Tuple[pd.DataFrame, Dict]:
        best_params = self._get_best_params_info()

        # TODO: make better
        keys_params_info = ['n_bins', 'n_sensors', 'n_rows']

        best_name = [name_result
                     for name_result, result in self.all_results.items()
                     if
                     all(result[LABEL_approximation_parameters][key1] == best_params[key1] for key1 in keys_params_info)
                     ][0]

        with open(f'{self.path_results}/{best_name}.json', 'r') as f:
            best_info = json.load(f)

        match = re.search(r'info_(\d{1,7})', best_name)
        df_number = match.group(1)
        best_df = pd.read_pickle(f'{self.path_results}/df_{df_number}.pkl')

        print(f'*** Best: {best_name} and df_{df_number} ***')

        if if_save:
            best_df.to_pickle(f'{self.dir_save_best}/best_df.pkl')

            with open(f'{self.dir_save_best}/best_info.json', 'w') as f:
                json.dump(best_info, f, indent=8)

        return best_df, best_info

    def _get_best_params_info(self) -> dict:
        best_params_label = 'best_params'
        best_av_metrics = 'best_average_metrics'
        dict_best_conf = {best_params_label: {}, best_av_metrics: 0}

        for n, values in self.df_metrics.iterrows():
            av_mean = 0

            for metric_category, metrics_names in self.dict_metrics.items():
                av_mean += self.df_metrics.loc[n, f'mean_{metric_category}_metric']

            av_mean /= len(self.dict_metrics)

            if av_mean > dict_best_conf[best_av_metrics]:
                dict_best_conf[best_params_label] = values
                dict_best_conf[best_av_metrics] = av_mean

        return dict_best_conf[best_params_label].to_dict()


def main():
    path_results = './df_approx_and_info_navigation'

    best_approx = BestApprox(path_results)
    best_approx.evaluate(if_save=True)
    best_approx.get_best(if_save=True)
    best_approx.plot_results(if_save=True)


if __name__ == '__main__':
    main()
