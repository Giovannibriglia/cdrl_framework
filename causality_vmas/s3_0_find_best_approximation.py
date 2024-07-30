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

from causality_vmas import LABEL_ciq_scores, LABEL_scores_binary, LABEL_scores_distance, \
    LABEL_approximation_parameters, LABEL_binary_metrics, LABEL_distance_metrics, LABEL_target_value, \
    LABEL_predicted_value, LABEL_target_feature_analysis
from causality_vmas.utils import values_to_bins


class BestApprox:
    def __init__(self, path_results: str, results: Dict[str, Any] = None,
                 dfs: Dict[str, pd.DataFrame] = None):
        self.df_metrics = None
        self.path_results = path_results
        self.dir_save_best = f'{path_results}/best'
        os.makedirs(self.dir_save_best, exist_ok=True)
        self.dict_metrics = {}

        self.all_results = results if results is not None else self._extract_json_results()
        self.all_dfs = dfs if dfs is not None else self._extract_pkl_dataframes()

    def _extract_json_results(self) -> Dict[str, Any]:
        json_files = [f for f in os.listdir(self.path_results) if f.endswith('.json')]
        return {file.replace('.json', ''): json.load(open(os.path.join(self.path_results, file), 'r')) for file in
                json_files}

    def _extract_pkl_dataframes(self) -> Dict[str, pd.DataFrame]:
        pkl_files = [f for f in os.listdir(self.path_results) if f.endswith('.pkl')]
        return {file.replace('.pkl', ''): pickle.load(open(os.path.join(self.path_results, file), 'rb')) for file in
                pkl_files}

    def evaluate(self, if_save: bool = False) -> pd.DataFrame:
        self._compute_metrics()
        if if_save:
            self.df_metrics.to_pickle(f'{self.path_results}/sensitive_analysis_metrics.pkl')

        return self.df_metrics

    def _compute_distance_metrics(self, true_labels: List, predicted_labels: List, max_error_possible: int) -> Dict:
        errors = np.array(true_labels) - np.array(predicted_labels)
        mae = np.mean(np.abs(errors))
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        max_error = abs(max(errors))

        # Metrics normalization
        normalized_metrics = {
            'mae': 1 - (mae / max_error_possible),
            'mse': 1 - (mse / (max_error_possible ** 2)),
            'rmse': 1 - (rmse / max_error_possible),
            'med_abs_err': 1 - (max_error / max_error_possible)
        }
        self.dict_metrics[LABEL_distance_metrics] = list(normalized_metrics.keys())
        return normalized_metrics

    def _compute_binary_clf_metrics(self, true_labels: List, predicted_labels: List) -> Dict:
        true = [1] * len(true_labels)
        pred = [1 if true_labels[n] == predicted_labels[n] else 0 for n in range(len(predicted_labels))]

        metrics = {
            'accuracy': accuracy_score(true, pred),
            'precision': precision_score(true, pred),
            'f1': f1_score(true, pred),
            'recall': recall_score(true, pred)
        }
        self.dict_metrics[LABEL_binary_metrics] = list(metrics.keys())
        return metrics

    def _compute_causality_metrics(self):
        # TODO: implement causality assessment for evaluation
        raise NotImplementedError

    def _compute_metrics(self) -> pd.DataFrame:
        metrics_list = []

        for name_result, result in tqdm(self.all_results.items(), desc="Processing Results..."):
            params_info = result[LABEL_approximation_parameters]
            target_feature = result[LABEL_target_feature_analysis]
            intervals = params_info['discrete_intervals'][target_feature]
            max_distance_error = len(intervals)
            scores = result[LABEL_ciq_scores]

            target_values = scores[LABEL_target_value]
            pred_values = scores[LABEL_predicted_value]

            if target_values and pred_values:
                target_bins = values_to_bins(target_values, intervals)
                pred_bins = values_to_bins(pred_values, intervals)
                binary_metrics = self._compute_binary_clf_metrics(target_bins, pred_bins)
                distance_metrics = self._compute_distance_metrics(target_bins, pred_bins, max_distance_error)
            else:
                binary_metrics = {key: 0 for key in self.dict_metrics[LABEL_binary_metrics]}
                distance_metrics = {key: 0 for key in self.dict_metrics[LABEL_distance_metrics]}

            metrics_list.append({**params_info, **binary_metrics, **distance_metrics})

        self.df_metrics = pd.DataFrame(metrics_list)
        for category, names in self.dict_metrics.items():
            self.df_metrics[f'mean_{category}_metric'] = self.df_metrics[names].mean(axis=1)

        return self.df_metrics

    def _heatmap_definition(self, metric_category: str, informative_keys: List, title: str = None,
                            if_save: bool = False):
        heatmap_data = self.df_metrics.pivot_table(values=f'mean_{metric_category}_metric',
                                                   index=informative_keys[0],
                                                   columns=informative_keys[1], aggfunc='mean')
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, cmap='viridis')
        plt.title(title)
        plt.xlabel(informative_keys[1])
        plt.ylabel(informative_keys[0])
        if if_save:
            plt.savefig(f'{self.path_results}/{metric_category}_metrics.jpg')
        plt.show()

    def plot_results(self, if_save: bool = False):
        # TODO: compute informative keys in a fancier way
        informative_keys = ['n_bins', 'n_sensors']
        for category in self.dict_metrics:
            title = f'{category}-metrics category (bigger -> better)'
            self._heatmap_definition(category, informative_keys, title, if_save)

    def get_best(self, if_save: bool = False) -> Tuple[pd.DataFrame, Dict]:
        best_params = self._get_best_params_info()
        # TODO: extract key info params in a fancier way
        keys_params_info = ['n_bins', 'n_sensors', 'n_rows']

        best_name = next(name for name, result in self.all_results.items()
                         if all(
            result[LABEL_approximation_parameters][key] == best_params[key] for key in keys_params_info))

        with open(f'{self.path_results}/{best_name}.json', 'r') as f:
            best_info = json.load(f)

        df_number = re.search(r'info_(\d+)', best_name).group(1)
        best_df = pd.read_pickle(f'{self.path_results}/df_{df_number}.pkl')

        if if_save:
            best_df.to_pickle(f'{self.dir_save_best}/best_df.pkl')
            with open(f'{self.dir_save_best}/best_info.json', 'w') as f:
                json.dump(best_info, f, indent=4)

        return best_df, best_info

    def _get_best_params_info(self) -> dict:
        best_conf = {'best_params': {}, 'best_average_metrics': 0}

        for _, values in self.df_metrics.iterrows():
            avg_metric = sum(values[f'mean_{category}_metric'] for category in self.dict_metrics) / len(
                self.dict_metrics)

            if avg_metric > best_conf['best_average_metrics']:
                best_conf['best_params'] = values
                best_conf['best_average_metrics'] = avg_metric

        return best_conf['best_params'].to_dict()


def main():
    path_results = './df_approx_and_info_navigation2'

    best_approx = BestApprox(path_results)
    best_approx.evaluate(if_save=True)
    best_approx.get_best(if_save=True)
    best_approx.plot_results(if_save=True)


if __name__ == '__main__':
    main()
