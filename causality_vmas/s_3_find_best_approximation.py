import json
import os
from typing import List, Dict
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

from causality_vmas import LABEL_ciq_results, LABEL_scores_binary, LABEL_scores_distance, \
    LABEL_approximation_parameters, LABEL_binary_metrics, LABEL_distance_metrics, LABEL_target_value, \
    LABEL_predicted_value


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

    def evaluate(self, if_save: bool = False):
        list_metrics_for_params = []
        for result in self.all_results:
            params_info = result[f'{LABEL_approximation_parameters}']

            scores = result[f'{LABEL_ciq_results}']

            target_values = scores[f'{LABEL_target_value}']
            pred_values = scores[f'{LABEL_predicted_value}']

            binary_metrics = self._compute_binary_clf_metrics(target_values, pred_values)
            distance_metrics = self._compute_distance_metrics(target_values, pred_values)

            dict_resume = {LABEL_approximation_parameters: params_info,
                           LABEL_binary_metrics: binary_metrics,
                           LABEL_distance_metrics: distance_metrics}

            list_metrics_for_params.append(dict_resume)

        self.df_metrics = pd.DataFrame(list_metrics_for_params)
        if if_save:
            self.df_metrics.to_pickle(f'{self.path_results}/sensitive_analysis_metrics.pkl')

    def _compute_distance_metrics(self, true_labels: List, predicted_labels: List) -> Dict:
        print('**** METTI A POSTO DISTANCE METRICS')
        errors = np.array(true_labels) - np.array(predicted_labels)
        mae = np.mean(np.abs(errors))
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        max_error = np.max(np.abs(errors))

        self.distance_metrics_names = ['mae', 'mse', 'rmse', 'med_abs_err']

        dict_distance_metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'med_abs_err': max_error
        }
        """print(f'Mean Absolute Error (MAE): {mae:.4f}')
        print(f'Mean Squared Error (MSE): {mse:.4f}')
        print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
        print(f'Max Error: {max_error:.4f}')"""

        return dict_distance_metrics

    def _compute_binary_clf_metrics(self, true_labels: List, predicted_labels: List) -> Dict:

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels, normalize)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)

        self.binary_metrics_names = ['accuracy', 'precision', 'f1', 'recall']

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

        return dict_binary_metrics

    def plot_results(self, if_save: bool = False):
        # Example function to convert dictionary to a tuple
        def dict_to_tuple(d):
            return tuple(sorted(d.items()))

        df = self.df_metrics.copy()

        params = df[f'{LABEL_approximation_parameters}']
        binary_metrics = df[f'{LABEL_binary_metrics}']
        distance_metrics = df[f'{LABEL_distance_metrics}']

        params_keys = list(params[0].keys())
        binary_metrics_keys = list(binary_metrics[0].keys())
        distance_metrics_keys = list(distance_metrics[0].keys())
        # Concatenate the lists
        all_keys = params_keys + binary_metrics_keys + distance_metrics_keys
        # Remove duplicates while maintaining order
        columns = list(dict.fromkeys(all_keys))

        rows = []
        for row_n in range(len(df)):
            par_value = list(params[row_n].values())
            bin_metrics_value = list(binary_metrics[row_n].values())
            dist_metrics_value = list(distance_metrics[row_n].values())

            row = par_value + bin_metrics_value + dist_metrics_value
            rows.append(row)

        new_df = pd.DataFrame(rows, columns=columns)

        # Calculate mean of distance metrics and binary metrics for each row
        new_df['mean_dist'] = new_df[self.distance_metrics_names].mean(axis=1)
        new_df['mean_bin'] = new_df[self.binary_metrics_names].mean(axis=1)

        # Create pivot tables for heatmaps
        dist_pivot = new_df.pivot_table(index='n_bins', columns='n_sensors', values='mean_dist')
        bin_pivot = new_df.pivot_table(index='n_bins', columns='n_sensors', values='mean_bin')

        # Plot the first heatmap (mean distance metrics)
        plt.figure(figsize=(10, 8))
        sns.heatmap(dist_pivot, annot=True, cmap='viridis')
        plt.title('Heatmap of Mean Distance Metrics')
        if if_save:
            plt.savefig('mean_distance_metrics_heatmap.png')
        plt.show()

        # Plot the second heatmap (mean binary metrics)
        plt.figure(figsize=(10, 8))
        sns.heatmap(bin_pivot, annot=True, cmap='viridis')
        plt.title('Heatmap of Mean Binary Metrics')
        if if_save:
            plt.savefig('mean_binary_metrics_heatmap.png')
        plt.show()


def main():
    path_results = './df_approx_and_info_navigation'

    best_approx = BestApprox(path_results)
    best_approx.evaluate()

    best_approx.plot_results()


if __name__ == '__main__':
    main()
