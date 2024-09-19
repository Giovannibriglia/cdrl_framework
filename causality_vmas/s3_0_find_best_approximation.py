import json
import logging
import os
import shutil
from typing import List, Dict, Any
import random
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from causality_vmas import (LABEL_ciq_scores, LABEL_binary_metrics, LABEL_distance_metrics, LABEL_target_value,
                            LABEL_predicted_value, LABEL_target_feature_analysis, LABEL_discrete_intervals,
                            LABEL_dir_storing_dict_and_info, LABEL_causal_graph_distance_metrics,
                            LABEL_causal_graph_similarity_metrics)
from causality_vmas.causality_algos import CausalDiscovery
from causality_vmas.utils import values_to_bins, get_numeric_part, get_ClusteringCoefficientSimilarity, \
    get_DegreeDistributionSimilarity, get_FrobeniusNorm, get_JaccardSimilarity, get_StructuralInterventionDistance, \
    get_StructuralHammingDistance, get_MeanAbsoluteError, get_MeanSquaredError, get_RootMeanSquaredError, \
    get_MedianAbsoluteError, list_to_graph, get_fully_connected_graph, get_empty_graph, process_singularities, \
    _create_intervals, constraints_causal_graph, markov_blanket

logging.basicConfig(level=logging.INFO, format='%(processName)s - %(message)s')


class BestApprox:
    def __init__(self, path_results: str, target: nx.DiGraph | List | pd.DataFrame = None):

        self.path_results = path_results
        if target is not None:
            if isinstance(target, nx.DiGraph):
                self.G_target = target
            elif isinstance(target, list):
                self.G_target = list_to_graph(target)
            elif isinstance(target, pd.DataFrame):
                try:
                    cd = CausalDiscovery(target)
                    logging.info('Causal discovery...')
                    cd.training(show_progress=True)
                    self.G_target = cd.return_causal_graph()
                except ValueError as e:
                    if 'Data correlation matrix is singular' in str(e):
                        logging.error("Singular matrix error encountered. Preprocessing data and retrying.")
                        preprocessed_data = process_singularities(target)
                        cd = CausalDiscovery(preprocessed_data)
                        logging.info('Causal discovery...')
                        cd.training(show_progress=True)
                        self.G_target = cd.return_causal_graph()
                    else:
                        raise e
            else:
                raise ValueError('the target variable provided is not supported')

            self.G_target = constraints_causal_graph(self.G_target)
            self.G_target = markov_blanket(self.G_target, 'agent_0_reward')

            self.empty_graph = get_empty_graph(self.G_target)
            self.fully_connected_graph = get_fully_connected_graph(self.G_target)

        else:
            self.G_target = None

        self.df_metrics = None
        self.dir_save_best = f'{path_results}/best'
        os.makedirs(self.dir_save_best, exist_ok=True)
        self.dict_metrics = {}

        logging.info('Reading files...')
        self.all_scores = self._extract_json_results()

    def _setup_binary_metrics(self):
        metrics = {
            'accuracy': accuracy_score,
            'precision': precision_score,
            'f1': f1_score,
            'recall': recall_score
        }
        self.dict_metrics[LABEL_binary_metrics] = metrics

    def _setup_distance_metrics(self, max_error_possible):
        metrics = {
            'mae': lambda errors: 1 - (get_MeanAbsoluteError(errors) / max_error_possible),
            'mse': lambda errors: 1 - (get_MeanSquaredError(errors) / (max_error_possible ** 2)),
            'rmse': lambda errors: 1 - (get_RootMeanSquaredError(errors) / max_error_possible),
            'med_abs_err': lambda errors: 1 - (get_MedianAbsoluteError(errors) / max_error_possible)
        }
        self.dict_metrics[LABEL_distance_metrics] = metrics

    def _setup_causality_distance_metrics(self):

        def _rescale_from_0_to_1(value, max_value, min_value):
            if min_value == max_value:
                return 0
            return (value - min_value) / (max_value - min_value)

        def compute_rescaled_shd(G_target, G_pred):
            shd_value = get_StructuralHammingDistance(G_target, G_pred)
            return 1 - _rescale_from_0_to_1(shd_value, max_shd, min_shd)

        def compute_rescaled_sid(G_target, G_pred):
            sid_value = get_StructuralInterventionDistance(G_target, G_pred)
            return 1 - _rescale_from_0_to_1(sid_value, max_sid, min_sid)

        def compute_rescaled_frob_norm(G_target, G_pred):
            frob_norm_value = get_FrobeniusNorm(G_target, G_pred)
            return 1 - _rescale_from_0_to_1(frob_norm_value, max_frob_norm, min_frob_norm)

        min_shd = 0
        max_shd = get_StructuralHammingDistance(self.G_target, self.fully_connected_graph)
        # print('SHD: ', max_shd, min_shd)

        min_sid = 0
        max_sid = get_StructuralInterventionDistance(self.G_target, self.fully_connected_graph)
        # print('SID: ', max_sid, min_sid)

        min_frob_norm = 0
        max_frob_norm = get_FrobeniusNorm(self.G_target, self.fully_connected_graph)
        # print('FROB: ', max_frob_norm, min_frob_norm)

        metrics = {
            'shd': compute_rescaled_shd,
            'sid': compute_rescaled_sid,
            'frob_norm': compute_rescaled_frob_norm,
        }
        self.dict_metrics[LABEL_causal_graph_distance_metrics] = metrics

    def _setup_causality_similarity_metrics(self):
        metrics = {
            'js_': get_JaccardSimilarity,
            'dds': get_DegreeDistributionSimilarity,
            'ccs': get_ClusteringCoefficientSimilarity
        }
        self.dict_metrics[LABEL_causal_graph_similarity_metrics] = metrics

    def _extract_json_results(self) -> dict[int, Any]:
        json_files = [f for f in os.listdir(self.path_results) if f.endswith('.json') and 'scores' in f]
        return {get_numeric_part(filename): json.load(open(os.path.join(self.path_results, filename), 'r')) for filename
                in json_files}

    def evaluate(self):
        self._compute_metrics()

        self._store_best()

    def _compute_distance_metrics(self, true_labels: List, predicted_labels: List) -> Dict:
        errors = np.array(true_labels) - np.array(predicted_labels)

        metrics = {}
        for metric_name, metric_computation in self.dict_metrics[LABEL_distance_metrics].items():
            metrics[metric_name] = metric_computation(errors)
        return metrics

    def _compute_binary_clf_metrics(self, true_labels: List, predicted_labels: List) -> Dict:
        true = [1] * len(true_labels)
        pred = [1 if true_labels[n] == predicted_labels[n] else 0 for n in range(len(predicted_labels))]

        metrics = {}
        for metric_name, metric_computation in self.dict_metrics[LABEL_binary_metrics].items():
            metrics[metric_name] = metric_computation(true, pred)
        return metrics

    def _compute_causality_distance_metrics(self, G_pred: nx.DiGraph):
        metrics = {}
        for metric_name, metric_computation in self.dict_metrics[LABEL_causal_graph_distance_metrics].items():
            if G_pred.number_of_edges() > 0 and G_pred.number_of_nodes() > 0:
                metrics[metric_name] = metric_computation(self.G_target, G_pred)
            else:
                metrics[metric_name] = 0
        return metrics

    def _compute_causality_similarity_metrics(self, G_pred: nx.DiGraph):
        metrics = {}
        for metric_name, metric_computation in self.dict_metrics[LABEL_causal_graph_similarity_metrics].items():
            if G_pred.number_of_edges() > 0 and G_pred.number_of_nodes() > 0:
                metrics[metric_name] = metric_computation(self.G_target, G_pred)
            else:
                metrics[metric_name] = 0

        return metrics

    def _compute_metrics(self) -> pd.DataFrame:
        metrics_list = []

        self._setup_binary_metrics()
        if self.G_target is not None and self.G_target.number_of_edges() > 0 and self.G_target.number_of_nodes() > 0:
            self._setup_causality_distance_metrics()
            self._setup_causality_similarity_metrics()

        self.map_numbers = {}
        count = 0
        for index_res, result in tqdm(self.all_scores.items(), desc="Processing Results..."):
            with open(f'{self.path_results}/approx_params_{index_res}.json', 'r') as file:
                params_approx = json.load(file)

            target_feature = result[LABEL_target_feature_analysis]
            scores = result[LABEL_ciq_scores]

            with open(f'{self.path_results}/others_{index_res}.json', 'r') as file:
                others = json.load(file)

            if target_feature in others[LABEL_discrete_intervals].keys():
                intervals_target_feature = others[LABEL_discrete_intervals][target_feature]
            else:
                df = pd.read_pickle(f'{self.path_results}/df_{index_res}.pkl')
                target_feature_series = df[target_feature]
                max_val_target_feature = target_feature_series.max()
                min_val_target_feature = target_feature_series.min()
                n_intervals = len(target_feature_series.unique())
                intervals_target_feature = _create_intervals(min_val_target_feature, max_val_target_feature, n_intervals)

            max_distance_error = len(intervals_target_feature)
            self._setup_distance_metrics(max_distance_error)

            target_values = scores[LABEL_target_value]
            pred_values = scores[LABEL_predicted_value]

            if target_values and pred_values:
                target_bins = values_to_bins(target_values, intervals_target_feature)
                pred_bins = values_to_bins(pred_values, intervals_target_feature)

                binary_metrics = self._compute_binary_clf_metrics(target_bins, pred_bins)
                distance_metrics = self._compute_distance_metrics(target_bins, pred_bins)
            else:
                binary_metrics = {key: 0 for key in self.dict_metrics[LABEL_binary_metrics]}
                distance_metrics = {key: 0 for key in self.dict_metrics[LABEL_distance_metrics]}

            if self.G_target is not None and self.G_target.number_of_edges() > 0 and self.G_target.number_of_nodes() > 0:
                with open(f'{self.path_results}/causal_graph_{index_res}.json', 'r') as file:
                    list_causal_graph = json.load(file)
                causal_graph = list_to_graph(list_causal_graph)
                causal_distance_metrics = self._compute_causality_distance_metrics(causal_graph)
                causal_similarity_metrics = self._compute_causality_similarity_metrics(causal_graph)
                metrics_list.append({**params_approx, **binary_metrics, **distance_metrics,
                                     **causal_distance_metrics, **causal_similarity_metrics})
            else:
                metrics_list.append({**params_approx, **binary_metrics, **distance_metrics})

            self.map_numbers[count] = index_res
            count += 1

        self.df_metrics = pd.DataFrame(metrics_list)
        for category, names in self.dict_metrics.items():
            if isinstance(names, dict):
                names = list(names.keys())
            self.df_metrics[f'mean_{category}_metric'] = self.df_metrics[names].mean(axis=1)

        self.df_metrics.to_pickle(f'{self.path_results}/sensitive_analysis_results.pkl')

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
        for category in self.dict_metrics:
            title = f'{category}-metrics category (bigger -> better)'
            informative_keys = self._get_most_informative_keys(category)
            self._heatmap_definition(category, informative_keys, title, if_save)

    def _store_best(self):

        index_best_conf = self._get_best_params_info()
        print('Best configuration index: ', index_best_conf)
        if index_best_conf > -1:
            # df
            shutil.copy(os.path.join(self.path_results, f'df_{index_best_conf}.pkl'),
                        os.path.join(self.dir_save_best, f'best_df.pkl'))
            # approx params
            shutil.copy(os.path.join(self.path_results, f'approx_params_{index_best_conf}.json'),
                        os.path.join(self.dir_save_best, f'best_approx_params.json'))
            # bn
            shutil.copy(os.path.join(self.path_results, f'bn_params_{index_best_conf}.json'),
                        os.path.join(self.dir_save_best, f'best_bn_params.json'))
            # causal graph
            shutil.copy(os.path.join(self.path_results, f'causal_graph_{index_best_conf}.json'),
                        os.path.join(self.dir_save_best, f'best_causal_graph.json'))
            # others
            shutil.copy(os.path.join(self.path_results, f'others_{index_best_conf}.json'),
                        os.path.join(self.dir_save_best, f'best_others.json'))
            # others
            shutil.copy(os.path.join(self.path_results, f'scores_{index_best_conf}.json'),
                        os.path.join(self.dir_save_best, f'best_scores.json'))

    def _get_best_params_info(self) -> int:
        best_conf = {'index': -1, 'best_params': {}, 'best_average_metrics': 0}

        for n, values in self.df_metrics.iterrows():
            with open(f'causal_graph_{n}.json') as f:
                causal_graph_list = json.load(f)
            causal_graph = list_to_graph(causal_graph_list)
            with open(f'bn_params_{n}.json') as f:
                bn_params = json.load(f)

            if causal_graph is not nx.DiGraph() and bn_params is not {}:
                avg_metric = 1
                for category in self.dict_metrics:
                    avg_metric *= values[f'mean_{category}_metric']

                if avg_metric > best_conf['best_average_metrics']:
                    best_conf['index'] = n
                    best_conf['best_params'] = values
                    best_conf['best_average_metrics'] = avg_metric

        best_index_here = best_conf['index']
        if best_index_here > -1:
            best_index_out = int(self.map_numbers[best_index_here])
            return best_index_out
        else:
            return -1

    def _get_most_informative_keys(self, metric_category: str) -> List:
        df = self.df_metrics.copy()

        scores_features = [col for col in df.columns if f'mean_{metric_category}_metric' in col]
        params_features = []
        for col in df.columns:
            if 'metric' not in col:
                if all(col not in metrics_list for metrics_list in self.dict_metrics.values()):
                    params_features.append(col)

        X = df[params_features]
        y = df[scores_features].values.ravel()

        if not np.isnan(y).any():
            model = RandomForestRegressor()
            model.fit(X, y)

            importances = sum(est.feature_importances_ for est in model.estimators_) / len(model.estimators_)

            feature_importances = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)

            top_2_features = sorted(feature_importances.head(2)['Feature'].tolist())
            return top_2_features
        else:
            return random.sample(params_features, 2)

def main(task):
    path_results = f'./{LABEL_dir_storing_dict_and_info}_{task}'

    df = pd.read_pickle(f'./dataframes/df_{task}_pomdp_discrete_actions_0.pkl')
    agent0_columns = [col for col in df.columns if 'agent_0' in col]
    df = df.loc[:, agent0_columns]

    best_approx = BestApprox(path_results, df)
    best_approx.evaluate()
    best_approx.plot_results(if_save=True)


if __name__ == '__main__':
    main('best')
