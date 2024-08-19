import logging
import os
from concurrent.futures import as_completed, ProcessPoolExecutor
from multiprocessing.pool import ThreadPool
from typing import Dict, Tuple, List
import networkx as nx
import pandas as pd
import psutil
from tqdm import tqdm

from causality_vmas import LABEL_target_value, LABEL_predicted_value, LABEL_grouped_features, LABEL_discrete_intervals
from causality_vmas.causality_algos import CausalDiscovery, SingleCausalInference
from causality_vmas.utils import (constraints_causal_graph, _navigation_approximation,
                                  plot_graph, extract_intervals_from_bn, inverse_approximation_function, bn_to_dict)

show_progress_cd = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(processName)s - %(message)s')


class CausalityInformativenessQuantification:
    def __init__(self, task_name: str, df_approx: pd.DataFrame, df_target: pd.DataFrame = None, **kwargs):

        self.df_approx = df_approx
        self.df_target = df_target if df_target is not None else self.df_approx.copy()

        self.task_name = task_name
        self.obs_train_to_test = inverse_approximation_function(self.task_name)

        target_feature = kwargs.get('target_feature', 'reward')
        self.cd_algo = kwargs.get('cd_algo', 'PC')
        self.n_test_samples = kwargs.get('n_test_samples', 10000)-1
        self.grouped_features = kwargs[LABEL_grouped_features]

        self.dict_bn = None
        self.discrete_intervals_bn = None
        self.dict_scores = None
        self.causal_graph = None
        self.ci = None

        self.action_col_name = str([s for s in df_approx.columns.to_list() if 'action' in s][0])
        if not self.action_col_name:
            raise ValueError('Action column did not find')
        self.reward_col_name = str([s for s in df_approx.columns.to_list() if 'reward' in s][0])
        if not self.reward_col_name:
            raise ValueError('Reward column did not find')

        if 'action' in target_feature:
            self.target_feature = self.action_col_name
        elif 'reward' in target_feature:
            self.target_feature = self.reward_col_name
        else:
            raise ValueError('Target feature is not correct')

    def evaluate(self) -> Tuple[Dict, nx.DiGraph, Dict]:
        self.dict_scores = {LABEL_target_value: [], LABEL_predicted_value: []}
        self.causal_graph = self.causal_graph if self.causal_graph is not None else nx.DiGraph()
        self.dict_bn = {}

        try:
            self.cd_process()
        except Exception as e:
            logging.error(f'Causal discovery error: {e}')
            return self.dict_scores, self.causal_graph, self.dict_bn

        self.ci_assessment(show_progress=True)
        return self.dict_scores, self.causal_graph, self.dict_bn

    def cd_process(self):
        logging.info('Causal discovery...')
        cd = CausalDiscovery(self.df_approx)
        cd.training(cd_algo=self.cd_algo, show_progress=show_progress_cd)
        self.causal_graph = cd.return_causal_graph()
        self.causal_graph = constraints_causal_graph(self.causal_graph)
        # plot_graph(self.causal_graph, title='', if_show=True)

    def ci_assessment(self, show_progress: bool = False):

        logging.info(f'Setting up causal bayesian network...')
        try:
            self.ci = SingleCausalInference(self.df_approx, self.causal_graph)
        except Exception as e:
            logging.error(f'Error in causal bayesian network definition: {e}')
            return

        cbn = self.ci.return_cbn()
        self.discrete_intervals_bn = extract_intervals_from_bn(cbn)
        self.dict_bn = bn_to_dict(cbn)

        df_test = self.df_target.loc[:min(self.n_test_samples, len(self.df_target)), :]

        tasks = [row.to_dict() for n, row in df_test.iterrows()]

        try:
            with ThreadPool() as pool:
                if show_progress:
                    results = list(tqdm(pool.imap(self._process_row, tasks), total=len(tasks),
                                        desc='Inferring causal knowledge...'))
                else:
                    results = pool.map(self._process_row, tasks)

            for target_value, pred_value in results:
                self.dict_scores[LABEL_target_value].append(target_value)
                self.dict_scores[LABEL_predicted_value].append(pred_value)

        except Exception as e:
            logging.error(f'Causal inference assessment failed: {e}')
            return

    def _process_row(self, row: Dict) -> Tuple[float, float]:

        value_target = row[self.target_feature]

        input_obs = {key: value for key, value in row.items() if self.target_feature not in key}

        if self.obs_train_to_test is not None:
            kwargs = {}
            kwargs[LABEL_discrete_intervals] = self.discrete_intervals_bn
            kwargs[LABEL_grouped_features] = self.grouped_features
            obs = self.obs_train_to_test(input_obs, **kwargs)
        else:
            obs = input_obs

        evidence = obs.copy()
        output_distribution = self.ci.infer(obs, self.target_feature, evidence)

        value_pred = max(output_distribution, key=output_distribution.get)
        value_pred = float(value_pred)
        value_pred = type(value_target)(value_pred)

        return value_target, value_pred


def main(task: str):
    task = task.lower()

    df_test = pd.read_pickle(f'./dataframes/df_{task}_pomdp_discrete_actions_0.pkl')
    agent0_columns = [col for col in df_test.columns if 'agent_0' in col]
    df_test = df_test.loc[:, agent0_columns]

    dict_approx = _navigation_approximation((df_test, {'BINS': 20, 'SENSORS': 1, 'ROWS': 20000}))
    df_train = dict_approx['new_df']
    # get_df_boundaries(df)

    ciq = CausalityInformativenessQuantification(task, df_train, df_test)
    ciq.evaluate()


if __name__ == '__main__':
    task_name = str(input('Select task: '))
    main(task_name)
