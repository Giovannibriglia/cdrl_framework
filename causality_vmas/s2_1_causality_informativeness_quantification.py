import logging
from functools import partial
from multiprocessing.pool import Pool
from typing import Dict, Tuple, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from causality_vmas import LABEL_target_value, LABEL_predicted_value, LABEL_grouped_features, LABEL_discrete_intervals
from causality_vmas.causality_algos import CausalDiscovery, SingleCausalInference
from causality_vmas.utils import (constraints_causal_graph, _navigation_approximation,
                                  inverse_approximation_function, bn_to_dict,
                                  get_process_and_threads, check_values_in_states, markov_blanket, plot_graph)

show_progress_cd = True

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(processName)s - %(message)s')


class CausalityInformativenessQuantification:
    def __init__(self, task_name: str, df_approx: pd.DataFrame, df_target: pd.DataFrame = None, **kwargs):
        self.df_approx = df_approx
        self.df_target = df_target if df_target is not None else df_approx
        self.task_name = task_name

        self.obs_train_to_test = inverse_approximation_function(self.task_name)
        self.target_feature = kwargs.get('target_feature', 'reward')
        self.cd_algo = kwargs.get('cd_algo', 'PC')
        self.n_test_samples = kwargs.get('n_test_samples', len(df_approx)) - 1

        self.grouped_features = kwargs.get(LABEL_grouped_features, None)

        self.action_col_name = self._find_column_name('action')
        self.reward_col_name = self._find_column_name('reward')

        if not self.action_col_name or not self.reward_col_name:
            raise ValueError('Action or Reward column not found')

        self.target_feature = self._determine_target_feature()

        self.n_threads, self.n_processes = get_process_and_threads()

        self.dict_bn = None
        self.discrete_intervals_bn = None
        self.dict_scores = None
        self.causal_graph = None
        self.ci = None

    def _find_column_name(self, keyword: str) -> Optional[str]:
        """Utility to find the column name containing the specified keyword."""
        for col in self.df_approx.columns:
            if keyword in col:
                return col
        return None

    def _determine_target_feature(self) -> str:
        """Determines the correct target feature based on action or reward."""
        if 'action' in self.target_feature:
            return self.action_col_name
        elif 'reward' in self.target_feature:
            return self.reward_col_name
        else:
            raise ValueError('Invalid target feature')

    def _validate_float_dict(self, features_dict: Dict[str, List[float]], float_dict: Dict[str, float]) -> bool:
        """Validates if the float values are within the allowed discrete intervals."""
        for key, value in float_dict.items():
            if value not in features_dict[key]:
                logging.warning(f"Value {value} for {key} not in {features_dict[key]}")
                return False
        return True

    def _process_row(self, row: pd.Series) -> pd.Series:
        """Processes a single row by applying the observation function and validating the result."""

        row_dict = row.to_dict()

        if self.obs_train_to_test:
            kwargs = {
                LABEL_discrete_intervals: self.discrete_intervals_bn,
                LABEL_grouped_features: self.grouped_features,
            }
            obs = self.obs_train_to_test(row_dict, **kwargs)
        else:
            obs = row_dict

        # Validate the processed observation
        self._validate_float_dict(self.discrete_intervals_bn, obs)

        return obs

    def _compute_prediction_value(self, obs: Dict) -> Tuple[float, float]:
        """Computes the predicted value and compares it with the target value."""

        value_target = obs.pop(self.target_feature)  # Extract the target feature
        evidence = {**obs}  # Prepare evidence for inference

        # Ensure values are valid in the Bayesian network states
        check_values_in_states(self.discrete_intervals_bn, obs, evidence)

        try:
            # Perform causal inference using the provided evidence
            output_distribution = self.ci.infer(obs, self.target_feature, evidence)
            value_pred = float(max(output_distribution, key=output_distribution.get))
            value_pred = type(value_target)(value_pred)  # Ensure the predicted value has the correct type
        except MemoryError as e:
            logging.error(f"MemoryError during inference: {e}")
            value_pred = np.NaN  # Handle memory issues gracefully
        except Exception as e:
            logging.exception(f"Unexpected error during prediction computation: {e}")
            raise e

        return value_target, value_pred

    def evaluate(self) -> Tuple[Dict, nx.DiGraph, Dict]:
        """Main evaluation method for causal informativeness quantification."""
        self.dict_scores = {LABEL_target_value: [], LABEL_predicted_value: []}
        self.causal_graph = self.causal_graph or nx.DiGraph()
        self.dict_bn = {}

        try:
            self.cd_process()
        except Exception as e:
            logging.error(f'Causal discovery error: {e}')
            return self.dict_scores, self.causal_graph, self.dict_bn

        if len(self.causal_graph.nodes) == 2:
            return self.dict_scores, self.causal_graph, self.dict_bn

        # try:
        self.ci_assessment(show_progress=True)
        """except Exception as e:
            logging.error(f'Causal inference assessment failed: {e}')"""

        return self.dict_scores, self.causal_graph, self.dict_bn

    def cd_process(self):
        """Performs causal discovery."""
        # logging.info('Causal discovery...')
        cd = CausalDiscovery(self.df_approx)
        cd.training(cd_algo=self.cd_algo, show_progress=show_progress_cd)
        self.causal_graph = cd.return_causal_graph()
        self.causal_graph = constraints_causal_graph(self.causal_graph)
        self.causal_graph = markov_blanket(self.causal_graph, self.target_feature)

    def ci_assessment(self, show_progress: bool = False, parallel: bool = True):
        """Performs causal inference assessment using the causal Bayesian network (CBN)."""

        # Set up the causal Bayesian network
        try:
            self.ci = SingleCausalInference(self.df_approx, self.causal_graph)
        except Exception as e:
            logging.error(f'Failed to initialize causal Bayesian network: {e}')
            return

        # Retrieve the necessary parts from the Bayesian network
        self.discrete_intervals_bn = self.ci.return_discrete_intervals_bn()
        cbn = self.ci.return_cbn()
        self.dict_bn = bn_to_dict(cbn)
        cbn_nodes = list(cbn.nodes)
        df_test = self.df_target.loc[:self.n_test_samples, :].copy()

        # Apply the processing function to each row of the dataframe
        series_test = df_test.apply(self._process_row, axis=1)
        df_test = pd.DataFrame(series_test.apply(pd.Series))
        df_test = df_test.loc[:, cbn_nodes]
        data = df_test.to_dict(orient='records')

        # Partial function to compute prediction value
        func = partial(self._compute_prediction_value)

        if show_progress:
            results = list(map(func, tqdm(data, total=len(df_test), desc='Inferring causal knowledge...')))
        else:
            results = list(map(func, data))

        # Store the results
        for target_value, pred_value in results:
            self.dict_scores[LABEL_target_value].append(target_value)
            self.dict_scores[LABEL_predicted_value].append(pred_value)


# https://stackoverflow.com/questions/57507832/unable-to-allocate-array-with-shape-and-data-type
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
    task_name = str(input('Select task: ')).lower()
    main(task_name)
