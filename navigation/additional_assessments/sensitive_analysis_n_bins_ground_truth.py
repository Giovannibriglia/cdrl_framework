import itertools
import json
import os
from navigation.additional_assessments.utils import run_causal_discovery
from navigation.utils import discretize_df

from path_repo import GLOBAL_PATH_REPO
import pandas as pd
import multiprocessing

x_semidim = 0.5
y_semidim = 0.5

N_ROWS = 200000


def run_for_ground_truth(df: pd.DataFrame, cd_algo: str = 'pc'):
    n_bins = 100
    df_copy = df.copy()
    new_df = discretize_df(df_copy, n_bins, x_semidim=x_semidim, y_semidim=y_semidim)

    DIR_SAVING = f'{GLOBAL_PATH_REPO}/navigation/additional_assessments/results_{cd_algo}'
    os.makedirs(DIR_SAVING, exist_ok=True)

    result = run_causal_discovery(new_df, cd_algo, show_progress_cd=True)

    new_df.to_pickle(f'{DIR_SAVING}/ground_truth_df_{n_bins}bins_.pkl')

    with open(f'{DIR_SAVING}/results_ground_truth.json', 'w') as json_file:
        json.dump(result, json_file)


def launch(cd_algo: str = None):
    # Ensure the context is set only once
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method('spawn')

    if cd_algo is None:
        CD_ALGO = input('Causal discovery algorithm: ')
    else:
        CD_ALGO = cd_algo

    dataframe = pd.read_pickle(f'{GLOBAL_PATH_REPO}/navigation/causal_knowledge/offline_ok/df_random_mdp_1000000.pkl')

    columns_agent0 = [s for s in dataframe.columns.to_list() if 'agent_0' in s]

    dataframe = dataframe.loc[:N_ROWS, columns_agent0]

    run_for_ground_truth(dataframe, CD_ALGO)


if __name__ == '__main__':
    launch()
