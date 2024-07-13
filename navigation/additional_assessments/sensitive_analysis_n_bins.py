import itertools
import json
import os
import time
from navigation.causality_algos import CausalDiscovery
from navigation.utils import discretize_df
from path_repo import GLOBAL_PATH_REPO
import pandas as pd
import multiprocessing

x_semidim = 0.5
y_semidim = 0.5

N_ROWS = 200000

show_progress_cd = True


def run_causal_discovery(df: pd.DataFrame, n_bins: int, n_sensors: int, cd_algo: str = 'pc'):
    df_copy = df.copy()
    new_df = discretize_df(df_copy, n_bins, n_sensors, x_semidim, y_semidim)

    initial_time = time.time()

    cd = CausalDiscovery(new_df)
    cd.training(cd_algo, show_progress=show_progress_cd)
    causal_graph = cd.return_causal_graph()

    computation_time = time.time() - initial_time  # in minutes

    result = {
        'causal_graph': causal_graph,
        'computation_time_sec': computation_time
    }

    DIR_SAVING = f'{GLOBAL_PATH_REPO}/navigation/additional_assessments/causal_graphs_{cd_algo}'
    os.makedirs(DIR_SAVING, exist_ok=True)

    with open(f'{DIR_SAVING}/causal_graph_{n_bins}bins_{n_sensors}sensors.json', 'w') as json_file:
        json.dump(result, json_file)

    print(
        f'Time to perform CD on df with {n_bins} bins for discretization and {n_sensors} sensors considered: {computation_time} sec')


def run_sensitive_analysis(df: pd.DataFrame, list_n_bins: list, list_n_sensors: list, cd_algo: str = 'pc'):
    combinations = list(itertools.product(list_n_bins, list_n_sensors))

    num_chunks = multiprocessing.cpu_count()
    chunk_size = len(combinations) // num_chunks + 1
    chunks = [combinations[i:i + chunk_size] for i in range(0, len(combinations), chunk_size)]

    with multiprocessing.Pool(processes=num_chunks) as pool:
        pool.starmap(run_causal_discovery, [(df, n_bins, n_sensors, cd_algo) for (n_bins, n_sensors) in combinations])


def launch(cd_algo: str = None):
    # Ensure the context is set only once
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method('spawn')

    if cd_algo is None:
        CD_ALGO = input('Causal discovery algorithm: ')
    else:
        CD_ALGO = cd_algo
    N_BINS_CONSIDERED = [5, 10, 15, 20, 30, 50, 100]
    N_SENSORS_CONSIDERED = [1, 3, 5, 8]
    dataframe = pd.read_pickle(f'{GLOBAL_PATH_REPO}/navigation/causal_knowledge/offline_ok/df_random_mdp_1000000.pkl')

    columns_agent0 = [s for s in dataframe.columns.to_list() if 'agent_0' in s]

    dataframe = dataframe.loc[:N_ROWS, columns_agent0]

    run_sensitive_analysis(dataframe, N_BINS_CONSIDERED, N_SENSORS_CONSIDERED, CD_ALGO)


if __name__ == '__main__':
    launch()
