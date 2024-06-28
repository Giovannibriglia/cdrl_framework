import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from navigation.utils import IQM_mean_std
from path_repo import GLOBAL_PATH_REPO
from navigation.causality_algos import CausalDiscovery, CausalInferenceForRL

n_bins = 20
agent_n = 0
n_rows = 50000
obs = 'mdp'


def _rescale_value(kind: str, value: float | int):
    def discretize_value(value, intervals):
        # Find the interval where the value fits
        for i in range(len(intervals) - 1):
            if intervals[i] <= value < intervals[i + 1]:
                return (intervals[i] + intervals[i + 1]) / 2
        # Handle the edge cases
        if value < intervals[0]:
            return intervals[0]
        elif value >= intervals[-1]:
            return intervals[-1]

    """def discretize_value(value, intervals):
        # it returns the index of the interval where the value fits.
        for i in range(len(intervals) - 1):
            if intervals[i] <= value < intervals[i + 1]:
                return i
        if value < intervals[0]:
            return 0
        elif value >= intervals[-1]:
            return len(intervals) - 2"""

    def create_intervals(min_val, max_val, n_intervals, scale='linear'):
        if scale == 'exponential':
            # Generate n_intervals points using exponential scaling
            intervals = np.logspace(0, 1, n_intervals, base=10) - 1
            intervals = intervals / (10 - 1)  # Normalize to range 0-1
            intervals = min_val + (max_val - min_val) * intervals
        elif scale == 'linear':
            intervals = np.linspace(min_val, max_val, n_intervals)
        else:
            raise ValueError("Unsupported scale type. Use 'exponential' or 'linear'.")
        return intervals

    if kind == 'reward':
        max_value = 1
        min_value = -1 * 4 - max(0.5, 0.5)
        # n = 10
        intervals = create_intervals(min_value, max_value, n_bins, scale='linear')
        # print('reward bins: ', n)
    elif kind == 'DX' or kind == 'DY':
        max_value = -1
        min_value = +1  # -self.x_semidim * 2 if kind == 'DX' else -self.y_semidim * 2
        # n = 10 # int((self.x_semidim/0.05)**2 * self.x_semidim*2) if kind == 'DX' else int((self.y_semidim/0.05)**2 * self.y_semidim*2)
        intervals = create_intervals(min_value, max_value, n_bins, scale='linear')
        # print('DX-DY bins: ', n)
    elif kind == 'VX' or kind == 'VY':
        max_value = 0.5
        min_value = -0.5
        # n = 5 # int((self.x_semidim/0.05)**2 * self.x_semidim*2) if kind == 'DX' else int((self.y_semidim/0.05)**2 * self.y_semidim*2)
        intervals = create_intervals(min_value, max_value, n_bins, scale='linear')
        # print('VX-VY bins: ', n)
    elif kind == 'sensor':
        max_value = 1.0
        min_value = 0.0
        # n = 5
        intervals = create_intervals(min_value, max_value, n_bins, scale='linear')
        # print('sensor bins: ', n)
    elif kind == 'posX' or kind == 'posY':
        max_value = 0.5 if kind == 'posX' else 0.5
        min_value = -0.5 if kind == 'posX' else -0.5
        # n = 20 # int((self.x_semidim/0.05)**2 * self.x_semidim*2) if kind == 'posX' else int((self.y_semidim/0.05)**2 * self.y_semidim*2)
        intervals = create_intervals(min_value, max_value, n_bins, scale='linear')
        # print('posX-posY bins: ', n)

    if kind == 'sensor':
        th = 0.7
        if value >= th:
            rescaled_value = 1
        elif 0 < value < th:
            rescaled_value = 0.5
        else:
            rescaled_value = 0
    else:
        index = discretize_value(value, intervals)
        rescaled_value = index
        # rescaled_value = int((index / (len(intervals) - 2)) * (n - 1))

    return rescaled_value


def get_new_df(dataframe: pd.DataFrame):
    agent_X_columns = [s for s in dataframe.columns.to_list() if f'agent_{agent_n}' in s]

    df = dataframe.loc[:, agent_X_columns]
    df_columns = df.columns.to_list()
    std_dev = df.std()
    non_zero_std_columns = std_dev[std_dev != 0].index
    df_filtered = df[non_zero_std_columns]

    new_dataframe = pd.DataFrame(columns=df_filtered.columns, index=df_filtered.index)
    for col in df_filtered.columns:
        if 'action' not in col:
            if 'sensor' in col:
                kind = 'sensor'
            elif 'DX' in col:
                kind = 'DX'
            elif 'DY' in col:
                kind = 'DY'
            elif 'reward' in col:
                kind = 'reward'
            elif 'pX' in col:
                kind = 'posX'
            elif 'PY' in col:
                kind = 'posY'
            elif 'VX' in col:
                kind = 'VX'
            elif 'VY' in col:
                kind = 'VY'
            else:
                kind = None

            if kind:
                new_dataframe[col] = df_filtered[col].apply(lambda value: _rescale_value(kind, value))
            else:
                new_dataframe[col] = df_filtered[col]
        else:
            new_dataframe[col] = df_filtered[col]

    return new_dataframe


def get_boundaries(dataframe: pd.DataFrame):
    for col in dataframe.columns.to_list():
        iqm_mean, iqm_std = IQM_mean_std(dataframe[col])
        print(
            f'{col} -> max: {dataframe[col].max()}, min: {dataframe[col].min()}, mean: {dataframe[col].mean()}, std: {dataframe[col].std()}, iqm_mean: {iqm_mean}, iqm_std: {iqm_std}')

    fig = plt.figure(dpi=500, figsize=(16, 9))
    plt.plot(dataframe['agent_0_reward'])
    plt.show()


df = pd.read_pickle(f'{GLOBAL_PATH_REPO}/navigation/causal_knowledge/offline_ok/df_random_{obs}_1000000.pkl')
df = df.head(n_rows)

new_df = get_new_df(df)

cd = CausalDiscovery(new_df, f'navigation/causal_knowledge/offline/navigation', f'agent{agent_n}_{obs}')
cd.training()
causal_graph = cd.return_causal_graph()
df_for_causality = cd.return_df()

ci = CausalInferenceForRL(df_for_causality, causal_graph)
causal_table = ci.create_causal_table()

causal_table.to_pickle(f'{GLOBAL_PATH_REPO}/navigation/causal_knowledge/offline/navigation/causal_table.pkl')
causal_table.to_pickle(f'{GLOBAL_PATH_REPO}/navigation/causal_knowledge/offline/navigation/causal_table.xlsx')
