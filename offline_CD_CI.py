import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import tensor
import torch
from navigation.utils import IQM_mean_std
from path_repo import GLOBAL_PATH_REPO
from navigation.causality_algos import CausalDiscovery, CausalInferenceForRL

n_bins = 10
agent_n = 0
n_rows = 50000
obs = 'mdp'
num_sensors = 1


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
        max_value = 0.35
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

    """if kind == 'sensor':
        th = 0.7
        if value >= th:
            rescaled_value = 1
        elif 0 < value < th:
            rescaled_value = 0.5
        else:
            rescaled_value = 0
    else:"""
    index = discretize_value(value, intervals)
    rescaled_value = index
    # rescaled_value = int((index / (len(intervals) - 2)) * (n - 1))

    return rescaled_value


def get_new_df(dataframe: pd.DataFrame):
    agent_X_columns = [s for s in dataframe.columns.to_list() if f'agent_{agent_n}' in s]

    dataframe = dataframe.loc[:, agent_X_columns]
    dataframe = group_sensor_variables(dataframe)
    std_dev = dataframe.std()
    non_zero_std_columns = std_dev[std_dev != 0].index

    df_filtered = dataframe[non_zero_std_columns]

    new_dataframe = pd.DataFrame(columns=df_filtered.columns, index=df_filtered.index)
    for col in df_filtered.columns:
        if 'action' not in col:
            if 'DX' in col:
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
            # elif 'sensor' in col:
            #    kind = 'sensor'
            else:
                kind = None

            if kind:
                new_dataframe[col] = df_filtered[col].apply(lambda value: _rescale_value(kind, value))
            else:
                new_dataframe[col] = df_filtered[col]
        else:
            new_dataframe[col] = df_filtered[col]

    # print(new_dataframe['agent_0_next_DX'].std())
    # new_dataframe = new_dataframe.loc[:, new_dataframe.std() != 0]

    return new_dataframe


def group_sensor_variables(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Step 1: Identify sensor columns
    sensor_columns = [col for col in dataframe.columns if 'sensor' in col]
    # Step 2: Determine maximum sensor value per row
    dataframe['sensor_on'] = dataframe[sensor_columns].idxmax(axis=1)
    # Step 3: Extract sensor number from column names
    dataframe['sensor_on'] = dataframe['sensor_on'].str.extract(r'sensor(\d+)').astype(int)
    # Step 4: Drop original sensor columns
    dataframe.drop(columns=sensor_columns, inplace=True)

    return dataframe


def get_boundaries(dataframe: pd.DataFrame):
    for col in dataframe.columns.to_list():
        iqm_mean, iqm_std = IQM_mean_std(dataframe[col])
        print(
            f'{col} -> max: {dataframe[col].max()}, min: {dataframe[col].min()}, mean: {dataframe[col].mean()}, std: {dataframe[col].std()}, iqm_mean: {iqm_mean}, iqm_std: {iqm_std}')

    fig = plt.figure(dpi=500, figsize=(16, 9))
    plt.plot(dataframe['agent_0_reward'])
    plt.show()


list_obs = [
    tensor([[-1.0526e-01, -7.3684e-01, -2.7756e-17, -2.7756e-17, 1.0000e+00,
             1.0000e+00, -1.0000e+00]], device='cuda:0', dtype=torch.float64),
    tensor([[-1.0526e-01, -7.3684e-01, -2.7756e-17, -2.7756e-17, 1.0000e+00,
             1.0000e+00, -1.0000e+00]], device='cuda:0', dtype=torch.float64),
    tensor([[-1.0526e-01, -7.3684e-01, -2.7756e-17, -2.7756e-17, 1.0000e+00,
             1.0000e+00, -1.0000e+00]], device='cuda:0', dtype=torch.float64),
    tensor([[-0.1053, -0.7368, -0.1053, 0.1053, 1.0000, 1.0000, -1.0000]],
           device='cuda:0', dtype=torch.float64),
    tensor([[-0.1053, -0.7368, -0.1579, 0.0526, 1.0000, 1.0000, -1.0000]],
           device='cuda:0', dtype=torch.float64),
    tensor([[-0.1053, -0.7368, -0.0526, 0.1579, 1.0000, 1.0000, -1.0000]],
           device='cuda:0', dtype=torch.float64),
    tensor([[-1.0526e-01, -6.3158e-01, -2.7756e-17, 1.0526e-01, 1.0000e+00,
             1.0000e+00, -1.0000e+00]], device='cuda:0', dtype=torch.float64)
]

obs_features = [f"agent_0_{feature}" for feature in
                ['next_PX', 'next_PY', 'next_VX', 'next_VY', 'next_DX', 'next_DY', 'sensor_on']]

if __name__ == '__main__':
    df = pd.read_pickle(f'{GLOBAL_PATH_REPO}/navigation/causal_knowledge/offline_ok/df_random_{obs}_1000000.pkl')
    df = df.head(n_rows)
    new_df = get_new_df(df)

    cd = CausalDiscovery(new_df, f'navigation/causal_knowledge/offline/navigation', f'agent{agent_n}_{obs}')
    cd.training(cd_algo='pc')
    causal_graph = cd.return_causal_graph()
    df_for_causality = cd.return_df()

    ci = CausalInferenceForRL(df_for_causality, causal_graph,
                              dir_name=f'navigation/causal_knowledge/offline/navigation',
                              env_name=f'agent{agent_n}_{obs}')

    for obs in list_obs:
        obs_dict = {key: obs[0, n].item() for n, key in enumerate(obs_features)}
        reward_action_values = ci.get_rewards_actions_values(obs_dict, online=True)
        print('\n', reward_action_values)
    # causal_table = ci.create_causal_table()

    #causal_table.to_pickle(f'{GLOBAL_PATH_REPO}/navigation/causal_knowledge/offline/navigation/causal_table.pkl')
    #causal_table.to_pickle(f'{GLOBAL_PATH_REPO}/navigation/causal_knowledge/offline/navigation/causal_table.xlsx')
