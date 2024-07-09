import pandas as pd
from torch import tensor
import torch
from navigation.utils import discretize_df, get_df_boundaries
from path_repo import GLOBAL_PATH_REPO
from navigation.causality_algos import CausalDiscovery, CausalInferenceForRL

n_bins = 10
agent_n = 0
n_rows = 50000
obs = 'mdp'
num_sensors = 1


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
                ['next_PX', 'next_PY', 'next_VX', 'next_VY', 'next_DX', 'next_DY'] +
                [f'sensor_on_{n}' for n in range(num_sensors)]]

if __name__ == '__main__':
    df = pd.read_pickle(f'{GLOBAL_PATH_REPO}/navigation/causal_knowledge/offline_ok/df_random_{obs}_1000000.pkl')
    df = df.head(n_rows)

    agent_X_columns = [s for s in df.columns.to_list() if f'agent_{agent_n}' in s]

    dataframe = df.loc[:, agent_X_columns]

    new_df = discretize_df(df, n_bins, num_sensors, 0.5, 0.5)

    cd = CausalDiscovery(new_df, f'navigation/causal_knowledge/offline/navigation', f'agent{agent_n}_{obs}')
    cd.training(cd_algo='pc')
    causal_graph = cd.return_causal_graph()
    df_for_causality = cd.return_df()

    ci = CausalInferenceForRL(df_for_causality, causal_graph,
                              dir_name=f'navigation/causal_knowledge/offline/navigation',
                              env_name=f'agent{agent_n}_{obs}')

    for obs in list_obs:
        obs_dict = {key: obs[0, n].item() for n, key in enumerate(obs_features)}
        reward_action_values = ci.get_actions_rewards_values(obs_dict, online=True)
        print('\n', reward_action_values)
    """causal_table = ci.create_causal_table()

    causal_table.to_pickle(f'{GLOBAL_PATH_REPO}/navigation/causal_knowledge/offline/navigation/causal_table.pkl')
    causal_table.to_pickle(f'{GLOBAL_PATH_REPO}/navigation/causal_knowledge/offline/navigation/causal_table.xlsx')"""
