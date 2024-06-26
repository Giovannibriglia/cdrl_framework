import pandas as pd
from path_repo import GLOBAL_PATH_REPO
from navigation.causality_algos import CausalDiscovery

df = pd.read_pickle(f'{GLOBAL_PATH_REPO}/navigation/causal_knowledge/offline/df_random_pomdp_1000000.pkl')
agent_n = 0

agent_X_columns = [s for s in df.columns.to_list() if f'agent_{agent_n}' in s]

df = df.loc[:, agent_X_columns]
std_dev = df.std()
non_zero_std_columns = std_dev[std_dev != 0].index
df_filtered = df[non_zero_std_columns]
print(df_filtered)
print(df['agent_0_action'].mean(), df['agent_0_action'].std())
cd = CausalDiscovery(df_filtered, f'causal_knowledge/offline/navigation', f'agent{agent_n}_pomdp')
cd.training()
