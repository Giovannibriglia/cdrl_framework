import pandas as pd
from navigation.utils import evaluate_results_sensitive_analysis_causal_graphs
from path_repo import GLOBAL_PATH_REPO

cd_algo = 'pc'
N_ROWS = 200000


path_folder = f'{GLOBAL_PATH_REPO}/navigation/additional_assessments'

path_results_causal_graphs = f'{path_folder}/causal_graphs_{cd_algo}'

dataframe = pd.read_pickle(f'{GLOBAL_PATH_REPO}/navigation/causal_knowledge/offline_ok/df_random_mdp_1000000.pkl')
columns_agent0 = [s for s in dataframe.columns.to_list() if 'agent_0' in s]
dataframe = dataframe.loc[:N_ROWS, columns_agent0]

df_results = evaluate_results_sensitive_analysis_causal_graphs(path_results_causal_graphs, dataframe, 0.5, 0.5)


