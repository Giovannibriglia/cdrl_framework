from navigation.utils import evaluate_causal_graphs
from path_repo import GLOBAL_PATH_REPO

path_folder = f'{GLOBAL_PATH_REPO}/navigation/additional_assessments'

path_results_causal_graphs = f'{path_folder}/causal_graphs_pc'

df_results = evaluate_causal_graphs(path_results_causal_graphs)

df_results.to_excel(f'{path_folder}/res_sensitive_analysis.xlsx')
df_results.to_pickle(f'{path_folder}/res_sensitive_analysis.pkl')
