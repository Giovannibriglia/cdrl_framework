import pandas as pd
import json

from causality_vmas import LABEL_dict_causality, LABEL_bn_dict, LABEL_causal_graph
from causality_vmas.s2_0_sensitive_analysis import SensitiveAnalysis
from causality_vmas.s_3_find_best_approximation import BestApprox
from causality_vmas.s_4_compute_causal_knowledge_for_rl import OfflineCausalInferenceForRL
from causality_vmas.utils import list_to_graph
from path_repo import GLOBAL_PATH_REPO


def main(task_name: str = 'navigation'):
    # TODO: obtain df
    df = pd.read_pickle(f'{GLOBAL_PATH_REPO}/causality_vmas/dataframes/{task_name}_mdp.pkl')
    agent0_columns = [col for col in df.columns if 'agent_0' in col]
    df = df.loc[:100001, agent0_columns]

    sensitive_analysis = SensitiveAnalysis(df, task_name)
    results, path_results = sensitive_analysis.computing_CIQs()

    best_approx = BestApprox(path_results)
    best_approx.evaluate(True)
    best_df, best_info = best_approx.get_best()
    best_approx.plot_results()

    graph_list = best_info[LABEL_dict_causality][LABEL_causal_graph]
    graph = list_to_graph(graph_list)
    bn_dict = best_info[LABEL_dict_causality][LABEL_bn_dict]

    offline_ci = OfflineCausalInferenceForRL(best_df, graph, bn_dict)
    ct = offline_ci.create_causal_table(show_progress=True)
    print(ct)


if __name__ == '__main__':
    main()
