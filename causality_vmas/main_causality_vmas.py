import json

import pandas as pd

from causality_vmas.s1_0_extract_df_task_vmas import VMASExperiment
from causality_vmas.s2_0_sensitive_analysis import SensitiveAnalysis
from causality_vmas.s3_0_find_best_approximation import BestApprox
from causality_vmas.s4_0_compute_causal_knowledge_for_rl import CausalInferenceForRL
from causality_vmas.utils import list_to_graph, is_folder_empty, inverse_approximation_function


def main(task_name: str = 'navigation'):
    print('Task: ', task_name)
    experiment = VMASExperiment(task_name)
    df_start, info_task, _ = experiment.run_experiment()

    agent0_columns = [col for col in df_start.columns if 'agent_0' in col]
    df_start = df_start.loc[:, agent0_columns]

    sensitive_analysis = SensitiveAnalysis(df_start, task_name, info_task)
    path_results = sensitive_analysis.computing_CIQs()

    best_approx = BestApprox(path_results, df_start)
    best_approx.evaluate()
    best_approx.plot_results(if_save=True)

    path_file = f'{path_results}/best'

    if not is_folder_empty(path_file):
        df_approx = pd.read_pickle(f'{path_file}/best_df.pkl')

        with open(f'{path_file}/best_causal_graph.json', 'r') as file:
            graph_list = json.load(file)
        causal_graph = list_to_graph(graph_list)

        with open(f'{path_file}/best_bn_params.json', 'r') as file:
            bn_dict = json.load(file)

        online = False
        causal_table = None
        obs_train_to_test = inverse_approximation_function(task_name)

        if (df_approx is not None and causal_graph is not None) or bn_dict is not None:
            offline_ci = CausalInferenceForRL(online, df_approx, causal_graph, bn_dict, causal_table,
                                              df_start, obs_train_to_test)
            ct = offline_ci.create_causal_table(show_progress=True)
            ct.to_pickle(f'{path_file}/causal_table.pkl')
            ct.to_excel('mario.xlsx')
        else:
            print('some files are empty')
    else:
        print('there are no best')


if __name__ == '__main__':
    main()
