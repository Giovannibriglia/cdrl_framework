import json
import pandas as pd

from causality_algos import CausalInferenceForRL
from causality_vmas import LABEL_dir_storing_dict_and_info
from causality_vmas.utils import list_to_graph, is_folder_empty, inverse_approximation_function


def main(task: str = 'navigation'):
    df_test = pd.read_pickle('./dataframes/df_navigation_pomdp_discrete_actions_0.pkl')
    agent0_columns = [col for col in df_test.columns if 'agent_0' in col]
    df_test = df_test.loc[:, agent0_columns]

    path_file = f'{LABEL_dir_storing_dict_and_info}_{task}/best'

    obs_train_to_test = inverse_approximation_function(task)

    if not is_folder_empty(path_file):
        dataframe = pd.read_pickle(f'{path_file}/best_df.pkl')

        with open(f'{path_file}/best_causal_graph.json', 'r') as file:
            graph_list = json.load(file)
        causal_graph = list_to_graph(graph_list)

        with open(f'{path_file}/best_bn_params.json', 'r') as file:
            bn_dict = json.load(file)

        if (dataframe is not None and causal_graph is not None) or bn_dict is not None:
            offline_ci = CausalInferenceForRL(False, dataframe, causal_graph, bn_dict,
                                              None, df_test, obs_train_to_test)
            ct = offline_ci.create_causal_table(show_progress=True)
            ct.to_pickle(f'{path_file}/causal_table.pkl')
            ct.to_excel('mario.xlsx')
        else:
            print('some files are empty')
    else:
        print('there are no best')


if __name__ == '__main__':
    main()
