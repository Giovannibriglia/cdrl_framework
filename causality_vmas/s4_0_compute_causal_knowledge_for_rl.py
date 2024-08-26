import json
import pandas as pd

from causality_algos import CausalInferenceForRL
from causality_vmas import LABEL_dir_storing_dict_and_info, LABEL_grouped_features
from causality_vmas.utils import list_to_graph, is_folder_empty, inverse_approximation_function


def main(task: str):
    path_file = f'{LABEL_dir_storing_dict_and_info}_{task}/best'

    obs_train_to_test = inverse_approximation_function(task)

    if not is_folder_empty(path_file):
        dataframe = pd.read_pickle(f'{path_file}/best_df.pkl')

        with open(f'{path_file}/best_causal_graph.json', 'r') as file:
            graph_list = json.load(file)
        causal_graph = list_to_graph(graph_list)

        with open(f'{path_file}/best_bn_params.json', 'r') as file:
            bn_dict = json.load(file)

        with open(f'{path_file}/best_others.json', 'r') as file:
            others = json.load(file)

        grouped_features = others[LABEL_grouped_features]

        if (dataframe is not None and causal_graph is not None) or bn_dict is not None:
            offline_ci = CausalInferenceForRL(online=True, df_train=dataframe, causal_graph=causal_graph, bn_dict=bn_dict,
                                              causal_table=None, obs_train_to_test=obs_train_to_test,
                                              grouped_features=grouped_features)
            causal_table = offline_ci.create_causal_table()

            causal_table.to_pickle(f'{path_file}/causal_table.pkl')
        else:
            print('some files are empty')
    else:
        print('there are no best')


if __name__ == '__main__':
    task_name = input('Select task: ')
    main(task_name)
