from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple
import pandas as pd
from tqdm import tqdm

from causality_vmas.causality_algos import CausalDiscovery, SingleCausalInference
from causality_vmas.utils import get_df_boundaries, constraints_causal_graph, bn_to_dict, \
    graph_to_list, _navigation_approximation
from causality_vmas import abs_path_causality_vmas, LABEL_causal_graph, LABEL_bn_dict, LABEL_target_value, \
    LABEL_predicted_value

show_progress_cd = True


class CausalityInformativenessQuantification:
    def __init__(self, df: pd.DataFrame, target_feature: str = 'action', cd_algo: str = 'PC'):
        self.df = df
        self.cd_algo = cd_algo

        self.causal_graph = None

        self.action_col_name = str([s for s in df.columns.to_list() if 'action' in s][0])
        if not self.action_col_name:
            raise ValueError('action column did not find')
        self.reward_col_name = str([s for s in df.columns.to_list() if 'reward' in s][0])
        if not self.reward_col_name:
            raise ValueError('reward column did not find')

        if target_feature == 'action':
            self.target_feature = self.action_col_name
        elif target_feature == 'reward':
            self.target_feature = self.reward_col_name
        else:
            raise ValueError('target feature is not correct')

    def evaluate(self) -> Tuple[Dict, Dict]:
        self.cd_process()
        res_score, res_causal = self.ci_assessment(show_progress=True)

        return res_score, res_causal

    def cd_process(self):
        cd = CausalDiscovery(self.df)
        cd.training(cd_algo=self.cd_algo, show_progress=show_progress_cd)
        self.causal_graph = cd.return_causal_graph()
        self.causal_graph = constraints_causal_graph(self.causal_graph)

    def ci_assessment(self, show_progress: bool = False) -> Tuple[Dict, Dict]:
        single_ci = SingleCausalInference(self.df, self.causal_graph)

        cbn = single_ci.return_cbn()
        cbn_in_dict = bn_to_dict(cbn)

        causality_dict = {LABEL_causal_graph: graph_to_list(self.causal_graph), LABEL_bn_dict: cbn_in_dict}
        res_score_dict = {LABEL_target_value: [], LABEL_predicted_value: []}

        selected_columns = [s for s in self.df.columns.to_list() if s != self.target_feature]

        tasks = list(self.df.iterrows())

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._process_row, task, selected_columns, self.target_feature, single_ci) for
                       task in tasks]
            if show_progress:
                for future in tqdm(as_completed(futures), total=len(futures), desc=f'Inferring causal knowledge...'):
                    target_value, pred_value = future.result()
                    res_score_dict[LABEL_target_value].append(target_value)
                    res_score_dict[LABEL_predicted_value].append(pred_value)
            else:
                for future in as_completed(futures):
                    target_value, pred_value = future.result()
                    res_score_dict[LABEL_target_value].append(target_value)
                    res_score_dict[LABEL_predicted_value].append(pred_value)

        return res_score_dict, causality_dict

    @staticmethod
    def _process_row(index_row, selected_columns, target_feature, single_ci) -> Tuple[float, float]:
        index, row = index_row
        input_ci = row[selected_columns].to_dict()
        output_ci = single_ci.infer(input_ci, target_feature)
        value_target = row[target_feature]
        value_pred = max(output_ci, key=output_ci.get)
        value_pred = type(value_target)(value_pred)
        return value_target, value_pred


if __name__ == '__main__':
    df = pd.read_pickle(f'{abs_path_causality_vmas}/causality_vmas/dataframes/navigation_mdp.pkl')

    agent0_columns = [col for col in df.columns if 'agent_0' in col]
    df = df.loc[:20001, agent0_columns]
    dict_approx = _navigation_approximation((df, {'BINS': 10, 'SENSORS': 1, 'ROWS': 20000}))
    df = dict_approx['new_df']
    # get_df_boundaries(df)

    ciq = CausalityInformativenessQuantification(df)
    print(ciq.evaluate())
