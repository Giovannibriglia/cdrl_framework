import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from causality_vmas.additional_assessments.extract_metrics import MetricsBetweenTwoGraphs, MetricsOnTheGraph
from causality_vmas.utils import list_to_causal_graph
from path_repo import GLOBAL_PATH_REPO
import os
import json


class CompareGraphsManager:
    def __init__(self, directory_path):
        self.directory_path = directory_path

    def load_json_files(self):
        json_files = [f for f in os.listdir(self.directory_path) if f.endswith('.json')]
        data = {}
        for json_file in json_files:
            with open(os.path.join(self.directory_path, json_file), 'r') as f:
                data[json_file] = json.load(f)
        return data

    def find_ground_truth(self, data):
        for file_name, content in data.items():
            if 'ground_truth' in file_name:
                return file_name, content
        return None, None

    def load_dataframe(self, n_bins: int, n_sensors: int):
        if n_sensors == 0:
            df_filename = f"ground_truth_df_{n_bins}bins.pkl"
            df_path = os.path.join(self.directory_path, df_filename)
            if os.path.exists(df_path):
                return pd.read_pickle(df_path)
            else:
                print(f"Dataframe file {df_filename} not found.")
                return None
        else:
            df_filename = f"df_discretized_{n_bins}bins_{n_sensors}sensors.pkl"
            df_path = os.path.join(self.directory_path, df_filename)
            if os.path.exists(df_path):
                return pd.read_pickle(df_path)
            else:
                print(f"Dataframe file {df_filename} not found.")
                return None

    def compare_graphs_with_ground_truth(self):
        data = self.load_json_files()
        ground_truth_file, ground_truth_content = self.find_ground_truth(data)

        if not ground_truth_content:
            print("No ground truth file found.")
            return

        causal_graph_ground_truth = list_to_causal_graph(ground_truth_content['causal_graph'])
        n_bins_gt = ground_truth_content['n_bins']
        if not nx.is_directed_acyclic_graph(causal_graph_ground_truth):
            self._plot_causal_graph(causal_graph_ground_truth, n_bins_gt, 0)

        df_ground_truth = self.load_dataframe(n_bins_gt, 0)

        if df_ground_truth is None:
            print("Ground truth dataframe not found.")
            return

        all_results = []
        pbar = tqdm(data.items())
        for file_name, content in pbar:
            causal_graph_pred = list_to_causal_graph(content['causal_graph'])
            n_bins_pred = content['n_bins']
            n_sensors_pred = content['n_sensors']

            if not nx.is_directed_acyclic_graph(causal_graph_pred):
                self._plot_causal_graph(causal_graph_pred, n_bins_pred, n_sensors_pred)
            df_pred = self.load_dataframe(n_bins_pred, n_sensors_pred)

            if df_pred is not None:
                comparator = MetricsBetweenTwoGraphs(causal_graph_ground_truth, causal_graph_pred, df_ground_truth,
                                                     df_pred)
                result = comparator.compare()
                result.update(content)
                all_results.append(result)

        return all_results

    def evaluate_single_graphs_metrics(self):
        data = self.load_json_files()

        all_results = []

        ground_truth_file, ground_truth_content = self.find_ground_truth(data)

        if ground_truth_content:
            causal_graph_ground_truth = list_to_causal_graph(ground_truth_content['causal_graph'])
            n_bins_gt = ground_truth_content['n_bins']
            df_ground_truth = self.load_dataframe(n_bins_gt, 0)
            if df_ground_truth is not None:
                comparator = MetricsOnTheGraph(df_ground_truth, causal_graph_ground_truth)

                result = comparator.assessment()
                result.update(ground_truth_content)
                all_results.append(result)

        pbar = tqdm(data.items())
        for file_name, content in pbar:

            causal_graph = list_to_causal_graph(content['causal_graph'])
            n_bins_other = content['n_bins']
            n_sensors_other = content.get('n_sensors', 0)
            df = self.load_dataframe(n_bins_other, n_sensors_other)

            if df is not None:
                comparator = MetricsOnTheGraph(df, causal_graph)
                result = comparator.assessment()
                result.update(content)

                all_results.append(result)

        return all_results

    @staticmethod
    def _plot_causal_graph(causal_graph: nx.DiGraph, n_bins: int, n_sensors: int = 0):
        import warnings
        warnings.filterwarnings("ignore")

        FONT_SIZE_NODE_GRAPH = 7
        ARROWS_SIZE_NODE_GRAPH = 30
        NODE_SIZE_GRAPH = 1000

        fig = plt.figure(dpi=1000)
        if n_sensors == 0:
            plt.title('Ground Truth', fontsize=16)
        else:
            plt.title(f'{n_bins} bins - {n_sensors} sensors', fontsize=16)

        nx.draw(causal_graph, with_labels=True, font_size=FONT_SIZE_NODE_GRAPH,
                arrowsize=ARROWS_SIZE_NODE_GRAPH,
                arrows=True,
                edge_color='orange', node_size=NODE_SIZE_GRAPH, font_weight='bold', node_color='skyblue',
                pos=nx.circular_layout(causal_graph))

        plt.show()


def main():
    path_results = './results_pc'
    assessment_sensitive_analysis = CompareGraphsManager(path_results)

    """results_between_graphs_and_ground_truth = assessment_sensitive_analysis.compare_graphs_with_ground_truth()
    df_res_between_graphs_and_ground_truth = pd.DataFrame(results_between_graphs_and_ground_truth)
    df_res_between_graphs_and_ground_truth.to_excel('res_between_graphs_and_ground_truth.xlsx')"""

    results_on_the_graphs = assessment_sensitive_analysis.evaluate_single_graphs_metrics()
    df_res_on_the_graphs = pd.DataFrame(results_on_the_graphs)
    df_res_on_the_graphs.to_excel('res_on_the_graphs.xlsx')


if __name__ == '__main__':
    main()
