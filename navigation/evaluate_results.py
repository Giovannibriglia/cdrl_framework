import json
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from navigation.utils import compute_avg_series_for_agent, IQM_mean_std

metrics = ['actions', 'rewards', 'time', 'n_collisions']


def main(path_metrics):
    with open(f'{path_metrics}', 'r') as f:
        dict_metrics = json.load(f)

    agents_iqm_results = {key: {} for key in metrics}
    for metric in metrics:
        fig = plt.figure(dpi=500, figsize=(16, 9))
        fig.suptitle(f'{metric}', fontsize=22)
        for agent_key, agent_data in dict_metrics.items():
            agent_avg_series = compute_avg_series_for_agent(agent_data, metric)
            agent_iqm_mean, agent_iqm_std = IQM_mean_std(agent_avg_series)
            agents_iqm_results[metric][agent_key] = {
                'iqm_mean': agent_iqm_mean,
                'iqm_std': agent_iqm_std
            }
            plt.plot(gaussian_filter1d(agent_avg_series, sigma=8))
        plt.show()

    for key, values in agents_iqm_results.items():
        print(f'{key} --> {values}')


if __name__ == '__main__':
    path_file_metrics = 'C:\\Users\giova\Documents\Research\cdrl_framework\\navigation\\results\\dqn_pomdp.json'
    main(path_file_metrics)

    # TODO: manage other metrics
    # TODO: plots across different agents - same algo and plots across same agent - different algos
    # TODO: tables
