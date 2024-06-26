import json
from navigation.utils import compute_iqm_and_std_for_agent


def main(path_metrics):
    with open(f'{path_metrics}', 'r') as f:
        dict_metrics = json.load(f)

    agent_iqm_results = {}
    for agent_key, agent_data in dict_metrics.items():
        agent_iqm_mean, agent_iqm_std = compute_iqm_and_std_for_agent(agent_data, 'rewards')
        agent_iqm_results[agent_key] = {
            'iqm_mean': agent_iqm_mean,
            'iqm_std': agent_iqm_std
        }
    """mean = 0
    for agent_key, results in agent_iqm_results.items():
        mean += results['iqm_mean']
        print(f"Agent {agent_key}: IQM Mean = {results['iqm_mean']}, IQM Std = {results['iqm_std']}")
    print(mean / 4)"""


if __name__ == '__main__':
    path_file_metrics = 'C:\\Users\giova\Documents\Research\cdrl_framework\\navigation\\results\dqn_mdp.json'
    main(path_file_metrics)
    print('***************')
    path_file_metrics = 'C:\\Users\giova\Documents\Research\cdrl_framework\\navigation\\results\qlearning_mdp.json'
    main(path_file_metrics)
    print('***************')
    path_file_metrics = 'C:\\Users\giova\Documents\Research\cdrl_framework\\navigation\\results\completely_causal_mdp.json'
    main(path_file_metrics)
    print('***************')
    path_file_metrics = 'C:\\Users\giova\Documents\Research\cdrl_framework\\navigation\\results\\random_mdp.json'
    main(path_file_metrics)

    # TODO: fix mean and std
    # TODO: plots
    # TODO: tables
