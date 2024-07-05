from navigation.trainer import run_simulations
from path_repo import GLOBAL_PATH_REPO
import pyglet

if __name__ == '__main__':
    pyglet.options['headless'] = True
    path_file_env = f'{GLOBAL_PATH_REPO}/config_simulations/config_env.yaml'
    path_file_algo = f'{GLOBAL_PATH_REPO}/config_simulations/causal_qlearning_online.yaml'
    run_simulations(path_file_env, path_file_algo, True)
