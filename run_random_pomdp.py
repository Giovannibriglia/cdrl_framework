from navigation.trainer import run_simulations
from path_repo import GLOBAL_PATH_REPO
import pyglet

if __name__ == '__main__':
    pyglet.options['headless'] = True
    path_file = f'{GLOBAL_PATH_REPO}/config_simulations/random_pomdp.yaml'
    run_simulations(path_file, False)
