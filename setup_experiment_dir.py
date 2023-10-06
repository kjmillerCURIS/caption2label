import os
import sys
import pickle


def setup_experiment_dir(experiment_dir, params_key):
    if os.path.exists(os.path.join(experiment_dir, 'params_key.pkl')):
        print('params_key.pkl already exists!')
        print('will double-check that params_key matches existing one, then continue...')
        with open(os.path.join(experiment_dir, 'params_key.pkl'), 'rb') as f:
            existing_params_key = pickle.load(f)

        if existing_params_key == params_key:
            print('yep, they match, have fun with the rest of your experiment!')
            return
        else:
            print('oh no! they don\'t match! terminating!')
            assert(False)

    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, 'params_key.pkl'), 'wb') as f:
        pickle.dump(params_key, f)


def usage():
    print('Usage: python setup_experiment_dir.py <experiment_dir> <params_key>')


if __name__ == '__main__':
    setup_experiment_dir(*(sys.argv[1:]))
