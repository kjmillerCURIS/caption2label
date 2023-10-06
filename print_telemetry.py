import os
import sys
import pickle

def print_telemetry(experiment_dir):
    with open(os.path.join(experiment_dir, 'telemetry.pkl'), 'rb') as f:
        telemetry = pickle.load(f)

    print(telemetry['train_losses'])


def usage():
    print('Usage: python print_telemetry.py <experiment_dir>')


if __name__ == '__main__':
    print_telemetry(*(sys.argv[1:]))
