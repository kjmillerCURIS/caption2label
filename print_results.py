import os
import sys
import pickle


def print_results(experiment_dir):
    if os.path.isdir(experiment_dir):
        with open(os.path.join(experiment_dir, 'results.pkl'), 'rb') as f:
            results_dict = pickle.load(f)
    else:
        with open(experiment_dir, 'rb') as f:
            results_dict = pickle.load(f)

    for kA in sorted(results_dict.keys()):
        if 'balanced_accuracy_as_percentage' in results_dict[kA]:
            print('results_dict[%s]["balanced_accuracy_as_percentage"] = %s'%(str(kA), results_dict[kA]['balanced_accuracy_as_percentage']))
            print('results_dict[%s]["unbalanced_accuracy_as_percentage"] = %s'%(str(kA), results_dict[kA]['unbalanced_accuracy_as_percentage']))
        else:
            for kB in sorted(results_dict[kA].keys()):
                print('results_dict[%s][%s]["balanced_accuracy_as_percentage"] = %s'%(str(kA), str(kB), results_dict[kA][kB]['balanced_accuracy_as_percentage']))
                print('results_dict[%s][%s]["unbalanced_accuracy_as_percentage"] = %s'%(str(kA), str(kB), results_dict[kA][kB]['unbalanced_accuracy_as_percentage']))


def usage():
    print('Usage: python print_results.py <experiment_dir>')


if __name__ == '__main__':
    print_results(*(sys.argv[1:]))
