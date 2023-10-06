import os
import sys
import numpy as np
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
from experiment_params.param_utils import get_params_key
from experiment_params.params import grab_params
from plot_results import get_xs_ys, load_baseline_results


EXPERIMENT_DIR_PREFIX = '../vislang-domain-exploration-data/caption2label-data/Experiments/cc3m_from_LaCLIP_startpoint/experiment_PatchAlignmentMultilabel'
CLIP_ENSEMBLE_BASELINE_EXPERIMENT_DIR = '../vislang-domain-exploration-data/caption2label-data/Experiments/baselines_from_LaCLIP_startpoint/experiment_LaCLIPCheckpointCLIPEnsembleBaseline'
TABLE_PREFIX = '../vislang-domain-exploration-data/caption2label-data/result_plots/patch_aligned_multilabel_laclip_result_table'


IS_DROP_CLF = True


def get_experiment_dir(smart_start, clip_aug, embedder_only, simple_avg, drop_clf):
    experiment_dir = EXPERIMENT_DIR_PREFIX
    if smart_start:
        experiment_dir = experiment_dir + 'SmartStart'
    if clip_aug:
        experiment_dir = experiment_dir + 'ClipAug'
    if embedder_only:
        experiment_dir = experiment_dir + 'EmbedderOnly'
    if simple_avg:
        experiment_dir = experiment_dir + 'SimpleAvg'
    if drop_clf:
        experiment_dir = experiment_dir + 'DropCLF'

    p = grab_params(get_params_key(experiment_dir))
    assert(p.do_patch_alignment)
    assert(p.do_multilabel)
    assert(p.image_embedder_smart_start == smart_start)
    assert(p.image_aug_type == {True : 'clip_A', False : 'simclr_A'}[clip_aug])
    if embedder_only:
        assert(p.image_num_learnable_layers == 0 and p.text_num_learnable_layers == 0)
    else:
        assert(p.image_num_learnable_layers == 1 and p.text_num_learnable_layers == 1)
    assert(p.aggregation_type == {True : 'avg', False : 'avg_drop_percentile'}[simple_avg])
    assert(p.drop_clf_token == drop_clf)

    return experiment_dir


#for non-baseline thingy
#return results, epoch_length
def load_results_one_experiment(smart_start, clip_aug, embedder_only, simple_avg, drop_clf, prevalent_classes_only=False):
    experiment_dir = get_experiment_dir(smart_start, clip_aug, embedder_only, simple_avg, drop_clf)
    print(experiment_dir)
    results_base = 'results.pkl'
    if prevalent_classes_only:
        results_base = 'results_prevalent_classes_only.pkl'

    results_filename = os.path.join(experiment_dir, results_base)
    with open(results_filename, 'rb') as f:
        results = pickle.load(f)

    with open(os.path.join(experiment_dir, 'telemetry.pkl'), 'rb') as f:
        telemetry = pickle.load(f)

    return results, telemetry['epoch_length']


#returns dict mapping from (smart_start, clip_aug, embedder_only, simple_avg) to results
#and also epoch_length
def load_results(prevalent_classes_only=False):
    results_dict = {}
    for smart_start in [False, True]:
        for clip_aug in [False, True]:
            for embedder_only in [False, True]:
                for simple_avg in [False, True]:
                    for drop_clf in [False, True]:
                        results, epoch_length = load_results_one_experiment(smart_start,clip_aug,embedder_only,simple_avg,drop_clf,prevalent_classes_only=prevalent_classes_only)
                        results_dict[(smart_start, clip_aug, embedder_only, simple_avg, drop_clf)] = results

    return results_dict, epoch_length


def make_table(results_dict, epoch_length, baseline_results, dataset_name, prevalent_classes_only=False):
    table_filename = TABLE_PREFIX + '_' + dataset_name + {True : '_prevalent_classes_only', False : ''}[prevalent_classes_only] + '.csv'
    if dataset_name == 'avg':
        assert(len(baseline_results) == 5)
        baseline_y = np.mean([baseline_results[dn]['balanced_accuracy_as_percentage'] for dn in sorted(baseline_results.keys())])
    else:
        baseline_y = baseline_results[dataset_name]['balanced_accuracy_as_percentage']

    f = open(table_filename, 'w')
    already_wrote_first_line = False
    for exp_id in sorted(results_dict.keys()):
        xs, ys = get_xs_ys(results_dict[exp_id], epoch_length, dataset_name)
        if not already_wrote_first_line:
            first_items = ['"hyperparams"'] + ['"%.3f epochs"'%(x) for x in xs] + ['"best_acc"', '"best_epoch"']
            f.write(','.join(first_items) + '\n')
            f.write(','.join(['"CLIP ensembling baseline"'] + ['"%.2f%%"'%(baseline_y) for _ in range(len(xs) + 1)] + ['"N/A"']) + '\n')
            already_wrote_first_line = True

        best_x = None
        best_y = float('-inf')
        for x, y in zip(xs, ys):
            if y > best_y:
                best_y = y
                best_x = x

        my_label = ', '.join([{True : 'SmartStart', False : 'RandomInit'}[exp_id[0]], {True : 'ClipAugs', False : 'SimCLRAugs'}[exp_id[1]], {True : 'EmbedderOnly', False : 'EmbedderAndBackboneTops'}[exp_id[2]], {True : 'SimpleAvg', False : 'DropPercentile'}[exp_id[3]], {True : 'DropCLFToken', False : 'IncludeCLFToken'}[exp_id[4]]])
        f.write(','.join(['"%s"'%(my_label)] + ['"%.2f%%"'%(y) for y in ys] + ['"%.2f%%"'%(best_y)] + ['"%.3f"'%(best_x)]) + '\n')
        
    f.close()


def tablify_results(prevalent_classes_only):
    prevalent_classes_only = bool(int(prevalent_classes_only))

    results_dict, epoch_length = load_results(prevalent_classes_only=prevalent_classes_only)
    baseline_results = load_baseline_results(prevalent_classes_only=prevalent_classes_only)
    for dataset_name in ['DTD', 'Food101', 'Flowers102', 'ImageNet1K', 'UCF101', 'avg']:
        make_table(results_dict, epoch_length, baseline_results, dataset_name, prevalent_classes_only=prevalent_classes_only)


def usage():
    print('Usage: python tablify_results.py <prevalent_classes_only>')


if __name__ == '__main__':
    tablify_results(*(sys.argv[1:]))
