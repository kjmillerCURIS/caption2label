import os
import sys
import numpy as np
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
from experiment_params.param_utils import get_params_key
from experiment_params.params import grab_params


EXPERIMENT_DIR_PREFIX = '../vislang-domain-exploration-data/caption2label-data/Experiments/cc3m_from_LaCLIP_startpoint/experiment_PatchAlignmentMultilabel'
CLIP_ENSEMBLE_BASELINE_EXPERIMENT_DIR = '../vislang-domain-exploration-data/caption2label-data/Experiments/baselines_from_LaCLIP_startpoint/experiment_LaCLIPCheckpointCLIPEnsembleBaseline'
PLOT_PREFIX = '../vislang-domain-exploration-data/caption2label-data/result_plots/patch_aligned_multilabel_laclip_result_plot'
LINESTYLE_DICT = {True : 'dashed', False : 'solid'}
COLOR_DICT = {(False, False, False) : 'r',
                (False, False, True) : 'limegreen',
                (False, True, False) : 'b',
                (False, True, True) : 'm',
                (True, False, False) : 'y',
                (True, False, True) : 'c',
                (True, True, False) : 'pink',
                (True, True, True) : 'gray'}
BASELINE_COLOR = 'k'
BASELINE_LINESTYLE = 'solid'


IS_DROP_CLF = True


def get_experiment_dir(smart_start, clip_aug, embedder_only, simple_avg):
    experiment_dir = EXPERIMENT_DIR_PREFIX
    if smart_start:
        experiment_dir = experiment_dir + 'SmartStart'
    if clip_aug:
        experiment_dir = experiment_dir + 'ClipAug'
    if embedder_only:
        experiment_dir = experiment_dir + 'EmbedderOnly'
    if simple_avg:
        experiment_dir = experiment_dir + 'SimpleAvg'

    if IS_DROP_CLF:
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
    assert(p.drop_clf_token == IS_DROP_CLF)

    return experiment_dir


#for non-baseline thingy
#return results, epoch_length
def load_results_one_experiment(smart_start, clip_aug, embedder_only, simple_avg, prevalent_classes_only=False):
    experiment_dir = get_experiment_dir(smart_start, clip_aug, embedder_only, simple_avg)
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
                    results, epoch_length = load_results_one_experiment(smart_start,clip_aug,embedder_only,simple_avg,prevalent_classes_only=prevalent_classes_only)
                    results_dict[(smart_start, clip_aug, embedder_only, simple_avg)] = results

    return results_dict, epoch_length


def load_baseline_results(prevalent_classes_only=False):
    results_filename = os.path.join(CLIP_ENSEMBLE_BASELINE_EXPERIMENT_DIR, {True:'results_prevalent_classes_only.pkl', False:'results.pkl'}[prevalent_classes_only])
    with open(results_filename, 'rb') as f:
        baseline_results = pickle.load(f)

    return baseline_results


def get_xs_ys(results, epoch_length, dataset_name):
    k_list = sorted(results.keys())
    xs = []
    ys = []
    for k in k_list:
        xs.append(k[0] + k[1] / epoch_length)
        if dataset_name == 'avg':
            assert(len(results[k]) == 5)
            ys.append(np.mean([results[k][dn]['balanced_accuracy_as_percentage'] for dn in sorted(results[k].keys())]))
        else:
            ys.append(results[k][dataset_name]['balanced_accuracy_as_percentage'])

    return xs, ys


def make_plot(results_dict, epoch_length, baseline_results, dataset_name, prevalent_classes_only=False, is_legend=False):
    plt.clf()
    plt.figure(figsize=[16,12])
    for exp_id in sorted(results_dict.keys()):
        linestyle = LINESTYLE_DICT[exp_id[0]]
        color = COLOR_DICT[exp_id[1:]]
        xs, ys = get_xs_ys(results_dict[exp_id], epoch_length, dataset_name)
        my_label = ', '.join([{True : 'SmartStart', False : 'RandomInit'}[exp_id[0]], {True : 'ClipAugs', False : 'SimCLRAugs'}[exp_id[1]], {True : 'EmbedderOnly', False : 'EmbedderAndBackboneTops'}[exp_id[2]], {True : 'SimpleAvg', False : 'DropPercentile'}[exp_id[3]]])
        plt.plot(xs, ys, linestyle=linestyle, color=color, label=my_label, marker='o')

    my_xlim = plt.xlim()
    if dataset_name == 'avg':
        assert(len(baseline_results) == 5)
        baseline_y = np.mean([baseline_results[dn]['balanced_accuracy_as_percentage'] for dn in sorted(baseline_results.keys())])
    else:
        baseline_y = baseline_results[dataset_name]['balanced_accuracy_as_percentage']

    plt.plot(my_xlim, [baseline_y, baseline_y], color=BASELINE_COLOR, linestyle=BASELINE_LINESTYLE, label='CLIP ensemble baseline')
    plt.xlim(my_xlim)
    #plt.ylim((0,100))
    if not is_legend:
        plt.title('%s%s%s'%(dataset_name, {True : ', prevalent classes only', False : ''}[prevalent_classes_only], {True : ', Drop CLF token', False : ''}[IS_DROP_CLF]))

    plt.xlabel('epoch')
    plt.ylabel('accuracy (%)')
    if is_legend:
        plt.legend(framealpha=1)

    os.makedirs(os.path.dirname(PLOT_PREFIX), exist_ok=True)
    if is_legend:
        plot_filename = PLOT_PREFIX + '_LEGEND.png'
    else:
        plot_filename = PLOT_PREFIX + '_' + dataset_name + {True : '_prevalent_classes_only', False : ''}[prevalent_classes_only] + {True : '_drop_clf_token', False : ''}[IS_DROP_CLF] + '.png'

    plt.savefig(plot_filename)
    plt.clf()


def plot_results(prevalent_classes_only):
    prevalent_classes_only = bool(int(prevalent_classes_only))

    results_dict, epoch_length = load_results(prevalent_classes_only=prevalent_classes_only)
    baseline_results = load_baseline_results(prevalent_classes_only=prevalent_classes_only)
    for dataset_name in ['DTD', 'Food101', 'Flowers102', 'ImageNet1K', 'UCF101', 'avg']:
        make_plot(results_dict, epoch_length, baseline_results, dataset_name, prevalent_classes_only=prevalent_classes_only)

    make_plot(results_dict, epoch_length, baseline_results, 'avg', prevalent_classes_only=prevalent_classes_only, is_legend=True)

def usage():
    print('Usage: python plot_results.py <prevalent_classes_only>')


if __name__ == '__main__':
    plot_results(*(sys.argv[1:]))
