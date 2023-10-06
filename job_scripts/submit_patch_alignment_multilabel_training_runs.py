import os
import sys
sys.path.append('..')
from setup_experiment_dir import setup_experiment_dir


DEBUG = False
IS_LACLIP = True #change to False to rerun experiments that started from OpenAI checkpoint
EXPERIMENT_BASE_DIR = '../vislang-domain-exploration-data/caption2label-data/Experiments'
PARAM_BASENAME = 'PatchAlignmentMultilabel'


def s_or_empty(flag, s):
    return s if flag else ''


def submit_patch_alignment_multilabel_training_runs(just_this_one_job=None):
    for smart_start in [False, True]:
        for clip_aug in [False, True]:
            for embedder_only in [False, True]:
                for simple_avg in [False, True]:
                    for drop_clf in [True]:
                        camel_suffix = s_or_empty(smart_start, 'SmartStart') + s_or_empty(clip_aug, 'ClipAug') + s_or_empty(embedder_only, 'EmbedderOnly') + s_or_empty(simple_avg, 'SimpleAvg') + s_or_empty(drop_clf, 'DropCLF')
                        snake_suffix = s_or_empty(smart_start, '_smart_start') + s_or_empty(clip_aug, '_clip_aug') + s_or_empty(embedder_only, '_embedder_only') + s_or_empty(simple_avg, '_simple_avg') + s_or_empty(drop_clf, '_drop_clf')
                        experiment_dir = os.path.join(EXPERIMENT_BASE_DIR, {False : 'cc3m', True : 'cc3m_from_LaCLIP_startpoint'}[IS_LACLIP], 'experiment_' + PARAM_BASENAME + camel_suffix)
                        job_name = 'train_clip_with_patch_alignment_and_multilabel' + s_or_empty(IS_LACLIP, '_laclip') + snake_suffix
                        if just_this_one_job is not None and job_name != just_this_one_job:
                            continue

                        os.makedirs(os.path.join('..', experiment_dir), exist_ok=True)
                        setup_experiment_dir(os.path.join('..', experiment_dir), PARAM_BASENAME + camel_suffix + 'Params')
                        my_cmd = 'qsub -N %s -v EXPERIMENT_DIR=%s,IS_LACLIP=%d run_train_clip_with_patch_alignment_and_multilabel_generic.sh'%(job_name, experiment_dir, int(IS_LACLIP))
                        print('submitting training run: "%s"'%(my_cmd))
                        os.system(my_cmd)
                        if DEBUG:
                            print('DEBUG MODE: let\'s see how that first run goes...')
                            return


if __name__ == '__main__':
    submit_patch_alignment_multilabel_training_runs(*(sys.argv[1:]))
