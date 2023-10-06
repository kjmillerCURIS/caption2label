import os
import sys


DEBUG = False
JOB_NAME_FILTER = None
IS_LACLIP = True #change to False to rerun experiments that started from OpenAI checkpoint
EXPERIMENT_BASE_DIR = '../vislang-domain-exploration-data/caption2label-data/Experiments'
PARAM_BASENAME = 'PatchAlignmentMultilabel'


def s_or_empty(flag, s):
    return s if flag else ''


def submit_patch_alignment_multilabel_eval_runs():
    for smart_start in [False, True]:
        for clip_aug in [False, True]:
            for embedder_only in [False, True]:
                for simple_avg in [False, True]:
                    for drop_clf in [True]:
                        camel_suffix = s_or_empty(smart_start, 'SmartStart') + s_or_empty(clip_aug, 'ClipAug') + s_or_empty(embedder_only, 'EmbedderOnly') + s_or_empty(simple_avg, 'SimpleAvg') + s_or_empty(drop_clf, 'DropCLF')
                        snake_suffix = s_or_empty(smart_start, '_smart_start') + s_or_empty(clip_aug, '_clip_aug') + s_or_empty(embedder_only, '_embedder_only') + s_or_empty(simple_avg, '_simple_avg') + s_or_empty(drop_clf, '_drop_clf')
                        experiment_dir = os.path.join(EXPERIMENT_BASE_DIR, {False : 'cc3m', True : 'cc3m_from_LaCLIP_startpoint'}[IS_LACLIP], 'experiment_' + PARAM_BASENAME + camel_suffix)
                        job_name = 'eval_clip_with_patch_alignment_and_multilabel' + s_or_empty(IS_LACLIP, '_laclip') + snake_suffix
                        if JOB_NAME_FILTER is not None:
                            if job_name not in JOB_NAME_FILTER:
                                continue

                        my_cmd = 'qsub -N %s -v EXPERIMENT_DIR=%s,IS_LACLIP=%d run_eval_patch_alignment_with_multilabel_generic.sh'%(job_name, experiment_dir, int(IS_LACLIP))
                        print('submitting eval run: "%s"'%(my_cmd))
                        os.system(my_cmd)
                        if DEBUG:
                            print('DEBUG MODE: let\'s see how that first run goes...')
                            return


if __name__ == '__main__':
    submit_patch_alignment_multilabel_eval_runs()
