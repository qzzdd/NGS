import argparse
import itertools
import subprocess
import random
import time

import numpy as np

"""
    Train NGS, and report the average score and standard deviation
"""

# === define paras ==================
parser = argparse.ArgumentParser()
random.seed(0)
seed = list(random.sample(range(0, 50), 10))
parser.add_argument('--steps', type=int, nargs='+', help='number of intermediate states in the meta graph')
parser.add_argument('--dataset', type=str, default='amazon', help='The dataset name. [yelp, amazon, mimic]')
args = parser.parse_args()


score = []; score_ad = []
# === run all case ===========
para_names = ['seed']
cmds = []
for values in itertools.product(seed):

    cmd_search = f'python train_search_ngs.py'
    cmd_eval = f'python train.py'
    for p, v in zip(para_names, values):
        cmd_search += f' --{p}={v}'
        cmd_eval += f' --{p}={v}'
    #cmd += ' --steps 3 4 5'
    cmd_search += ' --steps {}'.format(' '.join(list(map(str, args.steps))))
    cmd_search += ' --dataset {} --epochs 100 --gpu=2'.format(args.dataset)
    cmd_eval += ' --dataset {} --epochs 500 --gpu=2'.format(args.dataset)

    cmds.append(cmd_search)
    cmds.append(cmd_eval)

eval_flag = 0
for cmd in cmds:
    print(cmd)
    p = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    logging_str = p.communicate()[0].decode("utf-8").rstrip('\n').split('\n')
    vers = logging_str[-4].split(',')
    vers_ad = logging_str[-2].split(',')
    if eval_flag % 2:
        print(logging_str[3])
        for i in logging_str[-4:]:
            print(i)
        s_score = []
        for ver in vers:
            value = ver.split(' ')[-1]
            s_score.append(float(value))
        score.append(s_score)

        s_score_ad = []
        for ver in vers_ad:
            value = ver.split(' ')[-1]
            s_score_ad.append(float(value))
        score_ad.append(s_score_ad)
        print("\n")
        eval_flag += 1
    else:
        print(logging_str[-1])
        eval_flag += 1
        continue
score = np.array(score)
score_ad = np.array(score_ad)

print('F1 macro:%.5f + %.5f' % (np.mean(score, axis=0)[0], np.std(score, axis=0)[0]))
print('AUC:%.5f + %.5f' % (np.mean(score, axis=0)[1], np.std(score, axis=0)[1]))
print('GMean:%.5f + %.5f' % (np.mean(score, axis=0)[2], np.std(score, axis=0)[2]))
print("Best score {}" .format(np.max(score, axis=0)))
print("Worst score {}" .format(np.min(score, axis=0)))

print("\nAdjust the threshold:")
print('F1 macro: %.5f + %.5f' % (np.mean(score_ad, axis=0)[0], np.std(score_ad, axis=0)[0]))
print('GMean: %.5f + %.5f' % (np.mean(score_ad, axis=0)[1], np.std(score_ad, axis=0)[1]))
print("Best score {}" .format(np.max(score_ad, axis=0)))
print("Worst score {}" .format(np.min(score_ad, axis=0)))
