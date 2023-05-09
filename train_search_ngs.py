import copy
import os
import sys
import logging
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

from model_search_ngs import Model
from preprocess import cstr_nc
from preprocess import load_data, numpy_to_torch

"""
     Perform search
"""

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--n_hid', type=int, default=64, help='hidden dimension')
parser.add_argument('--alr', type=float, default=3e-4, help='learning rate for architecture parameters')
parser.add_argument('--steps', type=int, nargs='+', help='number of intermediate states in the meta graph')
parser.add_argument('--dataset', type=str, default='amazon', help='The dataset name. [yelp, amazon, mimic]')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=100, help='number of epochs for supernet training')
parser.add_argument('--eps', type=float, default=0.3, help='probability of random sampling')
parser.add_argument('--decay', type=float, default=0.9, help='decay factor for eps')
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()

prefix = "lr" + str(args.lr) + "_wd" + str(args.wd) + \
         "_h" + str(args.n_hid) + "_alr" + str(args.alr) + \
         "_s" + str(args.steps) + "_epoch" + str(args.epochs) + \
         "_cuda" + str(args.gpu) + "_eps" + str(args.eps) + "_d" + str(args.decay)

logdir = os.path.join("log/search_ngs", args.dataset)
if not os.path.exists(logdir):
    os.makedirs(logdir)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(logdir, prefix + ".txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info(args)


def main():

    torch.cuda.set_device(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    datadir = "data/"
    prefix = os.path.join(datadir)

    # * load data
    adjs_pt, node_feats, labels = load_data(prefix, args.dataset)

    node_types = np.zeros((adjs_pt[0].shape[0],), dtype=np.int32)
    num_node_types = node_types.max() + 1
    node_types = torch.from_numpy(node_types).cuda()
    # node_types = np.load(os.path.join(prefix, "node_types.npy"))

    if args.dataset == 'yelp':
        index = np.arange(len(labels))
        train_idx, rest_idx, train_target, rest_target = train_test_split(index, labels, stratify=labels,
                                                                          train_size=0.4, random_state=2, shuffle=True)
        valid_idx, test_idx, valid_target, test_target = train_test_split(rest_idx, rest_target, stratify=rest_target,
                                                                          test_size=0.67, random_state=2, shuffle=True)
    elif args.dataset == 'amazon':
        # 0-3304 are unlabeled nodes
        index = np.arange(3305, len(labels))
        train_idx, rest_idx, train_target, rest_target = train_test_split(index, labels[3305:], stratify=labels[3305:],
                                                                          train_size=0.4, random_state=2, shuffle=True)
        valid_idx, test_idx, valid_target, test_target = train_test_split(rest_idx, rest_target, stratify=rest_target,
                                                                          test_size=0.67, random_state=2, shuffle=True)

    pos_ratio = train_target[train_target == 1].shape[0] / len(train_idx)
    logging.info("Pos_ratio %.4f" % pos_ratio)

    train_idx, valid_idx, test_idx, train_target, valid_target, test_target = numpy_to_torch(
        train_idx, valid_idx, test_idx, train_target, valid_target, test_target
    )
    n_classes = train_target.max().item() + 1
    # print("Number of classes: {}".format(n_classes), "Number of node types: {}".format(num_node_types))

    model = Model(node_feats.size(1), args.n_hid, num_node_types, len(adjs_pt), n_classes, args.steps, cstr_nc[args.dataset]).cuda()

    optimizer_w = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )

    optimizer_a = torch.optim.Adam(
        model.alphas(),
        lr=args.alr
    )

    eps = args.eps
    best_val = None
    final = None
    anchor = None
    best_arch = {}
    patience = 0
    for epoch in range(args.epochs):
        train_error, val_error = train(node_feats, node_types, adjs_pt, train_idx, train_target, valid_idx, valid_target,
                                       model, optimizer_w, optimizer_a, eps)
        arch = model.parse()
        logging.info("Epoch {}; Train err {}; Val err {}; Arch {}".format(epoch + 1, train_error, val_error, arch))
        eps = eps * args.decay

        val_loss, f1_val, f1_test, auc, gmean = infer(model, node_feats, node_types, adjs_pt, valid_idx, valid_target,
                                                  test_idx, test_target)
        if best_val is None or val_error < best_val:
            best_val = val_error
            final = (f1_test, auc, gmean)
            best_arch[args.dataset] = arch
            anchor = epoch + 1
            patience = 0
        else:
            patience += 1
            if patience == 10 and epoch >= 50:
                break

    logging.info("Best val {} at epoch {}; Test score: F1-macro {}, AUC {}, GMean {}".format(
        best_val, anchor, final[0], final[1], final[2]))
    f = open('arch.json', 'w')
    f.write(json.dumps(best_arch))
    f.close()
    print('Test score: F1-macro %.4f, AUC %.4f, GMean %.4f' % (final[0], final[1], final[2]))
    print('Arch {}'.format(best_arch[args.dataset]))


def train(node_feats, node_types, adjs, train_idx, train_target, valid_idx, valid_target, model, optimizer_w, optimizer_a, eps):

    idxes_seq, idxes_res = model.sample(eps)

    optimizer_w.zero_grad()
    out = model(node_feats, node_types, adjs, idxes_seq, idxes_res)
    cls_loss_w = F.cross_entropy(out[train_idx], train_target)
    loss_w = cls_loss_w
    loss_w.backward()
    optimizer_w.step()

    optimizer_a.zero_grad()
    out = model(node_feats, node_types, adjs, idxes_seq, idxes_res)
    cls_loss_a = F.cross_entropy(out[valid_idx], valid_target)
    loss_a = cls_loss_a
    loss_a.backward()
    optimizer_a.step()

    return loss_w.item(), loss_a.item()


def infer(model, node_feats, node_types, adjs, valid_idx, valid_target, test_idx, test_target):
    model.eval()
    with torch.no_grad():
        idxes_seq, idxes_res = model.sample(eps=0)
        out = model(node_feats, node_types, adjs, idxes_seq, idxes_res)

    loss = F.cross_entropy(out[valid_idx], valid_target)
    f1_val = f1_score(valid_target.cpu().numpy(), torch.argmax(out[valid_idx], dim=-1).cpu().numpy(), average='macro')
    f1_test = f1_score(test_target.cpu().numpy(), torch.argmax(out[test_idx], dim=-1).cpu().numpy(), average='macro')

    prob = F.softmax(out, dim=1).detach().cpu().numpy()
    pos_prob = np.array(prob[:, 0])
    fpr, tpr, thresholds = metrics.roc_curve(test_target.cpu().numpy(), pos_prob[test_idx.cpu()], pos_label=0)
    auc = metrics.auc(fpr, tpr)

    confusion_m = metrics.confusion_matrix(test_target.cpu().numpy(), torch.argmax(out[test_idx], dim=-1).cpu().numpy())
    tn, fp, fn, tp = confusion_m.ravel()
    gmean = (tp * tn / ((tp + fn) * (tn + fp))) ** 0.5
    return loss.item(), f1_val, f1_test, auc, gmean


if __name__ == '__main__':
    main()
