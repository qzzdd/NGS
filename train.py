import copy
import os
import sys
import time, datetime
import json

import numpy as np
import logging
import argparse
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

from model import Model
from preprocess import load_data, numpy_to_torch

"""
     Train the discovered architectures
"""

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--alr', type=float, default=3e-4, help='learning rate for architecture parameters')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
parser.add_argument('--n_hid', type=int, default=64, help='hidden dimension')
parser.add_argument('--dataset', type=str, default='amazon', help='The dataset name. [yelp, amazon, mimic]')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=50, help='maximum number of training epochs')
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--seed', type=int, default=24)
parser.add_argument('--no_norm', action='store_true', default=False, help='disable layer norm')
parser.add_argument('--in_nl', action='store_true', default=False, help='non-linearity after projection')
args = parser.parse_args()

prefix = "lr" + str(args.lr) + "_wd" + str(args.wd) + "_h" + str(args.n_hid) + \
         "_drop" + str(args.dropout) + "_epoch" + str(args.epochs) + "_cuda" + str(args.gpu) + "_seed" + str(args.seed)

if args.no_norm is True:
    prefix += "_noLN"
if args.in_nl is True:
    prefix += "_nl"

logdir = os.path.join("log/eval", args.dataset)
if not os.path.exists(logdir):
    os.makedirs(logdir)

# load meta-graph
f2 = open('arch.json', 'r')
info_data = json.load(f2)
meta_template = copy.deepcopy(info_data[args.dataset])
steps = [len(meta) for meta in meta_template[0]]
prefix += "_s" + str(steps)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(logdir, prefix + ".txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info(args)


def main():

    torch.cuda.set_device(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    global meta_template
    steps = [len(meta) for meta in meta_template[0]]
    print("Steps: {}".format(steps))

    datadir = "data/"
    prefix = os.path.join(datadir)

    #* load data
    adjs_pt, node_feats, labels = load_data(prefix, args.dataset)

    node_types = np.zeros((adjs_pt[0].shape[0],), dtype=np.int32)
    num_node_types = node_types.max() + 1
    node_types = torch.from_numpy(node_types).cuda()

    logging.info("Arch {}".format(meta_template))

    #* load labels
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
    print("Number of classes: {}".format(n_classes), "Number of node types: {}".format(num_node_types))

    model = Model(node_feats.size(1), args.n_hid, num_node_types, n_classes, steps, dropout=args.dropout, use_norm=not args.no_norm, in_nl=args.in_nl).cuda()
    model.init_index(train_idx, train_target, valid_idx, valid_target)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )

    timestamp = time.time()
    timestamp = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H-%M-%S')
    dir_saver = './checkpoint/' + timestamp
    path_saver = os.path.join(dir_saver, '{}_{}.pkl'.format(args.dataset, 'darts'))

    best_val = None
    final = None
    anchor = None
    patience = 0
    for epoch in range(args.epochs):
        train_loss = train(node_feats, node_types, adjs_pt, train_idx, train_target, model, optimizer)
        val_loss, f1_val, test_score, test_score_ad = infer(
            node_feats, node_types, adjs_pt, valid_idx, valid_target, test_idx, test_target, model)
        logging.info("Epoch {}; Train err {}; Val err {}\n".format(epoch + 1, train_loss, val_loss))
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            final = test_score
            final_ad = test_score_ad
            if not os.path.exists(dir_saver):
                os.mkdir(dir_saver)
            print('  Saving model ...')
            torch.save(model.state_dict(), path_saver)
            anchor = epoch + 1
            patience = 0
        else:
            patience += 1
            if patience == 50 and epoch >= 100:
                #pass
                break
    logging.info("Best val {} at epoch {}; Test score: F1-macro {}, AUC {}, GMean {}".format(
        best_val, anchor, final[0], final[1], final[2]))
    logging.info("True Positive Rate {}; True Negative Rate {}".format(final[3], final[4]))
    logging.info("Threshold = 0.2: F1-macro {}, GMean {}".format(final_ad[0], final_ad[1]))
    logging.info("True Positive Rate {}; True Negative Rate {}".format(final_ad[2], final_ad[3]))

    print('Test score: F1-macro %.5f, AUC %.5f, GMean %.5f' % (final[0], final[1], final[2]))
    print("True Positive Rate %.5f; True Negative Rate %.5f" % (final[3], final[4]))
    print("Threshold = 0.2: F1-macro %.5f, GMean %.5f" % (final_ad[0], final_ad[1]))
    print("True Positive Rate %.5f; True Negative Rate %.5f" % (final_ad[2], final_ad[3]))


def train(node_feats, node_types, adjs, train_idx, train_target, model, optimizer):

    model.train()
    optimizer.zero_grad()
    out = model(node_feats, node_types, adjs, meta_template[0], meta_template[1])
    loss = F.cross_entropy(out[train_idx], train_target)
    loss.backward()
    optimizer.step()
    return loss.item()


def infer(node_feats, node_types, adjs, valid_idx, valid_target, test_idx, test_target, model):

    model.eval()
    with torch.no_grad():
        out = model(node_feats, node_types, adjs, meta_template[0], meta_template[1])
    loss = F.cross_entropy(out[valid_idx], valid_target)
    f1_val = f1_score(valid_target.cpu().numpy(), torch.argmax(out[valid_idx], dim=-1).cpu().numpy(), average='macro')
    f1_test = f1_score(test_target.cpu().numpy(), torch.argmax(out[test_idx], dim=-1).cpu().numpy(), average='macro')

    prob = F.softmax(out, dim=1).detach().cpu().numpy()
    pos_prob = np.array(prob[:, 0])
    fpr, tpr, thresholds = metrics.roc_curve(test_target.cpu().numpy(), pos_prob[test_idx.cpu()], pos_label=0)
    auc = metrics.auc(fpr, tpr)

    confusion_m = metrics.confusion_matrix(test_target.cpu().numpy(), torch.argmax(out[test_idx], dim=-1).cpu().numpy())
    tn, fp, fn, tp = confusion_m.ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    gmean = (tp * tn / ((tp + fn) * (tn + fp))) ** 0.5
    logging.info("In test set: %d, %d, %d, %d" % (tn, fp, fn, tp))
    logging.info("True Positive Rate {}; True Negative Rate {}".format(tpr, tnr))
    logging.info("Test score: F1-macro {}, GMean {}" .format(f1_test, gmean))

    # 调整分类阈值
    preds_ad = np.array(prob[:, 1])
    preds_ad = torch.IntTensor(preds_ad > 0.2)
    f1_ad = f1_score(test_target.cpu().numpy(), preds_ad[test_idx], average="macro")
    conf_ad = metrics.confusion_matrix(test_target.cpu().numpy(), preds_ad[test_idx])
    tn_ad, fp_ad, fn_ad, tp_ad = conf_ad.ravel()
    tpr_ad = tp_ad / (tp_ad + fn_ad)
    tnr_ad = tn_ad / (tn_ad + fp_ad)
    gmean_ad = (tp_ad * tn_ad / ((tp_ad + fn_ad) * (tn_ad + fp_ad))) ** 0.5

    logging.info("Adjust the threshold:")
    logging.info("In test set: %d, %d, %d, %d" % (tn_ad, fp_ad, fn_ad, tp_ad))
    logging.info("True Positive Rate {}; True Negative Rate {}".format(tpr_ad, tnr_ad))
    logging.info("Test score: F1-macro {}, GMean {}".format(f1_ad, gmean_ad))

    return loss.item(), f1_val, (f1_test, auc, gmean, tpr, tnr), (f1_ad, gmean_ad, tpr_ad, tnr_ad)


if __name__ == '__main__':
    main()
