import torch
import os
import scipy.sparse as sp
import numpy as np
import json

import random
import datetime
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp
from typing import Tuple, Optional, List, Dict
from functools import reduce
import sklearn.metrics as skm

import dgl
import dgl.nn.pytorch as dglnn
from dgl.dataloading import DataLoader, NeighborSampler

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

from tllib.modules.entropy import entropy
from tllib.modules.domain_discriminator import DomainDiscriminator
from reweight import ImportanceWeightModule
from tllib.alignment.dann import DomainAdversarialLoss
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import tsne, a_distance

from dataset import Tesseract
from model import SAGE, MyClassifier
from utils import collect_feature_monthly, collect_feature, inductive_split, load_subtensor, testing_monthly, compute_aut_metrics, evaluate

import os

def train(args, train_source_iter, train_target_iter, model,
          domain_adv_D, domain_adv_D_0,
          importance_weight_module, optimizer,
          lr_scheduler, epoch, device):

    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    tgt_accs = AverageMeter('Tgt Acc', ':3.1f')
    domain_accs_D = AverageMeter('Domain Acc for D', ':3.1f')
    domain_accs_D_0 = AverageMeter('Domain Acc for D_0', ':3.1f')
    partial_classes_weights_s = AverageMeter('Source Partial Weight', ':3.2f')
    non_partial_classes_weights_s = AverageMeter('Source Non-Partial Weight', ':3.2f')
    partial_classes_weights_t = AverageMeter('Target Partial Weight', ':3.2f')
    non_partial_classes_weights_t = AverageMeter('Target Non-Partial Weight', ':3.2f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs, tgt_accs,
         domain_accs_D, domain_accs_D_0, partial_classes_weights_s, non_partial_classes_weights_s,
         partial_classes_weights_t, non_partial_classes_weights_t],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    domain_adv_D.train()
    domain_adv_D_0.train()

    end = time.time()
    cls_losses, advD_losses, advD0_losses, entropy_losses, losses_ = 0, 0, 0, 0, 0
    # cls_losses, advD0_losses, entropy_losses, losses_ = 0, 0, 0, 0
    for i in range(args.iters_per_epoch):
        _, _, blocks_s = next(train_source_iter)
        _, _, blocks_t = next(train_target_iter)
        x_s = blocks_s[0].srcdata['features'].to_dense()
        labels_s = blocks_s[-1].dstdata['labels']
        x_t = blocks_t[0].srcdata['features'].to_dense()
        labels_t = blocks_t[-1].dstdata['labels']

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, f_s = model(blocks_s, x_s)
        y_t, f_t = model(blocks_t, x_t)

        # classification loss
        cls_loss = F.cross_entropy(y_s, labels_s)
        # print("cls_loss: ", cls_loss.item())

        # domain adversarial loss for D
        adv_loss_D = domain_adv_D(f_s.detach(), f_t.detach())
        # print("adv_loss_D: ", adv_loss_D.item())

        # get importance weights
        w_s = importance_weight_module.get_importance_weight(f_s, label=1)
        w_t = importance_weight_module.get_importance_weight(f_t, label=0)

        # domain adversarial loss for D_0
        adv_loss_D_0 = domain_adv_D_0(f_s, f_t, w_s=w_s, w_t=w_t)
        # adv_loss_D_0 = domain_adv_D_0(f_s, f_t)
        # adv_loss_D_0 = domain_adv_D_0(f_s, f_t, w_s=w_s)
        # adv_loss_D_0 = domain_adv_D_0(f_s, f_t, w_t=w_t)
        
        # entropy loss
        y_t = F.softmax(y_t, dim=1)
        entropy_loss = entropy(y_t, reduction='mean')
        # print("entropy_loss: ", entropy_loss.item())

        loss = cls_loss + 1.5 * args.trade_off * adv_loss_D + \
               args.trade_off * adv_loss_D_0 + args.gamma * entropy_loss
        # print("loss: ", loss.item())
        # print()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        cls_acc = accuracy(y_s, labels_s)[0]
        tgt_acc = accuracy(y_t, labels_t)[0]

        cls_losses += cls_loss.item()
        advD_losses += adv_loss_D.item()
        advD0_losses += adv_loss_D_0.item()
        entropy_losses += entropy_loss.item()
        losses_ += loss.item()

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        tgt_accs.update(tgt_acc.item(), x_s.size(0))
        domain_accs_D.update(domain_adv_D.domain_discriminator_accuracy, x_s.size(0))
        domain_accs_D_0.update(domain_adv_D_0.domain_discriminator_accuracy, x_s.size(0))

        # debug: output class weight averaged on the partial classes and non-partial classes respectively
        partial_class_weight_s, non_partial_classes_weight_s = \
            importance_weight_module.get_partial_classes_weight(w_s, labels_s)
        partial_classes_weights_s.update(partial_class_weight_s.item(), x_s.size(0))
        non_partial_classes_weights_s.update(non_partial_classes_weight_s.item(), x_s.size(0))

        partial_class_weight_t, non_partial_classes_weight_t = \
            importance_weight_module.get_partial_classes_weight(w_t, labels_t)
        partial_classes_weights_t.update(partial_class_weight_t.item(), x_t.size(0))
        non_partial_classes_weights_t.update(non_partial_classes_weight_t.item(), x_t.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return cls_losses/args.iters_per_epoch, advD_losses/args.iters_per_epoch,\
           advD0_losses/args.iters_per_epoch, entropy_losses/args.iters_per_epoch,\
           losses_/args.iters_per_epoch
    # return cls_losses/args.iters_per_epoch, 0,\
    #        advD0_losses/args.iters_per_epoch, entropy_losses/args.iters_per_epoch,\
    #        losses_/args.iters_per_epoch

def run(args, data):
    train_g_s, train_g_t, val_g, test_g, timestamp, device, num_classes, save_path = data

    in_dim = train_g_s.ndata['features'].shape[1]
    train_nid_s = torch.nonzero(train_g_s.ndata['train_mask'], as_tuple=True)[0]
    train_nid_t = torch.nonzero(train_g_t.ndata['test_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]

    sampler = NeighborSampler([int(fanout) for fanout in args.fan_out.split(',')],
                              prefetch_node_feats=['features'],
                              prefetch_labels=['labels'])
    use_uva = (args.mode == 'mixed')
    train_source_loader = DataLoader(train_g_s, train_nid_s, sampler, device=device,
                                     batch_size=args.batch_size, shuffle=True,
                                     drop_last=True, num_workers=0,
                                     use_uva=use_uva)
    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_loader = DataLoader(train_g_t, train_nid_t, sampler, device=device,
                                     batch_size=args.batch_size, shuffle=True,
                                     drop_last=True, num_workers=0,
                                     use_uva=use_uva)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # Construct classifier
    classifier = MyClassifier(SAGE(in_dim, args.num_hidden, args.num_hidden,
                                   args.num_layers, F.relu,
                                   args.drop_rate, args.agg),
                              num_classes, head=None, finetune=True).to(device)

    if args.phase == 'test':
        start = datetime.datetime.now()
        checkpoint_path = "./checkpoints/{}".format(args.timestamp)
        checkpoint = torch.load(os.path.join(checkpoint_path, "best.pth"))
        classifier.load_state_dict(checkpoint['classifier'])
        metrics = testing_monthly(args, getMonths, classifier, device, test_g)
        aut_acc, aut_f1, aut_p, aut_r = compute_aut_metrics(metrics)
        print("aut_acc {:.4f}\t aut_f1 {:.4f}\t aut_p {:.4f}\t aut_r {:.4f}".format(aut_acc, aut_f1, aut_p, aut_r))

        # import pandas as pd
        # from collections import defaultdict
        # def seasonAUT(a, b, c):
        #     return (a + 2 * b + c) / 4

        # metrics = pd.DataFrame(metrics, columns=['period', 'acc', 'f1', 'p', 'r'])
        # print(metrics)
        # metrics.to_excel("results.xlsx", sheet_name='sheet1')
        
        # metrics = metrics.loc[:, 'acc':]
        # ndata = defaultdict(list)
        # for key, col in metrics.iteritems():
        #     tmp = list()
        #     for i in range(0, len(col), 3):
        #         tmp.append(seasonAUT(col[i], col[i+1], col[i+2]))
        #     ndata["AUT_" + key] = tmp
        # ndata = pd.DataFrame(ndata)
        # for col in ndata:
        #     ndata[col] = ndata[col].round(decimals=4)
        # print(ndata)
        # ndata.to_excel("results2.xlsx", sheet_name='sheet1')

        end = datetime.datetime.now()
        print("Total testing time (split testing dataset & inference): %s" % (end - start))

        return

    if args.phase == 'val':
        start = datetime.datetime.now()
        checkpoint_path = "./checkpoints/{}".format(args.timestamp)
        checkpoint = torch.load(os.path.join(checkpoint_path, "best.pth"))
        classifier.load_state_dict(checkpoint['classifier'])
        eval_acc, eval_f1, eval_p, eval_r = evaluate(args, classifier, val_g, val_nid, device)
        print('Eval Acc {:.4f} | Eval F1: {:.4f} | Eval Precision: {:.4f} | Eval Recall: {:.4f}'
        .format(eval_acc, eval_f1, eval_p, eval_r))
        end = datetime.datetime.now()
        print("Total val time: %s" % (end - start))

        return

    # load pre-trained model
    if args.load_model != "":
        checkpoint = torch.load(os.path.join(save_path, "{}.pth".format(args.load_model)))
        classifier.load_state_dict(checkpoint)

    # define domain classifier D, D_0
    D = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=1024, batch_norm=False, sigmoid=True).to(device)
    D_0 = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=1024, batch_norm=False, sigmoid=True).to(device)

    if args.phase == 'analysis':

        start = datetime.datetime.now()
        checkpoint_path = "./checkpoints/{}".format(args.timestamp)
        checkpoint = torch.load(os.path.join(checkpoint_path, "best.pth"))
        classifier.load_state_dict(checkpoint['classifier'])
        D.load_state_dict(checkpoint['D'])
        importance_weight_module = ImportanceWeightModule(D, [1])

        feature_extractor = classifier.backbone.to(device)
        all_features = collect_feature(args, test_g, feature_extractor, device)
        torch.save(all_features, "./visual/features_drebin2.pt")

        source_sample = all_features[torch.cat((train_nid_s, val_nid), dim=0)]
        target_sample = all_features[train_nid_t]
        source_weight = importance_weight_module.get_importance_weight(source_sample, label=1)
        target_weight = importance_weight_module.get_importance_weight(target_sample, label=0)
        torch.save(torch.cat((source_weight, target_weight), dim=0), "./visual/weights.pt")

        return

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters() + D.get_parameters() + D_0.get_parameters(),
                    args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    domain_adv_D = DomainAdversarialLoss(D).to(device)
    domain_adv_D_0 = DomainAdversarialLoss(D_0).to(device)

    # define importance weight module
    importance_weight_module = ImportanceWeightModule(D, [1])

    # start training
    best_f1 = 0.
    cls_losses = []
    domain_D_losses = []
    domain_D0_losses = []
    entropy_losses = []
    losses = []
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    checkpoint_path = os.path.join(save_path, timestamp)
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    for epoch in range(args.num_epochs):
        # train for one epoch
        tic = time.time()
        cls_loss, domain_D_loss, domain_D0_loss, entropy_loss, loss\
        = train(args, train_source_iter, train_target_iter,
                classifier, domain_adv_D, domain_adv_D_0,
                importance_weight_module, optimizer, lr_scheduler,
                epoch, device)
        cls_losses.append(cls_loss)
        domain_D_losses.append(domain_D_loss)
        domain_D0_losses.append(domain_D0_loss)
        entropy_losses.append(entropy_loss)
        losses.append(loss)

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        # evaluate on validation set
        eval_acc, eval_f1, eval_p, eval_r = evaluate(args, classifier, val_g, val_nid, device)
        print('Eval Acc {:.4f} | Eval F1: {:.4f} | Eval Precision: {:.4f} | Eval Recall: {:.4f}'
        .format(eval_acc, eval_f1, eval_p, eval_r))

        # remember best f1 and save checkpoint
        torch.save({'classifier': classifier.state_dict(),
                    'D': D.state_dict()}, os.path.join(checkpoint_path, "latest.pth"))
        if eval_f1 > best_f1:
            shutil.copy(os.path.join(checkpoint_path, "latest.pth"), os.path.join(checkpoint_path, "best.pth"))
        best_f1 = max(eval_f1, best_f1)
        
        # torch.save(classifier.state_dict(), os.path.join(checkpoint_path, "latest.pth"))

    print("best_f1 = {:3.1f}".format(best_f1))

    # Testing
    start = datetime.datetime.now()
    checkpoint = torch.load(os.path.join(checkpoint_path, "best.pth"))
    classifier.load_state_dict(checkpoint['classifier'])
    metrics = testing_monthly(args, getMonths, classifier, device, test_g)
    aut_acc, aut_f1, aut_p, aut_r = compute_aut_metrics(metrics)
    end = datetime.datetime.now()
    print("Total testing time (split testing dataset & inference): %s" % (end - start))

    import pickle as pkl
    with open(os.path.join(checkpoint_path, "loss.pkl"), "wb") as f:
        pkl.dump((cls_losses, domain_D_losses, domain_D0_losses, entropy_losses, losses), f)

    report_file = os.path.join(save_path, "reports.csv")
    with open(report_file, 'a') as f:
        f.write(os.path.join(checkpoint_path, "best.pth") + "\n")
        f.write("{}\n".format(args))
        f.write("Best model performance\tAUT_acc {:.4f}\tAUT_F1 {:.4f}\tAUT_P {:.4f}\tAUT_R {:.4f}\n".format(aut_acc, aut_f1, aut_p, aut_r))
        f.write("\n")

def getMonths():
    # for year in [2013]:
    #     for month in range(1, 13):
    # for year in [2015]:
    #     for month in range(1, 4):
    # for year in [2015, 2016]:
    #     for month in range(1, 13):
    for year in [2012]:
        for month in range(1, 11):
            period = "{}-{}".format(year, month)
            test_mask = "test_mask_" + period
            yield period, test_mask

if __name__ == "__main__":
    print()
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--lr', type=float, default=0.2)
    argparser.add_argument('--momentum', type=float, default=0.9)
    argparser.add_argument('--weight-decay', type=float, default=1e-3)
    argparser.add_argument('--lr-gamma', type=float, default=1e-3)
    argparser.add_argument('--lr-decay', type=float, default=0.75)
    argparser.add_argument('--gamma', type=float, default=0.1)
    argparser.add_argument('--trade-off', type=int, default=5)
    argparser.add_argument('--num-epochs', type=int, default=100)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=2,
                            help="Number of gnn layers.")
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--agg', type=str, default='mean')
    argparser.add_argument('--load-model', type=str, default="",
                            help="Specify the name of pretrained model.")
    argparser.add_argument('--batch-size', type=int, default=64)
    argparser.add_argument('--iters-per-epoch', type=int, default=100)
    argparser.add_argument('--print-freq', type=int, default=100)
    argparser.add_argument('--drop-rate', type=float, default=0.5)
    argparser.add_argument("--mode", default='puregpu', choices=['cpu', 'mixed', 'puregpu'],
                            help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
                            "'puregpu' for pure-GPU training.")
    argparser.add_argument('--inductive', action='store_true',
                           help="Inductive learning setting")
    argparser.add_argument('--detailed', action='store_true', 
                            help="Print confusion matrix during inference process.")

    # model testing/validation/ananlysis
    argparser.add_argument('--timestamp', type=str, default='2023-03-02-22-48',
                            help="Specify the name of trained model for testing/validation/analysis")
    argparser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'val', 'analysis'],
                        help="When phase is 'test', only test the model.")
    args = argparser.parse_args()

    assert len(args.fan_out.split(',')) == args.num_layers, "Specify number of sampled neighbors for each layer."
    if args.gpu > -1:
        assert args.mode in ['mixed', 'puregpu'], "Use mixed or puregpu mode, when you specify a GPU device ID."
    else:
        assert args.mode == 'cpu', "No need to specify a GPU device ID, when you use cpu mode."
    print(args)

    first_start = datetime.datetime.now()
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load Datasets
    start = datetime.datetime.now()
    # data_dir = "/home/sunrui/data/GraphEvolveDroid"
    # feat_mtx_file = "drebin_feat_mtx.npz"
    data_dir = "/home/sunrui/data/drebin"
    feat_mtx_file = "drebin_feat.npz"
    adj_mtx_file = "drebin_knn_5.npz"
    dataset = Tesseract(data_dir, feat_mtx_file, adj_mtx_file)
    g = dataset[0]
    end = datetime.datetime.now()
    print("Loading dataset time: %s" % (end - start))

    num_classes = 2
    save_path = "./checkpoints"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    print("***********************", timestamp, "***********************")

    device = torch.device('cpu' if args.mode == 'cpu' else 'cuda:{}'.format(args.gpu))
    print("device: ", device)
    if args.inductive:
        train_g, val_g, test_g = inductive_split(g)
        if args.mode == 'puregpu':
            train_g_s = train_g.to(device)
            train_g_t = test_g.to(device)
            val_g = val_g.to(device)
            test_g = g.to(device)
        else:
            train_g_s = train_g
            train_g_t = test_g
    else:
        if args.mode == 'puregpu':
            train_g_s = train_g_t = val_g = test_g = g.to(device)
        else:
            train_g_s = train_g_t = val_g = test_g = g

    data = train_g_s, train_g_t, val_g, test_g, timestamp, device, num_classes, save_path

    run(args, data)
    last_end = datetime.datetime.now()
    print("Program total execution time: %s" % (last_end - first_start))