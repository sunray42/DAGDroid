import os
import scipy.sparse as sp
import numpy as np
import json
import sklearn.metrics as skm
import datetime
import time
import argparse
import random
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import torch.optim as optim

import dgl
import dgl.nn.pytorch as dglnn
from dgl.dataloading import DataLoader, NeighborSampler

from model import SAGE, MyClassifier
from dataset import Tesseract
from utils import inductive_split, evaluate, compute_metrics, compute_aut_metrics, testing_monthly

def run(args, device, train_g, model):
    train_nid = torch.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0].to(device)
    
    sampler = NeighborSampler([int(fanout) for fanout in args.fan_out.split(',')],
                              prefetch_node_feats=['features'],
                              prefetch_labels=['labels'])
    use_uva = (args.mode == 'mixed')
    train_dataloader = DataLoader(train_g, train_nid, sampler, device=device,
                                  batch_size=args.batch_size, shuffle=True,
                                  drop_last=False, num_workers=0,
                                  use_uva=use_uva)

    loss_fcn = nn.CrossEntropyLoss(weight=torch.Tensor([1, 5]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    avg = 0
    iter_tput = []
    hist_loss = []
    for epoch in range(args.num_epochs):
        model.train()
        tic = time.time()

        tic_step = time.time()
        batch_loss = []
        for step, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata['features'].to_dense()
            y = blocks[-1].dstdata['labels']
            y_hat, _ = model(blocks, x)
            loss = loss_fcn(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(output_nodes) / (time.time() - tic_step))
            if step % args.log_every == 0:
                acc = MF.accuracy(y_hat, y)
                gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'\
                .format(epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))
            tic_step = time.time()
            batch_loss.append(loss.item())

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        hist_loss.append(sum(batch_loss) / len(batch_loss))
        if epoch >= 5:
            avg += toc - tic

    if epoch >= 5:
        print('Avg epoch time: {}'.format(avg / (epoch - 4)))

    with open("./checkpoints/{}_loss.pkl".format(args.model_name), "wb") as f:
        pkl.dump(hist_loss, f)
    
def getMonths():
    """
    A testing periods generator
    """
    # for year in ['2013']:
    #     for month in range(1, 13):
    # for year in ['2015']:
    #     for month in range(1, 4):
    for year in ['2015', '2016']:
        for month in range(1, 13):
            period = year + "-" + str(month)
            test_mask = "test_mask_" + period
            yield period, test_mask

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--lr', type=float, default=1e-4)
    argparser.add_argument('--weight-decay', type=float, default=1e-5)
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=2,
                            help="Number of gnn layers.")
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--agg', type=str, default='mean')
    argparser.add_argument('--model-name', type=str, default='pretrain')
    argparser.add_argument('--batch-size', type=int, default=128)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=1)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument("--mode", default='puregpu', choices=['cpu', 'mixed', 'puregpu'],
                        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
                             "'puregpu' for pure-GPU training.")
    argparser.add_argument('--detailed', action='store_true', 
                            help="Print confusion matrix during inference process.")
    args = argparser.parse_args()
    assert len(args.fan_out.split(',')) == args.num_layers, "Specify number of sampled neighbors for each layer."
    if args.gpu > -1:
        assert args.mode in ['mixed', 'puregpu'], "Use mixed or puregpu mode, when you specify a GPU device ID."
    else:
        assert args.mode == 'cpu', "No need to specify a GPU device ID, when you use cpu mode."
    print()
    print(args)

    first_start = datetime.datetime.now()
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    start = datetime.datetime.now()
    data_dir = "/home/sunrui/data/GraphEvolveDroid"
    feat_mtx_file = "drebin_feat_mtx.npz"
    adj_mtx_file = "drebin_knn_5.npz"
    dataset = Tesseract(data_dir, feat_mtx_file, adj_mtx_file)
    g = dataset[0]
    end = datetime.datetime.now()
    print("Loading dataset time: %s" % (end - start))

    # 1 split graph for inductive learning
    start = datetime.datetime.now()
    train_g, _, _ = inductive_split(g)
    test_g = g
    end = datetime.datetime.now()
    print("Splitting dataset time: %s" % (end - start))

    if args.mode == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % args.gpu)

    if args.mode == 'puregpu':
        start = datetime.datetime.now()
        train_g = train_g.to(device)
        end = datetime.datetime.now()
        print("Moving train/val graph to device (%s) time: %s" % (device, end - start))

    # 2 create model
    in_size = train_g.ndata['features'].shape[1]
    out_size = dataset.num_classes
    model = MyClassifier(SAGE(in_size,
                              args.num_hidden,
                              args.num_hidden,
                              args.num_layers,
                              F.relu,
                              args.dropout,
                              args.agg),
                         out_size, head=None, finetune=True).to(device)


    # 3 model training
    run(args, device, train_g, model)

    # 4 save model
    torch.save(model.state_dict(), os.path.join("./checkpoints", "{}.pth".format(args.model_name)))
    
    # 5 Testing
    if args.mode == 'puregpu':
        start = datetime.datetime.now()
        test_g = test_g.to(device)
        end = datetime.datetime.now()
        print("Moving test graph to device (%s) time: %s" % (device, end - start))

    start = datetime.datetime.now()
    metrics = testing_monthly(args, getMonths, model, device, test_g)
    end = datetime.datetime.now()
    print("Total testing time: %s" % (end - start))

    # Compute AUT metrics
    aut_acc, aut_f1, aut_p, aut_r = compute_aut_metrics(metrics)
    print("Model performance\tAUT_acc {:.4f}\tAUT_F1 {:.4f}\tAUT_P {:.4f}\tAUT_R {:.4f}".format(aut_acc, aut_f1, aut_p, aut_r))

    last_end = datetime.datetime.now()
    print("Program total execution time: %s" % (last_end - first_start))