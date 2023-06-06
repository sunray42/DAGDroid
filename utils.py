import torch
import sklearn.metrics as skm
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn

def inductive_split(g):
    """Split the graph into training graph, validation graph by training
    and validation masks.  Suitable for inductive models."""
    train_g = g.subgraph(g.ndata['train_mask'])
    val_g = g.subgraph(g.ndata['val_mask'])
    test_g = g.subgraph(g.ndata['test_mask'])
    return train_g, val_g, test_g

def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes
    """
    batch_inputs = [nfeat[node].to_dense() for node in input_nodes]
    batch_inputs = torch.stack(batch_inputs)
    batch_inputs = batch_inputs.to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

def compute_metrics(pred, labels, detailed=False):
    """
    Compute the metrics of prediction given the labels.
    """
    acc = skm.accuracy_score(labels, pred)
    f1 = skm.f1_score(labels, pred)
    precision = skm.precision_score(labels, pred)
    recall = skm.recall_score(labels, pred)
    if detailed:
        print("Confusion_Matrix:\n{}".format(skm.confusion_matrix(labels, pred)))
    return acc, f1, precision, recall

def compute_aut_metrics(metrics):
    """
    Compute AUT version of each metrics
    """
    # Delete 'period' field for each row
    metrics = [metric[1:] for metric in metrics]
    metrics = np.array(metrics, dtype=np.float32)
    norm = metrics.shape[0] * 2 - 2
    aut_metrics = metrics[1:-1].sum(axis=0) + metrics.sum(axis=0)
    return aut_metrics / norm

def testing_monthly(args, getMonths, model, device, g):
    """
    Model is tested on dataset month by month.
    Save time by handling the inference process separately for different experimental settings.
    """
    metrics = list()
    model.eval()
    with torch.no_grad():
        pred = model.inference(g, device, args.batch_size)
    model.train()
    for period, test_mask in getMonths():
        test_nid = torch.nonzero(g.ndata[test_mask], as_tuple=True)[0]
        month_pred = pred[test_nid].cpu().data.numpy().argmax(axis=1)
        # month_pred = pred[test_nid]
        month_labels = g.ndata['labels'][test_nid].cpu()
        test_acc, test_f1, test_p, test_r = compute_metrics(month_pred, month_labels, args.detailed)
        print('Test period: {} | Test Acc: {:.4f} | Test F1: {:.4f} | Test Precision: {:.4f} | Test Recall: {:.4f}'.format(period, test_acc, test_f1, test_p, test_r))
        metrics.append([period, test_acc, test_f1, test_p, test_r])
    return metrics

def evaluate(args, model, g, nid, device):
    model.eval()
    with torch.no_grad():
        pred = model.inference(g, device, args.batch_size)
    model.train()
    pred = pred[nid].cpu().data.numpy().argmax(axis=1)
    labels = g.ndata['labels'][nid].cpu()
    return compute_metrics(pred, labels, args.detailed)

def collect_feature_monthly(args, g, getMonths, feature_extractor, device):
    res = dict()
    feature_extractor.eval()
    with torch.no_grad():
        f = feature_extractor.inference(g, device, args.batch_size)
    print("f: ", type(f), f.shape)
    train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
    res['train'] = f[train_nid]
    res['val'] = f[val_nid]
    test_res = dict()
    for period, test_mask in getMonths():
        test_nid = torch.nonzero(g.ndata[test_mask], as_tuple=True)[0]
        test_res[period] = f[test_nid]
    res['test'] = test_res
    return res

def collect_feature(args, g, feature_extractor, device):
    feature_extractor.eval()
    with torch.no_grad():
        f = feature_extractor.inference(g, device, args.batch_size)
    print("f: ", type(f), f.shape)
    return f