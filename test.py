import torch
from options import *
import numpy as np
from dataset_loader import *
from sklearn.metrics import roc_curve,auc,precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

def get_predicts(test_loader, net):
    load_iter = iter(test_loader)
    frame_predict = []
    
    for i in range(len(test_loader.dataset)//5):
        _data, _label = next(load_iter)
        
        _data = _data.cuda()
        _label = _label.cuda()
        res = net(_data)   
        
        a_predict = res.cpu().numpy().mean(0)   

        fpre_ = np.repeat(a_predict, 16)
        frame_predict.append(fpre_)

    frame_predict = np.concatenate(frame_predict, axis=0)
    return frame_predict

def get_metrics(frame_predict, frame_gt):
    metrics = {}

    fpr,tpr,_ = roc_curve(frame_gt, frame_predict)
    metrics['AUC'] = auc(fpr, tpr)
    
    precision, recall, th = precision_recall_curve(frame_gt, frame_predict)
    metrics['AP'] = auc(recall, precision)
    
    return metrics

def test(net, test_loader, test_info, step, model_file = None):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        load_iter = iter(test_loader)
        frame_gt = np.load("frame_label/xd_gt.npy")
        
        frame_predicts = get_predicts(test_loader, net)

        metrics = get_metrics(frame_predicts, frame_gt)
        
        test_info['step'].append(step)
        for score_name, score in metrics.items():
            metrics[score_name] = score * 100
            test_info[score_name].append(metrics[score_name])

        return metrics
