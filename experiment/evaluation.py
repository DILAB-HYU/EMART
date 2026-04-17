import collections
import numpy as np
import pandas as pd
import copy, pdb, time, warnings, torch

from torch import nn
from torch.utils import data
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score, classification_report

warnings.filterwarnings('ignore')

class EvalMetric(object):
    def __init__(self, multilabel=False):
        self.multilabel = multilabel
        self.pred_list = list()
        self.truth_list = list()
        self.top_k_list = list()
        self.loss_list = list()
        self.ploss_list = list()
        self.angle_instance_loss_list = list()
        self.demo_list = list()
        self.speaker_list = list()
        
    def append_classification_results(
        self, 
        labels,
        outputs,
        loss            =None,
        ploss          =None,
        angle_instance_loss = None,
        demographics    =None,
        speaker_id      =None
    ):
        predictions = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        for idx in range(len(predictions)):
            self.pred_list.append(predictions[idx])
            self.truth_list.append(labels.detach().cpu().numpy()[idx])
        if loss is not None: self.loss_list.append(loss.item())
        if ploss is not None: self.ploss_list.append(ploss.item())
        if angle_instance_loss is not None: self.angle_instance_loss_list.append(angle_instance_loss.item())
        if demographics is not None: 
            self.demo_list.append(demographics)
            # if demographics == "male": self.demo_list.append(1.0)
            # else: self.demo_list.append(0.0)
        if speaker_id is not None: self.speaker_list.append(speaker_id)
        
    def classification_summary(
        self, return_auc: bool=False
    ):
        result_dict = dict()
        if self.ploss_list is not None: 
            result_dict["ploss"] = np.mean(self.ploss_list)   
        if self.angle_instance_loss_list is not None: 
            result_dict["angle_instance_loss"] = np.mean(self.angle_instance_loss_list)  
        result_dict['acc'] = accuracy_score(self.truth_list, self.pred_list)*100
        result_dict['uar'] = recall_score(self.truth_list, self.pred_list, average="macro")*100
        result_dict['mf1'] = f1_score(self.truth_list, self.pred_list, average="weighted")*100
        result_dict['report'] = classification_report(self.truth_list, self.pred_list, output_dict=True)
        result_dict['top5_acc'] = (np.sum(self.top_k_list == np.array(self.truth_list).reshape(len(self.truth_list), 1)) / len(self.truth_list))*100
        result_dict['conf'] = np.round(confusion_matrix(self.truth_list, self.pred_list, normalize='true')*100, decimals=2)
        result_dict["loss"] = np.mean(self.loss_list)
        result_dict["sample"] = len(self.truth_list)
        return result_dict

