import json
import yaml
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import loralib as lora
import argparse, logging
import torch.multiprocessing
from transformers import RobertaTokenizer
from torch.cuda.amp import GradScaler 
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

import time, sys, os
import logging
import wandb
from datetime import datetime

from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from collections import defaultdict, deque

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'model'))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'dataloader'))

from model.wav2vec import Wav2VecWrapper
from model.wavlm_plus import WavLMWrapper
from model.custom_roberta import RobertaCrossAttn 
from model.prediction import  TextAudioClassifier, TextAudioClassifierForCrossModalAttn
from evaluation import EvalMetric

from dataloader import load_finetune_audios, set_finetune_dataloader, return_weights

from utils import Loss, BalancedSoftmaxLoss, BarlowTwinsLoss, plutchik_contrastive_loss_instance
from utils import parse_finetune_args, set_seed, log_epoch_result, log_best_result, excution_time, replace_report_labels


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 


# Model hidden states information
hid_dim_dict = {
    "wav2vec2_0":       768,
    "wav2vec2_0-large": 1024,

    "wavlm":            768, 
    "wavlm-large":      1024,

    "roberta-base":     768,
    "roberta-large":    1024,
}


def train_epoch(
    dataloader, 
    model, 
    device, 
    optimizer,
    weights,
    at_barlow_align,
    plutchik_instance_match,
    wandb_store=None,
    scaler=None,            # AMP scaler 
    grad_clip_norm=None,
    scheduler=None   
):
    model.train()
    
    if args.focal_loss:
        # neu sad ang joy sur fear dis 
        class_weights = torch.FloatTensor(
                    [1 / 0.469506857, 1 / 0.073096002, 1 / 0.117230814, 1 / 0.168368836, 1 / 0.119346367, 1 / 0.026116137, 1 / 0.026334987])
        class_weights = torch.log(class_weights)
        criterion = Loss(alpha=weights) 
    elif args.balanced_ce:
        class_counts = [4709, 1743, 1233, 1204, 683, 236, 242]
        criterion = BalancedSoftmaxLoss(class_counts)
    elif args.weighted_ce:
        criterion = nn.CrossEntropyLoss(weights).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    barlow_twins = BarlowTwinsLoss(batch_size=args.batch_size, dim=256, device=device)
    eval_metric = EvalMetric()
    
    for batch_idx, batch_data in enumerate(dataloader):
        model.zero_grad()
        optimizer.zero_grad()

        x, x_text, _, y, length = batch_data 
        x, y,  length = x.to(device), y.to(device),  length.to(device)
        
        outputs, audio_feat, text_feat, fuse_feat = model(audio_input = x, text_input = x_text, length=length)

        loss = criterion(outputs, y)

        if at_barlow_align:
            p_loss = barlow_twins(audio_feat, text_feat)
            loss += args.at_barlow_coeff * p_loss
        else:
            p_loss = None

        if plutchik_instance_match:
            angle_instance_loss = plutchik_contrastive_loss_instance(y, fuse_feat, args.weak_pos, dataset=args.dataset, mode=args.ssl_mode)
            loss += args.plutchik_instance_coeff * angle_instance_loss
        else:
            angle_instance_loss = None
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    grad_clip_norm
                )
            scaler.step(optimizer)
            if args.warmup:
                scheduler.step()     
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    grad_clip_norm
                )
            optimizer.step()
            if args.warmup:
                scheduler.step()    


        eval_metric.append_classification_results(y, outputs, loss, p_loss, angle_instance_loss) 
        
        if (batch_idx % 10 == 0 and batch_idx != 0) or batch_idx == len(dataloader) - 1:
            result_dict = eval_metric.classification_summary()
            logging.info(f'Fold {fold_idx} - Current Train Loss at epoch {epoch}, step {batch_idx+1}/{len(dataloader)}: {result_dict["loss"]:.3f}')
            logging.info(f'Fold {fold_idx} - Current Train UAR at epoch {epoch}, step {batch_idx+1}/{len(dataloader)}: {result_dict["uar"]:.2f}%')
            logging.info(f'Fold {fold_idx} - Current Train WF1 at epoch {epoch}, step {batch_idx+1}/{len(dataloader)}: {result_dict["mf1"]:.2f}%')
            logging.info(f'Fold {fold_idx} - Current Train ACC at epoch {epoch}, step {batch_idx+1}/{len(dataloader)}: {result_dict["acc"]:.2f}%')
            logging.info(f'Fold {fold_idx} - Current Train LR at epoch {epoch}, step {batch_idx+1}/{len(dataloader)}: {scheduler.optimizer.param_groups[0]["lr"]}')
            
            if args.pooling_mode  == 'weighted_pool':
                logging.info(f'Fold {fold_idx} - Current Train alpha at epoch {epoch}, step {batch_idx+1}/{len(dataloader)}: {model.text_model.alpha.item():.4f}')
            if at_barlow_align:
                logging.info(f'Fold {fold_idx} - Current Train Plutchik Loss at epoch {epoch}, step {batch_idx+1}/{len(dataloader)}: {result_dict["ploss"]:.3f}')
            logging.info(f'-------------------------------------------------------------------')
            if wandb_store:
                if args.pooling_mode  == 'weighted_pool':
                    wandb.log({
                        "Train Loss": result_dict["loss"],
                        "Train UAR": result_dict["uar"],
                        "Train WF1": result_dict["mf1"],
                        "Train ACC": result_dict["acc"],
                        "Learning Rate": scheduler.optimizer.param_groups[0]["lr"],
                        "alpha": model.text_model.alpha.item()
                    })
                else:
                    if at_barlow_align and plutchik_instance_match:
                        wandb.log({
                        "Train Loss": result_dict["loss"],
                        "Barlow align Loss": result_dict["ploss"],
                        "Russel Instance Loss": result_dict["angle_instance_loss"],
                        "Train UAR": result_dict["uar"],
                        "Train WF1": result_dict["mf1"],
                        "Train ACC": result_dict["acc"],
                        "Learning Rate": scheduler.optimizer.param_groups[0]["lr"]
                    })
                    elif at_barlow_align:
                        wandb.log({
                        "Train Loss": result_dict["loss"],
                        "Barlow align Loss": result_dict["ploss"],
                        "Train UAR": result_dict["uar"],
                        "Train WF1": result_dict["mf1"],
                        "Train ACC": result_dict["acc"],
                        "Learning Rate": scheduler.optimizer.param_groups[0]["lr"]
                    })

                    else:
                        wandb.log({
                            "Train Loss": result_dict["loss"],
                            "Train UAR": result_dict["uar"],
                            "Train WF1": result_dict["mf1"],
                            "Train ACC": result_dict["acc"],
                            "Learning Rate": scheduler.optimizer.param_groups[0]["lr"]
                        })
    logging.info(f'-------------------------------------------------------------------')
    result_dict = eval_metric.classification_summary()
    return result_dict

def validate_epoch(
    dataloader, 
    model, 
    device,
    weights,
    at_barlow_align,
    plutchik_instance_match,
    split:  str="Validation",
    wandb_store=None
):  
    model.eval()

    if args.focal_loss:
        criterion = Loss(alpha=weights)
    elif args.balanced_ce:
        class_counts = [4709, 1743, 1233, 1204, 683, 236, 242]
        criterion = BalancedSoftmaxLoss(class_counts)
    elif args.weighted_ce:
        criterion = nn.CrossEntropyLoss(weights).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    barlow_twins = BarlowTwinsLoss(batch_size=args.batch_size, dim=256, device=device)

    eval_metric = EvalMetric()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            
            x, x_text, _, y, length= batch_data
            x, y = x.to(device), y.to(device)

            outputs, audio_feat, text_feat, fuse_feat = model(audio_input = x, text_input = x_text, length=length)
            
            loss = criterion(outputs, y)

            if at_barlow_align:
                p_loss = barlow_twins(audio_feat, text_feat)
                loss += args.at_barlow_coeff * p_loss
            else:
                p_loss = None
            if plutchik_instance_match:
                angle_instance_loss = plutchik_contrastive_loss_instance(y, fuse_feat, args.weak_pos, dataset=args.dataset, mode=args.ssl_mode)
                loss += args.plutchik_instance_coeff * angle_instance_loss
            else:
                angle_instance_loss = None

            eval_metric.append_classification_results(y, outputs, loss, p_loss, angle_instance_loss)
        
            if (batch_idx % 50 == 0 and batch_idx != 0) or batch_idx == len(dataloader) - 1:
                result_dict = eval_metric.classification_summary()
                logging.info(f'Fold {fold_idx} - Current {split} Loss at epoch {epoch}, step {batch_idx+1}/{len(dataloader)} {result_dict["loss"]:.3f}')
                logging.info(f'Fold {fold_idx} - Current {split} UAR at epoch {epoch}, step {batch_idx+1}/{len(dataloader)} {result_dict["uar"]:.2f}%')
                logging.info(f'Fold {fold_idx} - Current {split} WF1 at epoch {epoch}, step {batch_idx+1}/{len(dataloader)} {result_dict["mf1"]:.2f}%')
                logging.info(f'Fold {fold_idx} - Current {split} ACC at epoch {epoch}, step {batch_idx+1}/{len(dataloader)} {result_dict["acc"]:.2f}%')
                if args.pooling_mode  == 'weighted_pool':
                    logging.info(f'Fold {fold_idx} - Current {split} alpha at epoch {epoch}, step {batch_idx+1}/{len(dataloader)}: {model.text_model.alpha.item():.4f}')
                if at_barlow_align:
                    logging.info(f'Fold {fold_idx} - Current {split} Plutchik Loss at epoch {epoch}, step {batch_idx+1}/{len(dataloader)}: {result_dict["ploss"]:.3f}')
                logging.info(f'-------------------------------------------------------------------')
                
    logging.info(f'-------------------------------------------------------------------')
    result_dict = eval_metric.classification_summary()
    if wandb_store:
        if args.pooling_mode  == 'weighted_pool':
            wandb.log({
                f"{split} Loss": result_dict["loss"],
                f"{split} UAR": result_dict["uar"],
                f"{split} WF1": result_dict["mf1"],
                f"{split} ACC": result_dict["acc"],
                f"{split} alpha": model.text_model.alpha.item()
            })
        else: 
            if at_barlow_align and plutchik_instance_match:
                wandb.log({
                f"{split} Loss": result_dict["loss"],
                f"{split} Barlow align Loss": result_dict["ploss"],
                f"{split} Russel Instance Loss": result_dict["angle_instance_loss"],
                f"{split} UAR": result_dict["uar"],
                f"{split} WF1": result_dict["mf1"],
                f"{split} ACC": result_dict["acc"],
                f"{split} Learning Rate": scheduler.optimizer.param_groups[0]["lr"]
            })
            elif at_barlow_align:
                wandb.log({
                f"{split} Loss": result_dict["loss"],
                f"{split} Barlow align Loss": result_dict["ploss"],
                f"{split} UAR": result_dict["uar"],
                f"{split} WF1": result_dict["mf1"],
                f"{split} ACC": result_dict["acc"],
                f"{split} Learning Rate": scheduler.optimizer.param_groups[0]["lr"]
            })
            else:
                wandb.log({
                    f"{split} Loss": result_dict["loss"],
                    f"{split} UAR": result_dict["uar"],
                    f"{split} WF1": result_dict["mf1"],
                    f"{split} ACC": result_dict["acc"],
                    f"{split}Learning Rate": scheduler.optimizer.param_groups[0]["lr"]
                })
    if split == "Validation"and not args.warmup: 
        scheduler.step(result_dict["loss"])
    return result_dict



if __name__ == '__main__':

    datetime_save = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(datetime_save)
    start_time = time.time()

    args = parse_finetune_args()
    print('args', args)

    if args.wandb:
        wandb.init(
        project='emotion_multimodal',
        name=args.exp_dir,
        tags=[args.tag],
        config=args
        )

    log_path = os.path.join('finetune', args.dataset, args.modal, args.setting, args.exp_dir, 'app.log')

    logging.basicConfig(
        filename=log_path,
        format='%(asctime)s %(levelname)-3s ==> %(message)s', 
        level=logging.INFO, 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    print(f"log save at: {log_path}")
    
    with open("../config/config.yml", "r") as stream: config = yaml.safe_load(stream)
    args.split_dir  = str(Path(config["project_dir"]).joinpath(args.split_data_dir)) # for stt data inference 
    args.data_dir   = str(Path(config["project_dir"]).joinpath("audio"))
    args.log_dir    = str(Path(config["project_dir"]).joinpath("finetune"))

    print(f"Loda Data From: {args.split_dir}")
    # Find device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    scaler = GradScaler(enabled=(device.type == "cuda"))  
    
    if torch.cuda.is_available(): print('GPU available, use GPU')
    
    best_dict = dict()
    if args.dataset in ["iemocap", "iemocap6", "meld", "meld7"]: total_folds = 2 # 
    else: total_folds = 6
    
    for fold_idx in range(1, total_folds):
    # Read train/dev file list
        train_file_list, dev_file_list, test_file_list = load_finetune_audios(
            args.split_dir, audio_path=args.data_dir, dataset=args.dataset, fold_idx=fold_idx
        )
        # Read weights of training data
        weights = return_weights(
            args.split_dir, dataset=args.dataset, fold_idx=fold_idx, log=args.class_weight_log, normalize=args.class_weight_norm
        )
    
        # Set train/dev/test dataloader
        train_dataloader = set_finetune_dataloader(
            args, train_file_list, is_train=True
        )
        dev_dataloader = set_finetune_dataloader(
            args, dev_file_list, is_train=False
        )
        test_dataloader = set_finetune_dataloader(
            args, test_file_list, is_train=False
        )
        # Define log dir
        log_dir = Path(args.log_dir).joinpath(
            args.dataset, 
            args.modal, 
            args.setting
        )
        Path.mkdir(log_dir, parents=True, exist_ok=True)
        # Set seeds
        if args.seed >= 0 :
            set_seed(args.seed)
        else:
            set_seed(8*fold_idx) # default setting 
        
        if args.dataset   in ["iemocap"]: num_class = 4
        elif args.dataset in ['iemocap6']: num_class = 6
        elif args.dataset in ["meld7"] : num_class = 7
        elif args.dataset in ["meld"]: num_class = 4
        
        ########### Representation Learning Model ###########
        if args.audio_model in ["wav2vec2_0", "wav2vec2_0-large"]: 
            audio_model = Wav2VecWrapper(args).to(device)
            audio_dim = hid_dim_dict[args.audio_model]
            
        elif args.audio_model in ["wavlm", "wavlm-large"]:
            audio_model = WavLMWrapper(args).to(device)
            audio_dim = hid_dim_dict[args.audio_model]
            
        if args.text_model in ["roberta-base", "roberta-large"]:
            text_model = RobertaCrossAttn(args, audio_model).to(device)
            text_dim = hid_dim_dict[args.text_model]
            tokenizer = RobertaTokenizer.from_pretrained(args.text_model)

        # Audio Modal
        if args.modal == 'audio':  
            text_model = None 
            text_dim   = None  
            
        # Text Modal        
        elif args.modal == 'text':
            audio_model = None 
            audio_dim   = None  
                
        ########### Prediciton model ###########
        if args.modal in ['multimodal']:
            model = TextAudioClassifierForCrossModalAttn(audio_model=audio_model ,text_model=text_model, \
                                    audio_dim=audio_dim, text_dim=text_dim,  \
                                    hidden_dim=args.hidden_dim, num_classes=num_class, dropout_prob = args.dr, \
                                    cross_modal_atten = args.cross_modal_atten, modal = args.modal, multimodal_pooling=args.multimodal_pooling).to(device)
        else: # unimodal classifier
            model = TextAudioClassifier(audio_model=audio_model ,text_model=text_model, \
                                        audio_dim=audio_dim, text_dim=text_dim, \
                                        hidden_dim=args.hidden_dim, num_classes=num_class, dropout_prob = args.dr, \
                                        cross_modal_atten = args.cross_modal_atten, modal = args.modal).to(device)

        if args.print_verbose:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"{name}: {'trainable' if param.requires_grad else 'frozen'}")      
        
        # Read trainable params
        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        params = sum([np.prod(p.size()) for p in model_parameters])
        logging.info(f'Trainable params size: {params/(1e6):.2f} M')
        total_params = sum([np.prod(p.size()) for p in model.parameters()])
        trainable_params = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
        logging.info(f'Trainable ratio: {trainable_params / total_params * 100:.2f} %')
        
        # Define optimizer
        if args.weight_decay > 0:
            optimizer = torch.optim.Adam(
                list(filter(lambda p: p.requires_grad, model.parameters())),
                lr=args.learning_rate, 
                weight_decay=args.weight_decay,
                betas=(0.9, 0.98)
            )
        else: 
            optimizer = torch.optim.Adam(
                list(filter(lambda p: p.requires_grad, model.parameters())),
                lr=args.learning_rate, 
                betas=(0.9, 0.98)
            )
        if args.warmup: 
            num_training_steps = len(train_dataloader) * args.num_epochs
            num_warmup_steps = int(num_training_steps * args.warmup_ratio)  
            if args.warmup_mode in ['linear']:
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps
                )
            elif args.warmup_mode in ['cosine']:
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps
                )        
        else: 
            # Define scheduler, patient = 5, minimum learning rate 5e-5
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', patience=args.n_patience, factor=0.5, verbose=True, min_lr=args.min_lr
            )

        # Training steps
        best_dev_uar, best_test_uar, best_epoch = 0, 0, 0
        best_dev_acc, best_test_acc = 0, 0
        best_dev_mf1, best_test_mf1 = 0, 0
        best_dev_loss, best_test_loss = 0, 0

        if args.best_metric == 'acc':
            best_dev_metric = best_dev_acc
        elif args.best_metric == 'mf1':
            best_dev_metric = best_dev_mf1
        elif args.best_metric == 'loss':
            best_dev_metric = best_dev_loss     
        result_hist_dict = dict()
        for epoch in range(args.num_epochs):
            if args.plutchik_instance_match and not args.plutchik_ipc_grad_start_ep:
                plutchik_instance_match = True
            elif args.plutchik_instance_match and args.plutchik_ipc_grad_start_ep and epoch >= args.plutchik_ipc_grad_start_ep - 1 :
                plutchik_instance_match = True
            else:
                plutchik_instance_match = False

            train_result = train_epoch(
                train_dataloader, model, device, optimizer, weights, args.at_barlow_align, plutchik_instance_match, wandb_store=args.wandb,
                scaler=scaler,               
                grad_clip_norm=None,
                scheduler=scheduler      
            )

            dev_result = validate_epoch(
                dev_dataloader, model, device, weights, args.at_barlow_align, plutchik_instance_match, wandb_store=args.wandb
            )
            
            test_result = validate_epoch(
                test_dataloader, model, device, weights,  args.at_barlow_align, plutchik_instance_match, split="Test", wandb_store=args.wandb
            )
            
            current_metric = dev_result[f"{args.best_metric}"]     
            if args.best_metric == "loss":
                is_better = current_metric < best_dev_metric
            else:
                is_better = current_metric > best_dev_metric

            if is_better:
                best_dev_metric = current_metric

                best_dev_uar = dev_result["uar"]
                best_test_uar = test_result["uar"]

                best_dev_acc = dev_result["acc"]
                best_test_acc = test_result["acc"]

                best_dev_mf1 = dev_result["mf1"]
                best_test_mf1 = test_result["mf1"]
                
                best_dev_loss = dev_result["loss"]
                best_test_loss = test_result["loss"]

                best_dev_report = dev_result["report"]
                best_test_report = test_result["report"]
                
                best_epoch = epoch
                torch.save(model.state_dict(), str(log_dir.joinpath(f'{args.exp_dir}_fold_{fold_idx}.pt')))
                
                if args.modal in ['multimodal', 'multimodal_concat']: 
                    torch.save(model.pred_linear.state_dict(), str(log_dir.joinpath(f'{args.exp_dir}_pred_fold_{fold_idx}.pt')))
                    torch.save(model.text_model.state_dict(), str(log_dir.joinpath(f'{args.exp_dir}_{args.text_model}_fold_{fold_idx}.pt')))
                    torch.save(model.audio_model.state_dict(), str(log_dir.joinpath(f'{args.exp_dir}_{args.audio_model}_fold_{fold_idx}.pt')))
                    
                elif args.modal in ['audio']: 
                    torch.save(model.pred_linear.state_dict(), str(log_dir.joinpath(f'{args.exp_dir}_pred_fold_{fold_idx}.pt')))
                    torch.save(model.audio_model.state_dict(), str(log_dir.joinpath(f'{args.exp_dir}_{args.audio_model}_fold_{fold_idx}.pt')))
                
                elif args.modal in ['text']: 
                    torch.save(model.pred_linear.state_dict(), str(log_dir.joinpath(f'{args.exp_dir}_pred_fold_{fold_idx}.pt')))
                    torch.save(model.text_model.state_dict(), str(log_dir.joinpath(f'{args.exp_dir}_{args.text_model}_fold_{fold_idx}.pt')))

            logging.info(f'-------------------------------------------------------------------')
            logging.info(f"Fold {fold_idx} - Best train epoch {best_epoch}, best dev UAR {best_dev_uar:.2f}%, best test UAR {best_test_uar:.2f}%")
            logging.info(f"Fold {fold_idx} - Best train epoch {best_epoch}, best dev F1 {best_dev_mf1:.2f}%, best test F1 {best_test_mf1:.2f}%")
            logging.info(f"Fold {fold_idx} - Best train epoch {best_epoch}, best dev ACC {best_dev_acc:.2f}%, best test ACC {best_test_acc:.2f}%")
            logging.info(f'-------------------------------------------------------------------')
            
            # log the current result
            log_epoch_result(result_hist_dict, epoch, train_result, dev_result, test_result, log_dir, fold_idx, args.exp_dir)

        # log the best results
        log_best_result(result_hist_dict, epoch, best_dev_uar, best_dev_acc, best_test_uar, best_test_acc, log_dir, fold_idx, args.exp_dir)

        best_dict[fold_idx] = dict()
        best_dict[fold_idx]["mf1"] = best_test_mf1
        best_dict[fold_idx]["uar"] = best_test_uar
        best_dict[fold_idx]["acc"] = best_test_acc
        best_dict[fold_idx]["report"] = best_test_report
        best_dict[fold_idx]["report"] = replace_report_labels(best_dict[fold_idx]["report"], args)
        
        # save best results
        jsonString = json.dumps(best_dict, indent=4)
        jsonFile = open(str(log_dir.joinpath(f'{args.exp_dir}_{datetime_save}_results.json')), "w")
        jsonFile.write(jsonString)
        jsonFile.close()
        if args.wandb:
            wandb.log({
                "Best Dev UAR": best_dev_uar,
                "Best Dev ACC": best_dev_acc,
                "Best Test UAR": best_test_uar,
                "Best Test ACC": best_test_acc,
                "Best Test WF1": best_test_mf1
            })

    uar_list = [best_dict[fold_idx]["uar"] for fold_idx in best_dict]
    mf1_list = [best_dict[fold_idx]["mf1"] for fold_idx in best_dict]
    acc_list = [best_dict[fold_idx]["acc"] for fold_idx in best_dict]
    best_dict["average"] = dict()
    best_dict["average"]["mf1"] = np.mean(mf1_list)
    best_dict["average"]["uar"] = np.mean(uar_list)
    best_dict["average"]["acc"] = np.mean(acc_list)
    
    best_dict["std"] = dict()
    best_dict["std"]["mf1"] = np.std(mf1_list)
    best_dict["std"]["uar"] = np.std(uar_list)
    best_dict["std"]["acc"] = np.std(acc_list)
    
    end_time = time.time()
    
    # save best results
    jsonString = json.dumps(best_dict, indent=4)
    jsonFile = open(str(log_dir.joinpath(f'{args.exp_dir}_{datetime_save}_results.json')), "w")
    jsonFile.write(jsonString)
    jsonFile.write(f'Trainable params size: {params/(1e6):.2f} M ')
    jsonFile.write(excution_time(start_time, end_time))
    jsonFile.write(str(args))
    jsonFile.close()