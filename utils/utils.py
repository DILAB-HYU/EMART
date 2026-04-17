import json
import torch
import random
import numpy as np
import transformers
import argparse, logging
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

transformers.logging.set_verbosity(40)

logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

label_dict = {
    "iemocap": {"0": "neu", "1": "sad", "2": "ang", "3": "hap"},
    "meld": {"0": "neutral", "1": "sadness", "2": "anger", "3": "joy"},
    "iemocap6": {"0": "neu", "1": "sad", "2": "fru", "3": "ang", "4": "hap", "5": "exc"},
    "meld7": {"0": "neutral", "1": "sadness", "2": "anger", "3": "joy", "4": "surprise", "5": "fear", "6": "disgust"}
}
        
class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, class_counts):
        super().__init__()
        class_counts = torch.tensor(class_counts, dtype=torch.float32)
        self.priors = class_counts / class_counts.sum() 
        self.log_priors = torch.log(self.priors)

    def forward(self, logits, targets):
        # logits shape: (B, C), targets shape: (B,)
        adjusted_logits = logits - self.log_priors.to(logits.device)
        return F.cross_entropy(adjusted_logits, targets)


# https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
class Loss(nn.Module):
    def __init__(self, gamma=1, alpha=None, size_average=True):
        super(Loss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target).view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
        
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', task_type='binary', num_classes=None):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        :param num_classes: Number of classes (only required for multi-class classification)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if task_type == 'multi-class' and alpha is not None and isinstance(alpha, (list, torch.Tensor)):
            assert num_classes is not None, "num_classes must be specified for multi-class classification"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Forward pass to compute the Focal Loss based on the specified task type.
        :param inputs: Predictions (logits) from the model.
                       Shape:
                         - binary/multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape:
                         - binary: (batch_size,)
                         - multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size,)
        """
        if self.task_type == 'binary':
            return self.binary_focal_loss(inputs, targets)
        elif self.task_type == 'multi-class':
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == 'multi-label':
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'.")

    def binary_focal_loss(self, inputs, targets):
        """ Focal loss for binary classification. """
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weighting
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_class_focal_loss(self, inputs, targets):
        """ Focal loss for multi-class classification. """
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)

        # Convert logits to probabilities with softmax
        probs = F.log_softmax(inputs, dim=1)

        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs)

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_label_focal_loss(self, inputs, targets):
        """ Focal loss for multi-label classification. """
        probs = torch.sigmoid(inputs)

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weight
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
    
def get_angle(labels, dataset='iemocap'):


    if dataset == 'iemocap6':
        russel_angle = {
            0: (0.0,0.0),               # 0° neutral
            1: (9 * math.pi / 8, 1.0),  # 270° sad
            2: (7 * math.pi / 8, 1.0),  # 180° frustrated
            3: (6 * math.pi / 8, 1.0),  # 180° angry
            4: (1 * math.pi / 8,1.0),   # 90° happy
            5: (3 * math.pi / 8, 1.0),  # 90° excited
        }
    elif dataset == 'iemocap':
        russel_angle = {
            0: (0.0,0.0),               # 0° neutral
            1: (9 * math.pi / 8, 1.0),  # 270° sad
            2: (6 * math.pi / 8, 1.0),  # 180° angry
            3: (3 * math.pi / 8, 1.0),  # 90° excited
        }
    elif dataset == 'meld7':
        russel_angle = {
            0: (0.0,0.0),              # neutral
            1: (9 * math.pi / 8, 1.0), # sad
            2: (6 * math.pi / 8, 1.0), # anger       
            3: (2 * math.pi / 8, 1.0), # joy
            4: (4 * math.pi / 8, 1.0), # surprise 
            5: (5 * math.pi / 8, 1.0),  # fear
            6: (7 * math.pi / 8, 1.0)  # disgust
        }
        
    angles = torch.tensor(
        [ russel_angle[int(l)][0] for l in labels ],
        device=labels.device
    )  

    intensities = torch.tensor(
        [ russel_angle[int(l)][1] for l in labels ],
        device=labels.device
    )  
    return angles, intensities

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwinsLoss(nn.Module):
    def __init__(self, batch_size,lambd=0.0051, dim=256, device='cuda'):
        super().__init__()
        self.batch_size = batch_size
        self.lambd = lambd
        self.bn = nn.BatchNorm1d(dim, affine=False).to(device)
    def forward(self, z1, z2):
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)
        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch_size)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss


def plutchik_contrastive_loss_instance(label_inputs, pred, weak_pos=0.2, tau=0.5, dataset='iemocap6', mode='supcon'):
    b_size = label_inputs.size(0)
    eps = 1e-6

    raw_diff = torch.abs(get_angle(label_inputs, dataset)[0].unsqueeze(1) - get_angle(label_inputs, dataset)[0].unsqueeze(0))      # [B,B]

    theta = torch.min(raw_diff, 2*math.pi - raw_diff)                     # [B,B]
    phi = math.pi - theta                                                 # [B,B]
    cos_phi = torch.cos(phi)
    sin_phi = torch.abs(torch.sin(phi))

    # cosine similarity
    z = F.normalize(pred, p=2, dim=1)                 # [B,D]
    cos_orig = torch.clamp(z @ z.T, -1, 1)            # [B,B]
    cos_tilde = cos_orig * cos_phi - sin_phi * torch.sqrt(1 - cos_orig**2 + eps)  # [B,B]

    # mask
    neutral_label = 0

    eye = torch.eye(b_size, dtype=torch.bool, device=pred.device)
    pos_mask = (label_inputs.unsqueeze(1) == label_inputs.unsqueeze(0)) & ~eye   

    anchor_is_neutral = (label_inputs == neutral_label).unsqueeze(1)              
    target_is_neutral = (label_inputs == neutral_label).unsqueeze(0)             
    weak_pos_mask = (anchor_is_neutral | target_is_neutral) & (~pos_mask) & (~eye)  
    strong_pos_mask = pos_mask                                                     
    neg_mask = (~strong_pos_mask) & (~weak_pos_mask) & (~eye)                      

    if mode == "supcon":
        # positive
        exp_strong = torch.exp(cos_orig / tau) * strong_pos_mask.float()
        exp_weak = torch.exp(0.1 * cos_orig / tau) * weak_pos_mask.float()

        # negative
        exp_neg = torch.exp(cos_tilde / tau) * neg_mask.float()

        sum_pos = exp_strong.sum(1) + weak_pos * exp_weak.sum(1)
        sum_neg = exp_neg.sum(dim=1) 
        pos_count = strong_pos_mask.sum(1) + weak_pos * weak_pos_mask.sum(1) 
        log_prob = (cos_orig / tau) - torch.log((sum_pos + sum_neg).unsqueeze(1) + eps)  # [B, 2B]

        loss = (-(
            (log_prob * (strong_pos_mask.float() + weak_pos_mask.float())).sum(1)
            / (pos_count+eps)
        )).mean()

    else: 
        # positive
        exp_strong = torch.exp(cos_orig / tau) * strong_pos_mask.float()
        exp_weak = torch.exp(cos_orig / tau) * weak_pos_mask.float()

        # negative
        exp_neg = torch.exp(cos_tilde / tau) * neg_mask.float()

        sum_pos = exp_strong.sum(1) + weak_pos * exp_weak.sum(1) 
        sum_neg = exp_neg.sum(dim=1) 
        loss = (-torch.log((sum_pos + eps)/ (sum_pos + sum_neg + eps))).mean()

    return loss




def log_gpu_memory_usage(log_dict, phase="inference"):
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # bytes to GB
    reserved_memory = torch.cuda.memory_reserved() / (1024 ** 3)  # bytes to GB

    log_dict[f"{phase}_allocated_memory_GB"] = allocated_memory
    log_dict[f"{phase}_reserved_memory_GB"] = reserved_memory

    print(f"{phase} Allocated memory: {allocated_memory:.4f} GB")
    print(f"{phase} Reserved memory: {reserved_memory:.4f} GB")



def replace_report_labels(report, args):
    '''
    Saving Classificatoin report with class-wise result 
    '''
    label_map = label_dict[args.dataset]
    print(label_map)
    new_report = {}
    for key, value in report.items():
        if key in label_map:
            new_key = label_map[key]
            new_report[new_key] = value
        else:
            new_report[key] = value
    return new_report

def flat_text(text):
    flat_texts = []
    for sublist in text:
        if isinstance(sublist, list):
            if len(sublist) == 1:
                flat_texts.extend(sublist)
            else: 
                flat_texts.append(" ".join(sublist))
        else:
            flat_texts.append(sublist)
    return flat_texts


def tokenize_texts(texts, tokenizer, max_txt_len=32, truncation_side = 'right', padding=None):
    flat_txt = flat_text(texts)
    if truncation_side == 'left': 
        tokenizer.truncation_side = "left" 
    
    if padding is not None: 
        encodings = tokenizer(flat_txt, padding="max_length", truncation=False, max_length=max_txt_len, return_tensors="pt")
    else: 
        encodings = tokenizer(flat_txt, padding=True, truncation=True, max_length=max_txt_len, return_tensors="pt")
    
    input_ids = encodings.input_ids.to('cuda')
    attention_mask = encodings.attention_mask.to('cuda')
    return input_ids, attention_mask

def tokenize_texts_with_current(texts, tokenizer, max_txt_len=128, device="cuda"):
    if isinstance(texts, list) and isinstance(texts[0], list):
        texts = [" ".join(t) for t in texts]
    elif isinstance(texts, list) and isinstance(texts[0], str):
        texts = [" ".join(texts)]
    elif isinstance(texts, str):
        texts = [texts]

    input_ids_batch, attention_mask_batch = [], []

    for text in texts:
        enc = tokenizer(text, add_special_tokens=True, truncation=False, return_tensors="pt")
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]

        cur_id = tokenizer.convert_tokens_to_ids("[Current]")
        cur_positions = (input_ids == cur_id).nonzero(as_tuple=True)[0]
        if len(cur_positions) == 0:
            truncated = tokenizer(
                text,
                max_length=max_txt_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            input_ids_batch.append(truncated.input_ids[0])
            attention_mask_batch.append(truncated.attention_mask[0])
            continue
        cur_pos = cur_positions[0].item()

        if len(input_ids) > max_txt_len:
            keep_from = max(0, len(input_ids) - max_txt_len)
            if cur_pos < keep_from:
                keep_from = cur_pos
            input_ids = input_ids[keep_from:]
            attention_mask = attention_mask[keep_from:]

        if len(input_ids) < max_txt_len:
            pad_len = max_txt_len - len(input_ids)
            pad_id = tokenizer.pad_token_id
            input_ids = torch.cat([input_ids, torch.tensor([pad_id] * pad_len)])
            attention_mask = torch.cat([attention_mask, torch.tensor([0] * pad_len)])

        input_ids_batch.append(input_ids[:max_txt_len])
        attention_mask_batch.append(attention_mask[:max_txt_len])

    input_ids = torch.stack(input_ids_batch).to(device)
    attention_mask = torch.stack(attention_mask_batch).to(device)
    return input_ids, attention_mask


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def int_or_false(v):
    if v is None:       
        return False
    try:
        return int(v)   
    except ValueError:
        raise argparse.ArgumentTypeError(f"input should be an integer: {v}")


def excution_time(start_time, end_time):
    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    duration_string = f"Duration: {hours} hours, {minutes} minutes, {seconds} seconds"
    return duration_string

def set_seed(seed):
    print(f"seed: {seed}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def get_results(input_dict):
    return_dict = dict()
    return_dict["mf1"] = input_dict["mf1"]
    return_dict["uar"] = input_dict["uar"]
    return_dict["acc"] = input_dict["acc"]
    return_dict["loss"] = input_dict["loss"]
    return return_dict

def log_epoch_result(
    result_hist_dict:       dict, 
    epoch:                  int,
    train_result:           dict,
    dev_result:             dict,
    test_result:            dict,
    log_dir:                str,
    fold_idx:               int,
    exp_dir:                str,
):
    # read result
    result_hist_dict[epoch] = dict()
    result_hist_dict[epoch]["train"] = get_results(train_result)
    result_hist_dict[epoch]["dev"] = get_results(dev_result)
    result_hist_dict[epoch]["test"] = get_results(test_result)
    
    # dump the dictionary
    jsonString = json.dumps(result_hist_dict, indent=4)
    #jsonFile = open(str(log_dir.joinpath(f'{exp_dir}_fold_{fold_idx}.json')), "w")
    jsonFile = open(str(log_dir.joinpath(f'fold_{fold_idx}.json')), "w")
    jsonFile.write(jsonString)
    jsonFile.close()
    
    
def log_best_result(
    result_hist_dict:       dict, 
    epoch:                  int,
    best_dev_uar:           float,
    best_dev_acc:           float,
    best_test_uar:          float,
    best_test_acc:          float,
    log_dir:                str,
    fold_idx:               int,
    exp_dir:                str
):
    # log best result
    result_hist_dict["best"] = dict()
    result_hist_dict["best"]["dev"], result_hist_dict["best"]["test"] = dict(), dict()
    result_hist_dict["best"]["dev"]["uar"] = best_dev_uar
    result_hist_dict["best"]["dev"]["acc"] = best_dev_acc
    result_hist_dict["best"]["test"]["uar"] = best_test_uar
    result_hist_dict["best"]["test"]["acc"] = best_test_acc

    # save results for this fold
    jsonString = json.dumps(result_hist_dict, indent=4)
    #jsonFile = open(str(log_dir.joinpath(f'{exp_dir}_fold_{fold_idx}.json')), "w")
    jsonFile = open(str(log_dir.joinpath(f'fold_{fold_idx}.json')), "w")
    jsonFile.write(jsonString)
    jsonFile.close()

def parse_finetune_args():
    # parser
    parser = argparse.ArgumentParser(description='emo2vec finetune experiments')
    parser.add_argument(
        '--data_dir', 
        default='../dataset',
        type=str, 
        help='raw audio path'
    )

    parser.add_argument(
        '--model_dir', 
        default='./model',
        type=str, 
        help='model save path'
    )

    parser.add_argument(
        '--split_data_dir', 
        default='train_split',
        type=str, 
        help='train split path'
    )
    parser.add_argument(
        '--log_dir', 
        default='./finetune',
        type=str, 
        help='model save path'
    )

    parser.add_argument(
        '--learning_rate', 
        default=0.0002,
        type=float,
        help="learning rate",
    )

    parser.add_argument(
        '--num_epochs', 
        default=50,
        type=int,
        help="total training rounds",
    )
    
    parser.add_argument(
        '--optimizer', 
        default='adam',
        type=str,
        help="optimizer",
    )
    
    parser.add_argument(
        '--dataset',
        default="iemocap",
        type=str,
        help="Dataset name",
    )
    
    parser.add_argument(
        '--audio_duration', 
        default=6,
        type=int,
        help="audio length for training"
    )
    parser.add_argument(
        '--num_layers',
        default=1,
        type=int,
        help="num of layers",
    )

    parser.add_argument(
        '--hidden_size',
        default=256,
        type=int,
        help="hidden size",
    )
    
    parser.add_argument(
        '--finetune_audio', 
        default=True,
        type=str2bool, 
        help='Whether to finetune audio model'
    )
    
    parser.add_argument(
        '--hidden_dim', 
        default=256,
        type=int, 
        help='hidden dim of prediction model  '
    )


    parser.add_argument(
        '--downstream', 
        default=False,
        type=bool, 
        help='Flag to use downstream model'
    )
    
    parser.add_argument(
        '--exp_dir', 
        default="exp",
        type=str, 
        help='Exp dir'
    )
    
    parser.add_argument(
        '--max_audio_len',  # DON'T CHANGE 
        default=6,
        type=int, 
        help='max_audio_len'
    )

    parser.add_argument(
        '--max_txt_len', 
        default=128, 
        type=int, 
        help='max_txt_len'
    )
    

    parser.add_argument(
        '--ws', 
        default=False,
        type=str2bool, 
        help='weighted sum feature'
    )

    parser.add_argument(
        '--wg', 
        default=False,
        type=str2bool, 
        help='weighted gate'
    )
    
    parser.add_argument(
        '--cross_modal_atten', 
        default=False,
        type=str2bool, 
        help='weighted gate'
    )

    parser.add_argument(
        '--modal', 
        default="audio",
        type=str, 
        help='[audio, text, multimodal]'
    )
    
    parser.add_argument(
        '--audio_model', 
        default="None",
        type=str, 
        help="Audio Modal Representation model"
    )
    
    parser.add_argument(
        '--text_model', 
        default="None",
        type=str, 
        help='Text Modal Representation model'
    )

    parser.add_argument(
        '--print_verbose', 
        default=True,
        type=str2bool, 
        help='print verbose'
    )

    # LoRA arguments
    parser.add_argument(
        '--lora_alpha', 
        default=16, 
        type=int, 
        help='LoRA alpha for weight scaling'
    )

    parser.add_argument(
        '--lora_dropout', 
        default=0.1, 
        type=float, 
        help='Dropout probability for LoRA layers'
    )

    parser.add_argument(
        '--lora_target_modules', 
        default="dense", 
        type=str, 
        help='List of target modules for LoRA'
    )

    parser.add_argument(
        '--finetune_roberta', 
        default=True,
        type=str2bool, 
        help='finetune roberta'
    )
    
    parser.add_argument(
        '--dr', 
        default=0.5,
        type=float, 
        help='dropout ratio'
    )
    
    parser.add_argument(
        '--self_attn', 
        default=False,
        type=str2bool, 
        help='self atten before cross modal attn'
    )

    parser.add_argument(
        '--num_hidden_layers', 
        default=None,
        type=int, 
        help='number of hidden layers using repersentation, if none, all layers are used'
    )

    parser.add_argument(
        '--batch_size', 
        default=32,
        type=int, 
        help='number of batch size'
    )

    parser.add_argument(
        '--truncation_side',
        default='right',
        type=str,
        help='tokenizer truncation side. right (truncates from the end), left (truncates from the beginning)'
    )

    parser.add_argument(
        '--tokenize_mode',
        default='default',
        type=str,
        help='Whether to use adaptive tokenize or not [default, adaptive]'
    )

    parser.add_argument(
        '--num_workers',
        default=6,
        type=int,
        help='num_workers for dataloader'
    )


    parser.add_argument(
        '--min_lr',
        default=2e-5,
        type=float,
        help='lr for schedular'
    )
    

    parser.add_argument(
        '--wandb',
        default=False,
        type=str2bool,
        help='store wandb log or not'
    )
    parser.add_argument(
        '--tag', 
        default="team", 
        type=str, 
        help='exp tag [seong, lee, team]'
    )

    parser.add_argument(
        '--plutchik_metric',
        nargs='?',
        const=None,
        default=None,
        help='dist or theta_r'
    )

    parser.add_argument(
        '--weak_pos', 
        default=0.2,
        type=float, 
        help='weak pos in plutchik contrastive loss'
    )
    parser.add_argument(
        '--plutchik_align',
        default=False,
        type=str2bool,
        help='use plutchik alignment loss'
    )
    parser.add_argument(
        '--plutchik_align_coeff', 
        default=1.0,
        type=float, 
        help='coefficient of plutchik align loss (step1)'
    )

    parser.add_argument(
        '--plutchik_tau', 
        default=0.5,
        type=float
        )

    parser.add_argument(
        '--focal_loss', 
        default=False,
        type=str2bool, 
        help='whether to use focal loss. default = False (Cross Entropy Loss)'
    )
    parser.add_argument(
        '--ssl_coeff', 
        default=1.0,
        type=float, 
        help='coefficient of contrastive loss'
    )
    parser.add_argument(
        '--ssl_mode', 
        default='supcon',
        type=str, 
        help='contrastive loss mode [supcon, ntxent_sum]'
    )
    
    parser.add_argument(
        '--plutchik_instance_match', 
        default=False,
        type=str2bool, 
        help='whether to use plutchik instance match loss. (step 2)'
    )

    parser.add_argument(
        '--plutchik_instance_coeff', 
        default=1.0,
        type=float, 
        help='coefficient of plutchik instance loss (step2)'
    )

    parser.add_argument(
        '--at_barlow_align',
        default=False,
        type=str2bool,
        help='use at barlow align loss'
    )
    parser.add_argument(
        '--at_barlow_coeff', 
        default=0.01,
        type=float, 
        help='coefficient of at barlow align loss (step1)'
    )

    parser.add_argument(
        "--plutchik_ipc_grad_start_ep",
        nargs="?",              
        const=False,            
        default=False,          
        type=int_or_false       
    )



    parser.add_argument(
        '--weight_decay', 
        default=-1,
        type=float, 
        help='weight decay'
    )

    parser.add_argument(
        '--warmup_ratio', 
        default=0.01,
        type=float, 
        help='warmup'
    )
    
    parser.add_argument(
        '--warmup', 
        default=False,
        type=str2bool, 
        help='whether to use warmup or not'
    )
        
    parser.add_argument(
        '--warmup_mode', 
        default='linear',
        type=str, 
        help='warmup mode [cosine, linear]'
    )
    parser.add_argument(
        '--pooling_mode', 
        default=None,
        type=str, 
        help='Pooling mode for roberta [None, curr_only, weighted pool]'
    ) 
    
    parser.add_argument(
        '--clamp', 
        default=1e-6,
        type=float, 
        help='clamp'
    )
    
    parser.add_argument(
        '--speaker_mode', 
        default=None,
        type=str, 
        help='speaker_mode for additional special token [without_name, spk_idx, self_other]'  
    )
    
    parser.add_argument(
        '--best_metric', 
        default="mf1",
        type=str, 
        help='Metric for model selection e.g., [acc, mf1, loss]'  
    )
    
    parser.add_argument(
        '--padding', 
        default=None, 
        type=str, 
        help='tokenizer padding mode default mode is longest, options: [max_length]'
    )

    parser.add_argument(
        '--n_patience', 
        type=int, 
        default=5,
        help='n_patience for schedular'
    )

    parser.add_argument(
        '--load_pt', 
        action='store_true',
        help='load preprocessed .pt data'
    )
    
    parser.add_argument(
        '--balanced_ce', 
        default=False,
        type=str2bool, 
        help='balanced_ce'
    )
    
    parser.add_argument(
        '--weighted_ce', 
        default=False,
        type=str2bool, 
        help='cross entropy with class weight'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=-1
    )
    
    parser.add_argument(
        '--class_weight_log', 
        default=False,
        type=str2bool, 
    )

    parser.add_argument(
        '--class_weight_norm', 
        default=True,
        type=str2bool, 
    )
        
    parser.add_argument(
        '--multimodal_pooling', 
        default=None,
        type=str, 
        help='pooling mode for fusion feature, if none, default is same with pooling mode'  
    )
    
    args = parser.parse_args()
    setting = f'lr{str(args.learning_rate).replace(".", "")}_ep{args.num_epochs}'
    args.setting = setting
    
    return args

