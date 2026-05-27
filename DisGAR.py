import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel
from torch.optim import AdamW
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    roc_curve, 
    matthews_corrcoef, 
    balanced_accuracy_score
)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class TextDetectionDataset(Dataset):
    def __init__(self, csv_file: str, tokenizer, max_length: int = 2048, is_train: bool = True):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        text1 = str(self.data.iloc[idx, 0])
        raw_label = str(self.data.iloc[idx, 1]).strip()
        label1 = 0 if raw_label == "1" else 1  # 0: Human, 1: Machine/AI
        
        enc1 = self.tokenizer(text1, truncation=True, max_length=self.max_length, add_special_tokens=True)
        
        item = {
            'input_ids': enc1['input_ids'], 
            'attention_mask': enc1['attention_mask'], 
            'labels': label1
        }
        
        if self.is_train:
            text2 = str(self.data.iloc[idx, 3])
            enc2 = self.tokenizer(text2, truncation=True, max_length=self.max_length, add_special_tokens=True)
            item.update({
                'rewrite_ids': enc2['input_ids'], 
                'rewrite_mask': enc2['attention_mask'], 
                'rewrite_labels': 1 
            })
            
        return item

class PairedDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list) -> dict:
        input_ids = [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]
        attention_mask = [torch.tensor(item['attention_mask'], dtype=torch.long) for item in batch]
        labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
        
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        
        out_dict = {
            'input_ids': input_ids, 
            'attention_mask': attention_mask, 
            'labels': labels
        }
        
        if 'rewrite_ids' in batch[0]:
            rewrite_ids = [torch.tensor(item['rewrite_ids'], dtype=torch.long) for item in batch]
            rewrite_mask = [torch.tensor(item['rewrite_mask'], dtype=torch.long) for item in batch]
            rewrite_labels = torch.tensor([item['rewrite_labels'] for item in batch], dtype=torch.long)
            
            rewrite_ids = torch.nn.utils.rnn.pad_sequence(rewrite_ids, batch_first=True, padding_value=self.pad_token_id)
            rewrite_mask = torch.nn.utils.rnn.pad_sequence(rewrite_mask, batch_first=True, padding_value=0)
            
            out_dict.update({
                'rewrite_ids': rewrite_ids, 
                'rewrite_mask': rewrite_mask, 
                'rewrite_labels': rewrite_labels
            })
            
        return out_dict

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        device = features.device
        batch_size = features.shape[0]

        sim_matrix = torch.div(torch.matmul(features, features.T), self.temperature)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - sim_max.detach() 

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        loss = -mean_log_prob_pos.mean()
        return loss

class AdvancedSiameseDetector(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.base_model = AutoModel.from_pretrained(
            args.model_name_or_path, 
            torch_dtype=torch.bfloat16, 
            device_map="cuda"
        )
        
        if args.checkpoint_dir:
            print(f"Mounting the saved LoRA weights to the base model: {args.checkpoint_dir}")
            self.base_model = PeftModel.from_pretrained(
                self.base_model, args.checkpoint_dir, is_trainable=args.do_train
            )
        else:
            print(f"Initializing new LoRA adapter for training...")
            lora_config = LoraConfig(
                r=args.lora_r, 
                lora_alpha=args.lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=args.lora_dropout, 
                bias="none", 
                task_type="FEATURE_EXTRACTION"
            )
            self.base_model = get_peft_model(self.base_model, lora_config)
            
        hidden_size = self.base_model.config.hidden_size
        
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), 
            nn.GELU(), 
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, args.latent_dim)
        ).to("cuda").to(torch.bfloat16)

        self.classifier = nn.Linear(args.latent_dim, 2).to("cuda").to(torch.bfloat16)
        
        if args.checkpoint_dir:
            print(f"Loading custom Projector and Classifier weights...")
            self.projector.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "projector.pt")))
            self.classifier.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "classifier.pt")))
        
        self.loss_cls = nn.CrossEntropyLoss()
        self.loss_supcon = SupervisedContrastiveLoss(temperature=args.temperature)

    def masked_mean_pooling(self, token_features, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).to(token_features.dtype)
        sum_features = (token_features * mask_expanded).sum(dim=1)
        valid_lengths = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_features / valid_lengths

    def get_last_valid_token_feature(self, token_features, attention_mask):
        batch_size = token_features.shape[0]
        last_indices = attention_mask.sum(dim=1).long() - 1 
        last_indices = last_indices.clamp(min=0)
        batch_indices = torch.arange(batch_size, device=token_features.device)
        return token_features[batch_indices, last_indices, :]

    def forward(self, input_ids, attention_mask, labels=None, rewrite_ids=None, rewrite_mask=None, rewrite_labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        target_r = self.projector(outputs.last_hidden_state)
        
        target_r_GAA = self.masked_mean_pooling(target_r, attention_mask)
        target_r_L = self.get_last_valid_token_feature(target_r, attention_mask) 
        target_z = F.normalize(target_r_L, p=2, dim=1) 
        logits = self.classifier(target_r_GAA).to(torch.float32)

        if rewrite_ids is None: 
            loss = self.loss_cls(logits, labels) if labels is not None else None
            return loss, logits

        rewrite_outputs = self.base_model(input_ids=rewrite_ids, attention_mask=rewrite_mask)
        rewrite_r = self.projector(rewrite_outputs.last_hidden_state)
        rewrite_r_L = self.get_last_valid_token_feature(rewrite_r, rewrite_mask)
        rewrite_z = F.normalize(rewrite_r_L, p=2, dim=1)

        loss_c = self.loss_cls(logits, labels)
        
        all_z = torch.cat([target_z, rewrite_z], dim=0).to(torch.float32)
        all_labels = torch.cat([labels, rewrite_labels], dim=0)
        loss_s = self.loss_supcon(all_z, all_labels)

        total_loss = loss_c + loss_s
        return total_loss, logits


def compute_comprehensive_metrics(y_true, y_scores):
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    if len(np.unique(y_true)) < 2:
        return {"auroc": 0.0, "aupr": 0.0, "tpr_at_5": 0.0, "mcc": 0.0, "bal_acc": 0.0, "best_tre": 0.0}

    auroc = roc_auc_score(y_true, y_scores)
    aupr = average_precision_score(y_true, y_scores)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    tpr_at_5 = np.interp(0.05, fpr, tpr)
    
    best_mcc = -1.0
    best_bal_acc = 0.0
    best_tre = 0.0
    
    for thresh in thresholds:
        preds = (y_scores >= thresh).astype(int)
        mcc = matthews_corrcoef(y_true, preds)
        bal_acc = balanced_accuracy_score(y_true, preds)
        
        if mcc > best_mcc:
            best_mcc = mcc
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_tre = thresh

    return {
        "auroc": auroc,
        "aupr": aupr,
        "tpr_at_5": tpr_at_5,
        "mcc": best_mcc,
        "bal_acc": best_bal_acc,
        "best_tre": best_tre
    }

def toggle_parameters(model: nn.Module, train_lora: bool, train_heads: bool):
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = train_lora
        elif any(h in name for h in ["projector", "classifier"]):
            param.requires_grad = train_heads
        else:
            param.requires_grad = False 

def run_epoch(model, loader, optimizer=None, scheduler=None, is_train=True, device="cuda"):
    model.train() if is_train else model.eval()
    total_loss, all_scores, all_labels = 0, [], []
    
    with torch.set_grad_enabled(is_train):
        for batch in tqdm(loader, desc="Train" if is_train else "Eval"):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            rewrite_ids = batch.get('rewrite_ids').to(device) if is_train and 'rewrite_ids' in batch else None
            rewrite_mask = batch.get('rewrite_mask').to(device) if is_train and 'rewrite_mask' in batch else None
            rewrite_labels = batch.get('rewrite_labels').to(device) if is_train and 'rewrite_labels' in batch else None
            
            if is_train: 
                optimizer.zero_grad()
                
            loss, logits = model(ids, mask, labels, rewrite_ids, rewrite_mask, rewrite_labels)
            
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler: 
                    scheduler.step()
                
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=-1)[:, 1] 
            all_scores.extend(probs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(loader)
    
    if not is_train:
        metrics = compute_comprehensive_metrics(all_labels, all_scores)
        return avg_loss, metrics
    else:
        auc_score = roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else 0.0
        return avg_loss, auc_score


def main(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    collator = PairedDataCollator(pad_token_id=tokenizer.pad_token_id)
    
    train_loader = None
    if args.do_train:
        train_dataset = TextDetectionDataset(args.train_file, tokenizer, max_length=args.max_length, is_train=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
        
    eval_loader = None
    if args.do_eval:
        eval_dataset = TextDetectionDataset(args.eval_file, tokenizer, max_length=args.max_length, is_train=False)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    model = AdvancedSiameseDetector(args)

    if args.do_train:
        phases = [
            {
                "name": "Phase 1: Head Warmup (Freeze LoRA)", 
                "epochs": args.warmup_epochs, 
                "train_lora": False,   
                "train_heads": True,
                "lr_lora": 0.0,
                "lr_head": args.lr_head        
            },
            {
                "name": "Phase 2: Joint Fine-tuning (Unfreeze LoRA)", 
                "epochs": args.joint_epochs, 
                "train_lora": True,    
                "train_heads": True,
                "lr_lora": args.lr_lora,        
                "lr_head": args.lr_head
            }
        ]

        total_ep = 0
        for p in phases:
            if p['epochs'] <= 0:
                continue
                
            print(f"\n start train: {p['name']}")
            toggle_parameters(model, p['train_lora'], p['train_heads'])
            
            optimizer_grouped_parameters = []
            if p['train_lora']:
                optimizer_grouped_parameters.append({
                    "params": [param for name, param in model.named_parameters() if "lora" in name.lower() and param.requires_grad],
                    "lr": p['lr_lora']
                })
            if p['train_heads']:
                optimizer_grouped_parameters.append({
                    "params": [param for name, param in model.named_parameters() if any(h in name for h in ["projector", "classifier"]) and param.requires_grad],
                    "lr": p['lr_head']
                })
                
            if not optimizer_grouped_parameters: 
                continue
                
            optimizer = AdamW(optimizer_grouped_parameters)
            total_steps = len(train_loader) * p['epochs']
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

            for _ in range(p['epochs']):
                total_ep += 1
                
                train_loss, train_auc = run_epoch(model, train_loader, optimizer, scheduler, is_train=True)
                print(f" Global Ep {total_ep} | Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
                
                if args.do_eval and eval_loader is not None:
                    eval_loss, eval_metrics = run_epoch(model, eval_loader, optimizer=None, scheduler=None, is_train=False)
                    print(f"   Global Ep {total_ep} | Eval Loss: {eval_loss:.4f}")
                    print(f"   Eval AUROC: {eval_metrics['auroc']:.4f} | Eval AUPR: {eval_metrics['aupr']:.4f}")
                    print(f"   TPR@5%    : {eval_metrics['tpr_at_5']:.4f}")
                    print(f"   Eval MCC  : {eval_metrics['mcc']:.4f} | Eval Balanced Accuracy: {eval_metrics['bal_acc']:.4f}")
                    print(f"   Eval Tre  : {eval_metrics['best_tre']:.4f}")


                save_path = os.path.join(args.output_dir, f"epoch_{total_ep}")
                os.makedirs(save_path, exist_ok=True)
                model.base_model.save_pretrained(save_path)
                torch.save(model.projector.state_dict(), os.path.join(save_path, "projector.pt"))
                torch.save(model.classifier.state_dict(), os.path.join(save_path, "classifier.pt"))
                print(f"model saved to: {save_path}")

    elif args.do_eval:
        print("Start evaluation...")
        if not args.checkpoint_dir:
            print("--checkpoint_dir is not provided, currently using randomly initialized network layer weights! ")
            return 
            
        eval_loss, eval_metrics = run_epoch(model, eval_loader, optimizer=None, scheduler=None, is_train=False)
        print(f"   [Eval Result] Eval Loss: {eval_loss:.4f}")
        print(f"   Eval AUROC: {eval_metrics['auroc']:.4f} | Eval AUPR: {eval_metrics['aupr']:.4f}")
        print(f"   TPR@5%    : {eval_metrics['tpr_at_5']:.4f}")
        print(f"   Eval MCC  : {eval_metrics['mcc']:.4f} | Eval Balanced Accuracy: {eval_metrics['bal_acc']:.4f}")
        print(f"   Eval Tre  : {eval_metrics['best_tre']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Siamese Detector Training Script")

    parser.add_argument("--model_name_or_path", type=str, default="./model", help="base model")
    parser.add_argument("--train_file", type=str, default="./dataset/train/train.csv", help="training set path")
    parser.add_argument("--eval_file", type=str, default="./dataset/MIRAGE/polish.csv", help="test set path")
    parser.add_argument("--output_dir", type=str, default="test", help="model save path")

    parser.add_argument("--checkpoint_dir", type=str, default=None, help="model loading path")
    
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--max_length", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=4, help="batch_size")
    parser.add_argument("--warmup_epochs", type=int, default=1, help="Warm-up Stage epochs")
    parser.add_argument("--joint_epochs", type=int, default=3, help="Joint Fine-tuning Stag epochs")
    parser.add_argument("--lr_head", type=float, default=1e-4, help="top-layer learning rate")
    parser.add_argument("--lr_lora", type=float, default=1e-4, help="base model learning rate")
    
    parser.add_argument("--latent_dim", type=int, default=256, help="GAR dimension")
    parser.add_argument("--temperature", type=float, default=0.07, help="temperature parameter")
    
    parser.add_argument("--lora_r", type=int, default=8, help="r")
    parser.add_argument("--lora_alpha", type=int, default=32, help="alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="dropout")
    
    args = parser.parse_args()
    main(args)