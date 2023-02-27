import pandas as pd
import numpy as np
from collections import Counter
import json
from tqdm import tqdm
import random
import pickle
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler, TensorDataset, RandomSampler
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import torch
import torch.nn as nn
from datasets import load_dataset
import time
from torch.utils.data import DataLoader

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"


class EntailModel(nn.Module):
    def __init__(self, base_model, tokenizer):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.ff = nn.Linear(768, 1)

    def forward(self, input_ids):
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long().to(input_ids.device)
        op = self.base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return self.ff(op["pooler_output"])


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, optimizer, dataloader, device):
    model.train()
    ep_t_loss, batch_num = 0, 0
    loss_fct = nn.BCEWithLogitsLoss()

    for ix, batch in tqdm(enumerate(dataloader)):
        batch = tuple(t.to(device) for t in batch)
        input_ids, labels = batch
        optimizer.zero_grad()

        output_dct = model(input_ids=input_ids)

        loss = loss_fct(output_dct.view(-1), labels.float().view(-1))
        loss.backward()
        optimizer.step()

        batch_num += 1
        ep_t_loss += loss.item()
    return ep_t_loss / batch_num


def evaluate(model, dataloader, device, threshold=0.5):
    model.eval()
    ep_t_loss, batch_num = 0, 0
    loss_fct = nn.BCEWithLogitsLoss()
    preds, actual = [], []
    for ix, batch in tqdm(enumerate(dataloader)):
        batch = tuple(t.to(device) for t in batch)
        input_ids, labels = batch
        with torch.no_grad():
            output_dct = model(input_ids=input_ids)

        loss = loss_fct(output_dct.view(-1), labels.float().view(-1))

        batch_num += 1
        ep_t_loss += loss.item()
        preds.extend((torch.sigmoid(output_dct.view(-1)) >= threshold).long().tolist())
        actual.extend(labels.view(-1).tolist())
    print("VALIDATION STATS:\n", classification_report(actual, preds, zero_division=0))
    return ep_t_loss / batch_num


def execute(model, train_dl, valid_dl, model_name, device, early_stopping=3):
    t_loss, v_loss, early_stopping_marker = [], [], []
    N_EPOCHS = 20
    best_valid_loss = float('inf')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for epoch in range(N_EPOCHS):
        print("Epoch: {}, Training ...\n".format(epoch))
        start_time = time.time()

        tr_l = train(model, optimizer, train_dl, device)
        t_loss.append(tr_l)

        print("Epoch: {}, Evaluating ...\n".format(epoch))
        vl_l = evaluate(model, valid_dl, device)
        v_loss.append(vl_l)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if vl_l <= best_valid_loss:
            best_valid_loss = vl_l
            print("FOUND BEST MODEL!")
            print("SAVING BEST MODEL!")
            torch.save(model.state_dict(), model_name)
            early_stopping_marker.append(False)
        else:
            early_stopping_marker.append(True)
        print(f'Epoch: {epoch} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Total Loss: {tr_l:.3f} | Val Total Loss: {vl_l:.3f}')
        if all(early_stopping_marker[-early_stopping:]) and len(early_stopping_marker) >= early_stopping:
            print("Early stopping training as the Validation loss did NOT improve for last " + \
                  str(early_stopping) + " iterations.")
            break


def get_dataloaders(filename):
    train_dict = pickle.load(open(filename, "rb"))
    batch_size = 32
    train_data = TensorDataset(torch.tensor(train_dict["train_x"]), torch.tensor(train_dict["train_y"]))
    valid_data = TensorDataset(torch.tensor(train_dict["valid_x"]), torch.tensor(train_dict["valid_y"]))

    train_dl = DataLoader(train_data, batch_size=batch_size, sampler=RandomSampler(train_data), num_workers=2)
    valid_dl = DataLoader(valid_data, batch_size=batch_size, sampler=SequentialSampler(valid_data), num_workers=2)
    return train_dl, valid_dl


def main():
    pretrain_model_name = "./roberta_mnli_pretraining.pt"
    finetune_model_name = "./roberta_values_finetuning.pt"
    pretrain_data = "../data/train_dict_formatted.pkl"
    finetune_data = "../data/train_dict_formatted_value_dataset.pkl"

    device = torch.device("cuda:{}".format(0)) if torch.cuda.is_available() else "cpu"
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    roberta = RobertaModel.from_pretrained("roberta-base")
    pretrain_model = EntailModel(roberta, tokenizer).to(device)

    print("STARTING PRE-TRAINING")
    pretrain_train_dl, pretrain_valid_dl = get_dataloaders(pretrain_data)
    execute(pretrain_model, pretrain_train_dl, pretrain_valid_dl, pretrain_model_name, device)
    print("DONE PRE-TRAINING")

    state_dict = torch.load(pretrain_model_name)
    pretrain_model.load_state_dict(state_dict)
    print("Loaded pre-trained MNLI weights!")
    print("STARTING FINE-TUNING")
    finetune_train_dl, finetune_valid_dl = get_dataloaders(finetune_data)
    execute(pretrain_model, finetune_train_dl, finetune_valid_dl, finetune_model_name, device)
    print("DONE FINE-TUNING")


if __name__ == '__main__':
    main()
# RESULT
# 12272it [1:28:25,  2.31it/s]Epoch: 0, Evaluating ...
#
#
# 614it [00:43, 13.97it/s]VALIDATION STATS:
#                precision    recall  f1-score   support
#
#            0       0.94      0.93      0.94     12705
#            1       0.88      0.89      0.89      6942
#
#     accuracy                           0.92     19647
#    macro avg       0.91      0.91      0.91     19647
# weighted avg       0.92      0.92      0.92     19647
#
# FOUND BEST MODEL!
# SAVING BEST MODEL!
# Epoch: 0 | Time: 89m 10s
# 	Train Total Loss: 0.258 | Val Total Loss: 0.207
