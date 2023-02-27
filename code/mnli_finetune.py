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
from mnli_pre_training import EntailModel, train, evaluate, epoch_time

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"


def main():
    train_dict = pickle.load(open("../data/train_dict_formatted_value_dataset.pkl", "rb"))

    batch_size = 32
    train_data = TensorDataset(torch.tensor(train_dict["train_x"]), torch.tensor(train_dict["train_y"]))
    valid_data = TensorDataset(torch.tensor(train_dict["valid_x"]), torch.tensor(train_dict["valid_y"]))

    train_dl = DataLoader(train_data, batch_size=batch_size, sampler=RandomSampler(train_data), num_workers=2)
    valid_dl = DataLoader(valid_data, batch_size=batch_size, sampler=SequentialSampler(valid_data), num_workers=2)

    N_EPOCHS = 20
    best_valid_loss = float('inf')
    model_name = "roberta_values_finetuning.pt"
    early_stopping = 3
    device = torch.device("cuda:{}".format(0)) if torch.cuda.is_available() else "cpu"
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    roberta = RobertaModel.from_pretrained("roberta-base")
    model = EntailModel(roberta, tokenizer).to(device)

    state_dict = torch.load("./roberta_mnli_pretraining.pt")
    model.load_state_dict(state_dict)
    print("Loaded pre-trained MNLI weights!")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    t_loss, v_loss, early_stopping_marker = [], [], []

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


if __name__ == '__main__':
    main()

# 2060it [01:54, 17.96it/s]VALIDATION STATS:
#                precision    recall  f1-score   support
#
#            0       0.82      0.75      0.78     32950
#            1       0.77      0.83      0.80     32950
#
#     accuracy                           0.79     65900
#    macro avg       0.79      0.79      0.79     65900
# weighted avg       0.79      0.79      0.79     65900
#
# FOUND BEST MODEL!
# SAVING BEST MODEL!
# Epoch: 0 | Time: 19m 28s
# 	Train Total Loss: 0.411 | Val Total Loss: 0.466
