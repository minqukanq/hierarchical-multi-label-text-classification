import time
import os
import shutil
import logging
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup)

from models.harnn import HARNN
from utils.data_loader import TextDataset
from utils import data_helper as dh


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
        self.MSELoss = nn.MSELoss()

    def forward(self, first_logits, second_logits, third_logits, fourth_logits, global_logits,
                first_scores, second_scores, input_y_first, input_y_second, input_y_third,
                input_y_fourth, input_y):
        # Local Loss
        losses_1 = self.BCEWithLogitsLoss(first_logits, input_y_first.float())
        losses_2 = self.BCEWithLogitsLoss(second_logits, input_y_second.float())
        losses_3 = self.BCEWithLogitsLoss(third_logits, input_y_third.float())
        losses_4 = self.BCEWithLogitsLoss(fourth_logits, input_y_fourth.float())
        local_losses = losses_1 + losses_2 + losses_3 + losses_4

        # Global Loss
        global_losses = self.BCEWithLogitsLoss(global_logits, input_y.float())

        # Hierarchical violation Loss
        return local_losses + global_losses

def train(args):
    logging.info("Loading Data...")
    train_dataset = TextDataset(args, args.train_file_path)
    valid_dataset = TextDataset(args, args.test_file_path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    
    vocab_size = dh.get_vocab_size(args.train_file_path, args.test_file_path)
    
    logging.info("Init nn...")
    net = HARNN(num_classes_list=args.num_classes_layer, total_classes=args.total_classes, vocab_size=vocab_size,
                    embedding_size=args.embedding_size, lstm_hidden_size=args.lstm_hidden_size,
                    attention_unit_size=args.attention_unit_size,
                    fc_hidden_size=args.fc_hidden_size, beta=args.beta,
                    drop_prob=args.drop_prob)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(args.device)
    
    criterion = Loss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=args.l2_reg_lambda, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=len(train_loader)*args.epochs)

    logging.info("Training...")
    is_best = 0
    for epoch in range(args.epochs):
        train_loss = 0.0
        train_cnt = 0

        for train_iter, (x_train, y_train, y_train_0, y_train_1, y_train_2, y_train_3) in enumerate(train_loader):
            x_train, y_train, y_train_0, y_train_1, y_train_2, y_train_3 = \
                [i.to(args.device) for i in [x_train, y_train, y_train_0, y_train_1, y_train_2, y_train_3]]

            _, outputs = net(x_train)
            loss = criterion(outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6],
                             y_train_0, y_train_1, y_train_2, y_train_3, y_train)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            train_loss += loss.item()
            train_cnt += x_train.size()[0]
            
            if train_iter % args.print_every == 0:
                logging.info('[%d, %5d] loss: %.3f' % (epoch + 1, train_cnt + 1, train_loss / train_cnt))
            
        if epoch % args.evaluate_every == 0:
            val_loss = 0.0
            val_cnt = 0
            eval_pre_tk = [0.0 for _ in range(args.top_num)]
            eval_rec_tk = [0.0 for _ in range(args.top_num)]
            eval_F_tk = [0.0 for _ in range(args.top_num)]
            true_onehot_labels = []
            predicted_onehot_scores = []
            predicted_onehot_labels_ts = []
            predicted_onehot_labels_tk = [[] for _ in range(args.top_num)]
            for x_val, y_val, y_val_0, y_val_1, y_val_2, y_val_3 in val_loader:
                x_val, y_val, y_val_0, y_val_1, y_val_2, y_val_3 = \
                    [i.to(args.device) for i in [x_val, y_val, y_val_0, y_val_1, y_val_2, y_val_3]]
                scores, outputs = net(x_val)
                scores = scores[0]
                loss = criterion(outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6],
                                 y_val_0, y_val_1, y_val_2, y_val_3, y_val)
                val_loss += loss.item()
                val_cnt += x_val.size()[0]
                # Prepare for calculating metrics
                for onehot_labels in y_val:
                    true_onehot_labels.append(onehot_labels.tolist())
                for onehot_scores in scores:
                    predicted_onehot_scores.append(onehot_scores.tolist())
                # Predict by threshold
                batch_predicted_onehot_labels_ts = \
                    dh.get_onehot_label_threshold(scores=scores.cpu().detach().numpy(), threshold=args.threshold)
                for onehot_labels in batch_predicted_onehot_labels_ts:
                    predicted_onehot_labels_ts.append(onehot_labels)
                # Predict by topK
                for num in range(args.top_num):
                    batch_predicted_onehot_labels_tk = \
                        dh.get_onehot_label_topk(scores=scores.cpu().detach().numpy(), top_num=num + 1)
                    for onehot_labels in batch_predicted_onehot_labels_tk:
                        predicted_onehot_labels_tk[num].append(onehot_labels)

            # Calculate Precision & Recall & F1
            eval_pre_ts = precision_score(y_true=np.array(true_onehot_labels),
                                          y_pred=np.array(predicted_onehot_labels_ts), average='micro')
            eval_rec_ts = recall_score(y_true=np.array(true_onehot_labels),
                                       y_pred=np.array(predicted_onehot_labels_ts), average='micro')
            eval_F_ts = f1_score(y_true=np.array(true_onehot_labels),
                                 y_pred=np.array(predicted_onehot_labels_ts), average='micro')
            # Calculate the average AUC
            eval_auc = roc_auc_score(y_true=np.array(true_onehot_labels),
                                     y_score=np.array(predicted_onehot_scores), average='micro')
            # Calculate the average PR
            eval_prc = average_precision_score(y_true=np.array(true_onehot_labels),
                                               y_score=np.array(predicted_onehot_scores), average='micro')
            is_best = eval_prc > best_auprc
            best_auprc = max(eval_prc, best_auprc)

            for num in range(args.top_num):
                eval_pre_tk[num] = precision_score(y_true=np.array(true_onehot_labels),
                                                   y_pred=np.array(predicted_onehot_labels_tk[num]), average='micro')
                eval_rec_tk[num] = recall_score(y_true=np.array(true_onehot_labels),
                                                y_pred=np.array(predicted_onehot_labels_tk[num]), average='micro')
                eval_F_tk[num] = f1_score(y_true=np.array(true_onehot_labels),
                                          y_pred=np.array(predicted_onehot_labels_tk[num]), average='micro')
            logging.info("All Validation set: Loss {0:g} | AUC {1:g} | AUPRC {2:g}"
                        .format(val_loss / val_cnt, eval_auc, eval_prc))
            logging.info("Predict by threshold: Precision {0:g}, Recall {1:g}, F {2:g}"
                        .format(eval_pre_ts, eval_rec_ts, eval_F_ts))
            logging.info("Predict by topK:")
            for num in range(args.top_num):
                logging.info("Top{0}: Precision {1:g}, Recall {2:g}, F {3:g}"
                            .format(num + 1, eval_pre_tk[num], eval_rec_tk[num], eval_F_tk[num]))

        if epoch % args.checkpoint_every == 0:
            timestamp = str(int(time.time()))
            save_checkpoint({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auprc': best_auprc,
            }, is_best, filename=os.path.join(os.path.curdir, "model", "epoch%d.%s.pth" % (epoch, timestamp)))

    logging.info('Finished Training.')

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.deterministic = False

def main():
    logging.basicConfig(
    filename='./logs/harnn-pytorch.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser()
#     args = parser.parse_args()
    args = parser.parse_args(args=[])
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.seed = 42
    set_seed(args.seed)

    args.num_classes_layer = [9, 128, 661, 8364]
    args.total_classes = 9162

    args.train_file_path = 'data/train_sample.json'
    args.test_file_path = 'data/test_sample.json'
    args.valid_file_path = 'data/validation_sample.json'

    args.print_every = 1
    args.evaluate_every = 1
    args.checkpoint_every = 1

    args.embedding_size = 400
    args.seq_length = 256

    
    args.batch_size = 2
    args.epochs = 20
    args.max_grad_norm = 0.1
    args.drop_prob = 0.5
    args.l2_reg_lambda = 0
    args.learning_rate = 5e-5
    args.beta = 0.3

    args.lstm_hidden_size = 256
    args.fc_hidden_size = 256

    args.attention_unit_size = 100
    
    args.threshold = 0.5
    args.top_num = 5
    args.best_auprc = 0


    train(args)

if __name__=='__main__':
    main()
