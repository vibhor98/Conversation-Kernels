
import os
import math
import time
import torch
import random
import numpy as np
import pandas as pd
import pickle as pkl

from sentence_transformers import InputExample
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from augmented_encoder import ContextAugmentedEncoder
from imblearn.over_sampling import SMOTEN


def save_checkpoint(save_path, model, optimizer):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_f1': best_val_f1,
        'best_epoch': best_epoch
    }, save_path)

    print('Saved model checkpoint to {}.'.format(save_path))


def batch_to_device(batch, target_device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


def smart_batching_collate(batch):
    num_texts = len(batch[0].texts)
    texts = [[] for _ in range(num_texts+1)]
    labels = []

    for example in batch:
        for idx, text in enumerate(example.texts):
            texts[idx].append(text)
        texts[-1].append(' [SEP] '.join(example.texts))
        labels.append(example.label)

    labels = torch.tensor(labels)

    sentence_features = []
    for idx in range(num_texts+1):
        to_tokenize = [[str(s).strip().lower() for s in texts[idx]]]
        tokenized = tokenizer(*to_tokenize, pad_to_max_length=True, truncation='longest_first', return_token_type_ids=True,
            return_tensors='pt', max_length=256)
        sentence_features.append(tokenized)

    return sentence_features, labels


def do_check_and_update_metrics(val_f1, val_acc, epoch, losses, model, optimizer):
    global best_val_f1
    global best_epoch
    if val_f1 >= best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch
        save_checkpoint('./checkpoints/' + str(epoch), model, optimizer)

    new_metrics = {
        'epoch': epoch,
        'train_loss': sum(losses) / len(losses),
        'val_f1': val_f1,
        'val_acc': val_acc
    }
    print('epoch {}: training loss {:.4f}, '
        'validation macro-f1: {:.4f}, accuracy: {:.4f} (history best: {:.4f} at epoch {}). '
            .format(new_metrics['epoch'], new_metrics['train_loss'],
                new_metrics['val_f1'], new_metrics['val_acc'], best_val_f1, best_epoch))


def over_sampling(subsample):
    new_sample = []
    for sample in subsample:
        if sample[0].label == 0:
            choice = random.choices([True, False], weights=[0.5, 0.5], k=1)     # 0.8, 0.2
            if choice[0] == True:
                new_sample.append(sample)
        elif sample[0].label == 1 or sample[0].label == 2:
            new_sample.append(sample)
            # new_sample.append(sample)
        elif sample[0].label == 3:
            new_sample.append(sample)
            new_sample.append(sample)
    return new_sample


def smote_oversampling(sample_df):
    X = sample_df.drop('label', axis=1)
    y = sample_df['label']
    smote = SMOTEN(sampling_strategy='auto', k_neighbors=5, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    X_resampled['label'] = y_resampled
    return X_resampled


def ancestor_window(data, node_id, window_size):
    sentences = [data['nodes'][node_id]['text']]
    label = data['nodes'][node_id]['category']
    for i in range(window_size):
        parent_id = data['edges'][node_id]
        if not math.isnan(parent_id) and parent_id in data['nodes']:
            sentences.append(data['nodes'][parent_id]['text'])
            node_id = parent_id
        else:
            sentences.append('')
    return sentences, label


def sibling_window(data, node_id, child_edges, window_size):
    sentences = [data['nodes'][node_id]['text']]
    label = data['nodes'][node_id]['category']
    parent_id = data['edges'][node_id]
    if not math.isnan(parent_id) and parent_id in data['nodes']:
        siblings = child_edges[parent_id]
        indx = 0
        for sib_id in siblings:
            if indx < window_size and sib_id != node_id:
                sentences.append(data['nodes'][sib_id]['text'])
                indx += 1
    if len(sentences) < window_size+1:
        for _ in range(window_size+1-len(sentences)):
            sentences.append('')
    return sentences, label


def children_window(data, node_id, child_edges, window_size):
    sentences = [data['nodes'][node_id]['text']]
    label = data['nodes'][node_id]['category']
    if node_id in child_edges:
        children = child_edges[node_id]
        for child_id in children:
            if not math.isnan(child_id) and child_id in data['nodes']:
                sentences.append(data['nodes'][child_id]['text'])
                if len(sentences) == window_size+1:
                    break
    if len(sentences) < window_size+1:
        for _ in range(window_size+1-len(sentences)):
            sentences.append('')
    return sentences, label


def one_hop_window(data, node_id, child_edges, window_size):
    sentences = [data['nodes'][node_id]['text']]
    label = data['nodes'][node_id]['category']
    # parent_id
    parent_id = data['edges'][node_id]
    if not math.isnan(parent_id) and parent_id in data['nodes']:
        sentences.append(data['nodes'][parent_id]['text'])

    # children
    if node_id in child_edges:
        children = child_edges[node_id]
        for child_id in children:
            if not math.isnan(child_id) and child_id in data['nodes']:
                sentences.append(data['nodes'][child_id]['text'])
                if len(sentences) == window_size+1:
                    break

    if len(sentences) < window_size+1:
        for _ in range(window_size+1-len(sentences)):
            sentences.append('')
    return sentences, label


def two_hop_window(data, node_id, child_edges, window_size):
    sentences = [data['nodes'][node_id]['text']]
    label = data['nodes'][node_id]['category']

    parent_id = data['edges'][node_id]
    if not math.isnan(parent_id) and parent_id in data['edges']:
        parent_parent_id = data['edges'][parent_id]
        if not math.isnan(parent_parent_id) and parent_parent_id in data['nodes']:
            sentences.append(data['nodes'][parent_parent_id]['text'])

    if node_id in child_edges:
        children = child_edges[node_id]
        for child_id in children:
            if len(sentences) == window_size+1:
                break
            if not math.isnan(child_id) and child_id in child_edges:
                child_child_ids = child_edges[child_id]
                for child_child_id in child_child_ids:
                    if not math.isnan(child_child_id) and child_child_id in data['nodes']:
                        sentences.append(data['nodes'][child_child_id]['text'])
                        if len(sentences) == window_size+1:
                            break

    if len(sentences) < window_size+1:
        for _ in range(window_size+1-len(sentences)):
            sentences.append('')
    return sentences, label


category_dict = {'Insightful': 1, 'Informative': -1, 'Interesting': -1, 'Funny': 0,
    'Troll': 0, 'Flamebait': 0, 'Offtopic': 0, 'Redundant': 0}


tokenizer = AutoTokenizer.from_pretrained("roberta-base")

device = torch.device('cuda:1')

# Split dataset into train and test set.
dataset_path = './slashdot_graphs/'

def preprocess_data(dataset_path):
    files = os.listdir(dataset_path)
    num_pos_labels = 0
    dataset_samples = []
    labels = []
    num_comments = 0

    for indx, file in enumerate(files):
        if indx >= 0:
            # print('Processing', indx, file)
            data = pkl.load(open(dataset_path + file, 'rb'))
            num_comments += len(data['nodes'])

            # Just for Random Walk.
            child_edges = {}
            for node_id in data['nodes'].keys():
                key = data['edges'][node_id]
                if key in child_edges:
                    child_edges[key].append(node_id)
                else:
                    child_edges[key] = [node_id]

            for node_id in data['nodes'].keys():
                if data['nodes'][node_id]['category'] == data['nodes'][node_id]['category']:
                    sentences, label = ancestor_window(data, node_id, 3)
                    dataset_samples.append(InputExample(texts=sentences, label=category_dict[label]))

                    sentences, label = sibling_window(data, node_id, child_edges, 3)
                    dataset_samples.append(InputExample(texts=sentences, label=category_dict[label]))

                    sentences, label = children_window(data, node_id, child_edges, 3)
                    dataset_samples.append(InputExample(texts=sentences, label=category_dict[label]))

                    if category_dict[label] == 0:
                        num_pos_labels += 1
    print('Total no. of comments:', num_comments)
    return num_pos_labels, dataset_samples

num_pos_labels, dataset_samples = preprocess_data(dataset_path)

print('No. of positive labels:', num_pos_labels)
print(len(dataset_samples))


dataset_subsample = []
# Subsample dataset based on the given class "1"
for i in range(0, len(dataset_samples), 3):
    if dataset_samples[i].label == 1 and num_pos_labels >= 0:
        num_pos_labels -= 1
        dataset_subsample.append([dataset_samples[i], dataset_samples[i+1], dataset_samples[i+2]])
    elif dataset_samples[i].label == 0:
        dataset_subsample.append([dataset_samples[i], dataset_samples[i+1], dataset_samples[i+2]])


random.shuffle(dataset_subsample)
dataset_sample_restr = []
for sample in dataset_subsample:
    dataset_sample_restr.extend(sample)

print(len(dataset_sample_restr))

rem = math.ceil(len(dataset_sample_restr) * 0.8) % 2    #3
train_split = math.ceil(len(dataset_sample_restr) * 0.8) - rem

# rem = math.ceil(len(dataset_sample_restr) * 0.9) % 2    #3
# val_split = math.ceil(len(dataset_sample_restr) * 0.9) - rem

train_samples = dataset_sample_restr[ : train_split]
val_samples = dataset_sample_restr[train_split : ]   #val_split]
# test_samples = dataset_sample_restr[val_split : ]
train_dataloader = DataLoader(train_samples, shuffle=False, batch_size=3, collate_fn=smart_batching_collate)
val_dataloader = DataLoader(val_samples, shuffle=False, batch_size=3, collate_fn=smart_batching_collate)
test_dataloader = DataLoader(dataset_sample_restr, shuffle=False, batch_size=3, collate_fn=smart_batching_collate)


EPOCHS = 3
n_classes = 2
learning_rate = 1e-5

best_val_f1 = 0.0
best_epoch = -1

model = ContextAugmentedEncoder.from_pretrained('roberta-base')
model.cuda(1)
model.train()

loss_fcn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(EPOCHS):
    print('EPOCH', epoch)
    t0 = time.time()
    pred_all = []
    labels_all = []
    loss_all = []
    for features, labels in train_dataloader:
        labels = labels.to(device)
        features = list(map(lambda batch: batch_to_device(batch, device), features))

        # forward
        logits = model(features)
        loss = loss_fcn(logits, labels[0])
        pred = torch.argmax(logits)

        pred_all.append(pred.detach().cpu().numpy())
        labels_all.append(labels[0].detach().cpu().numpy())
        loss_all.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    dur = time.time() - t0

    acc = accuracy_score(pred_all, labels_all)
    print("Epoch {} | Time(s) {:.4f} | Loss {:.4f} | Train Accuracy {:.4f}"
         .format(epoch, dur, np.mean(loss_all), acc))

    pred_all = []
    labels_all = []
    for features, labels in val_dataloader:
        labels = labels.to(device)
        features = list(map(lambda batch: batch_to_device(batch, device), features))

        logits = model(features)
        pred = torch.argmax(logits)
        pred_all.append(pred.detach().cpu().numpy())
        labels_all.append(labels[0].detach().cpu().numpy())

    do_check_and_update_metrics(f1_score(pred_all, labels_all, average='macro'),
        accuracy_score(pred_all, labels_all), epoch, loss_all, model, optimizer)

model.eval()

pred_all = []
labels_all = []
for features, labels in test_dataloader:
    labels = labels.to(device)
    features = list(map(lambda batch: batch_to_device(batch, device), features))

    logits = model(features)
    pred = torch.argmax(logits)
    pred_all.append(pred.detach().cpu().numpy())
    labels_all.append(labels[0].detach().cpu().numpy())


print("Test Precision:", precision_score(pred_all, labels_all, average='macro'))
print("Test Recall:", recall_score(pred_all, labels_all, average='macro'))
print("Test F1-score:", f1_score(pred_all, labels_all, average='macro'))
print("Test Accuracy:", accuracy_score(pred_all, labels_all))
print(classification_report(labels_all, pred_all))
