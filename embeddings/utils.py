from __future__ import print_function

import umap as umap
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix, accuracy_score

from matplotlib import pyplot

import seaborn as sns

from sklearn import preprocessing, model_selection


def split_data(node_subjects, train_size, val_size):
    train_subjects, test_subjects = model_selection.train_test_split(
        node_subjects, train_size=train_size, test_size=None, stratify=node_subjects
    )
    val_subjects, test_subjects = model_selection.train_test_split(
        test_subjects, train_size=val_size, test_size=None, stratify=test_subjects
    )
    return train_subjects, val_subjects, test_subjects


# Converting to numeric arrays
def convert_to_numeric_arrays(train_subjects, val_subjects, test_subjects):
    target_encoding = preprocessing.LabelBinarizer()
    train_targets = target_encoding.fit_transform(train_subjects)
    val_targets = target_encoding.transform(val_subjects)
    test_targets = target_encoding.transform(test_subjects)
    return train_targets, val_targets, test_targets


def getxy(g, data, node_subjects, train_size, val_size):
    train_subjects, val_subjects, test_subjects = split_data(node_subjects, train_size, val_size)
    y_train, y_val, y_test = convert_to_numeric_arrays(train_subjects, val_subjects, test_subjects)


    train_indices = g.nodes_to_loc(train_subjects.index)
    val_indices = g.nodes_to_loc(val_subjects.index)
    test_indices = g.nodes_to_loc(test_subjects.index)

    x_train = np.take(data, train_indices, axis=0).squeeze(0)
    x_val = np.take(data, val_indices, axis=0).squeeze(0)
    x_test = np.take(data, test_indices, axis=0).squeeze(0)

    return x_train, x_val, x_test, y_train, y_val, y_test


def closest_power2(n):
    nex_power_2 = 2 ** (n - 1).bit_length()
    ex_power_2 = 2 ** ((n - 1).bit_length() - 1)
    if n-ex_power_2 < nex_power_2-n:
        return ex_power_2
    return nex_power_2


def evaluate_preds(true, pred):
    auc = roc_auc_score(true, pred)
    pr = average_precision_score(true, pred)
    bin_pred = [1 if p > 0.5 else 0 for p in pred]
    f_score = f1_score(true, bin_pred)
    print('ROC AUC:', auc)
    print('PR AUC:', pr)
    print('F1 score:', f_score)
    print(confusion_matrix(true, bin_pred, normalize='true'))
    return auc, pr, f_score


def evaluation(y_true, y_pred):
    #bin_pred = [1 if p > 0.5 else 0 for p in y_pred]
    b = np.zeros_like(y_pred)
    b[np.arange(y_pred.shape[0]), np.argmax(y_pred, axis=1)] = 1

    macroF1 = f1_score(y_true, b, average='macro')
    accuracy = accuracy_score(y_true, b)

    print("macroF1: " + str(macroF1))
    print("accuracy: " + str(accuracy))




def plot_gcn_embeddings(embeddings, labels):
    u = umap.UMAP(random_state=42)
    umap_embs = u.fit_transform(embeddings[0])
    pyplot.figure(figsize=(10, 5))
    ax = sns.scatterplot(x=umap_embs[:, 0], y=umap_embs[:, 1], hue=labels)
    pyplot.show()