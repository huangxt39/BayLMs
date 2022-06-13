import argparse
from itertools import product
import math
import nltk
from pathlib import Path

import glob
from xml.etree import ElementTree as ET
import os
import random
import pickle
import re
import numpy as np

from ngram_lm import remove_punc, LanguageModel

SOS = "<s> "
EOS = "</s>"
UNK = "<UNK>"



I_user_per_fold = int(210 / 5)
data_folds_path = './data_folds'

if not os.path.exists(data_folds_path):
    print('making cross validation folds')
    GT = "./pan22-author-profiling-training-2022-03-29/en/truth.txt"
    true_values = {}
    f=open(GT, encoding='utf-8')
    for line in f:
        linev = line.strip().split(":::")
        true_values[linev[0]] = linev[1]
    f.close()


    I=[]
    NI=[]
    for key in true_values.keys():
        if true_values[key] == "I":
            I.append(key)
        else:
            NI.append(key)
    print(len(I), len(NI))
    assert len(I) == len(NI)

    
    random.shuffle(I)
    random.shuffle(NI)

    all_user_data = {}
    for FILE in glob.glob("./pan22-author-profiling-training-2022-03-29/en/*.xml"):
        #The split command below gets just the file name,
        #without the whole address. The last slicing part [:-4]
        #removes .xml from the name, so that to get the user code

        parsedtree = ET.parse(FILE)
        documents = parsedtree.iter("document")

        texts = []
        for doc in documents:
            texts.append(doc.text)

        USERCODE = FILE.split("/")[-1][:-4]
        all_user_data[USERCODE] = texts


    user_folds = []
    for k in range(5):
        s = k*I_user_per_fold
        e = s + I_user_per_fold
        user_labels = {}
        for i in range(s, e):
            user_labels[I[i]] = "I"
            user_labels[NI[i]] = "NI"
        user_folds.append(user_labels)
        

    data_folds = [user_folds, all_user_data]
    pickle.dump(data_folds, open(data_folds_path, 'wb'), -1)
    print('cross validation folds dumped')

else:
    data_folds = pickle.load(open(data_folds_path, 'rb'))
    user_folds, all_user_data = data_folds

def merge_folds(train_users):
    result = {}
    for user_labels in train_users:
        result.update(user_labels)
    return result

# parameter searching
args_combines = []
for n in [1, 2, 3, 4]:
    for lap in [0.01, 0.1, 0.3, 1.0, 3.0, 10.0]: # [1, 2, 3, 4, 5]
        for remove_p in [0.0, ]: #0.25, 0.5, 1
            args_combines.append((n, lap, remove_p))

for args in args_combines:

    n_gram, laplace, remove_p = args
    print("\n n: ", n_gram, "\t laplace: ", laplace, "\t remove singleton probability: ", remove_p)

    acc_list = []
    ratio_list = []
    for k in range(5):

        train_users = []
        for i, user_labels in enumerate(user_folds):
            if i != k :
                train_users.append(user_labels)
            else:
                valid_labels = user_labels

        train_users = merge_folds(train_users)

        Ironic_train_data = []
        Non_ironic_train_data = []
        for USERCODE in train_users.keys():
            if train_users[USERCODE] == "I":
                Ironic_train_data.extend(all_user_data[USERCODE])
            else:
                Non_ironic_train_data.extend(all_user_data[USERCODE])
        # print(len(Ironic_train_data))
        # print(len(Non_ironic_train_data))

        valid_data = {}
        for USERCODE in valid_labels.keys():
            valid_data[USERCODE] = all_user_data[USERCODE]
        # print(len(valid_data))

        
        Ironic_train_data = remove_punc(Ironic_train_data)
        Non_ironic_train_data = remove_punc(Non_ironic_train_data)
        for key in valid_data.keys():
            valid_data[key] = remove_punc(valid_data[key])

        # print("Loading {}-gram model...".format(n_gram))
        lm = LanguageModel(Ironic_train_data, Non_ironic_train_data, n_gram, laplace=laplace, remove_p=remove_p)
        # print("Vocabulary size: {}".format(len(lm.vocab)))

        total_score = 0
        mistake_num = 0
        mistake_I = 0
        for user in valid_data.keys():
            texts = valid_data[user]
            user_label = valid_labels[user]
            
            ironic_pp, non_ironic_pp = lm.perplexity(texts)
            

            if ironic_pp < non_ironic_pp:
                prediction = "I"
            else:
                prediction = "NI"

            if prediction == user_label:
                total_score += 1
            else:
                mistake_num += 1
                if prediction == "I":
                    mistake_I += 1

        # print('accuracy: ', total_score / len(valid_data))
        acc_list.append( total_score / len(valid_data) )
        # print('I in mistake ratio: ', mistake_I / mistake_num)
        ratio_list.append(mistake_I / mistake_num)

    acc_list = np.array(acc_list)
    ratio_list = np.array(ratio_list)
    print( "accuracy mean: ", acc_list.mean(), "\t accuracy std: ", acc_list.std(), "\t ratio mean: ", ratio_list.mean(), )