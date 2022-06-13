# code based on https://github.com/joshualoehr/ngram-language-model

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

SOS = "<s> "
EOS = "</s>"
UNK = "<UNK>"

def remove_punc(str_lis):
    result = []
    for item in str_lis:
        item = re.sub('\s', ' ', item)
        item = re.sub('[^a-zA-Z0-9 ]', '', item)
        result.append(item)
    return result

def add_sentence_tokens(sentences, n):
    """Wrap each sentence in SOS and EOS tokens.

    For n >= 2, n-1 SOS tokens are added, otherwise only one is added.

    Args:
        sentences (list of str): the sentences to wrap.
        n (int): order of the n-gram model which will use these sentences.
    Returns:
        List of sentences with SOS and EOS tokens wrapped around them.

    """
    sos = SOS * (n-1) if n > 1 else SOS
    return ['{}{} {}'.format(sos, s, EOS) for s in sentences]

def replace_singletons(tokens, p):
    """Replace tokens which appear only once in the corpus with <UNK>.
    
    Args:
        tokens (list of str): the tokens comprising the corpus.
    Returns:
        The same list of tokens with each singleton replaced by <UNK>.
    
    """
    vocab = nltk.FreqDist(tokens)
    result = [UNK if vocab[token] == 1 and random.random() < p else token for token in tokens]
    # print('UNK rate:', sum([1 if token==UNK else 0 for token in result])/len(result))
    return result

def preprocess(sentences, n, remove_p):
    """Add SOS/EOS/UNK tokens to given sentences and tokenize.

    Args:
        sentences (list of str): the sentences to preprocess.
        n (int): order of the n-gram model which will use these sentences.
    Returns:
        The preprocessed sentences, tokenized by words.

    """
    sentences = add_sentence_tokens(sentences, n)
    tokens = ' '.join(sentences).split(' ')
    if remove_p > 0:
        tokens = replace_singletons(tokens, remove_p)
    return tokens


class LanguageModel(object):
    """An n-gram language model trained on a given corpus.
    
    For a given n and given training corpus, constructs an n-gram language
    model for the corpus by:
    1. preprocessing the corpus (adding SOS/EOS/UNK tokens)
    2. calculating (smoothed) probabilities for each n-gram

    Also contains methods for calculating the perplexity of the model
    against another corpus, and for generating sentences.

    Args:
        train_data (list of str): list of sentences comprising the training corpus.
        n (int): the order of language model to build (i.e. 1 for unigram, 2 for bigram, etc.).
        laplace (int): lambda multiplier to use for laplace smoothing (default 1 for add-1 smoothing).

    """

    def __init__(self, train_data_1, train_data_2, n, laplace=1, remove_p=0.0):
        self.n = n
        self.laplace = laplace
        self.tokens_1 = preprocess(train_data_1, n, remove_p)
        self.tokens_2 = preprocess(train_data_2, n, remove_p)
        self.vocab  = nltk.FreqDist(self.tokens_1 + self.tokens_2)
        self.model_1, self.model_2  = self._create_model()
        self.masks  = list(reversed(list(product((0,1), repeat=n))))

    def _smooth(self):
        """Apply Laplace smoothing to n-gram frequency distribution.
        
        Here, n_grams refers to the n-grams of the tokens in the training corpus,
        while m_grams refers to the first (n-1) tokens of each n-gram.

        Returns:
            dict: Mapping of each n-gram (tuple of str) to its Laplace-smoothed 
            probability (float).

        """

        n_grams = nltk.ngrams(self.tokens_1, self.n)
        n_vocab = nltk.FreqDist(n_grams)

        m_grams = nltk.ngrams(self.tokens_1, self.n-1)
        m_vocab = nltk.FreqDist(m_grams)

        model_1 = {}
        for n_gram, n_count in n_vocab.items():
            m_gram = n_gram[:-1]
            m_count = m_vocab[m_gram]
            model_1[n_gram] = (self.laplace + n_count ) / (m_count + self.laplace * len(self.vocab))
        for m_gram, m_count in m_vocab.items():
            out_of_vocab_n_gram = tuple(list(m_gram) + [UNK])
            model_1[out_of_vocab_n_gram] = self.laplace / (m_count + self.laplace * len(self.vocab))
        out_of_m_gram = tuple([UNK]*self.n)
        model_1[out_of_m_gram] = 1 / len(self.vocab)


        # for m_gram, m_count in m_vocab.items():
        #     for k in self.vocab.keys():
        #         new_n_gram = tuple(list(m_gram) + [k])
        #         model_1[new_n_gram] = (self.laplace + n_vocab.get(new_n_gram,0) ) / (m_count + self.laplace * len(self.vocab))
        #         print(len(model_1))

        n_grams = nltk.ngrams(self.tokens_2, self.n)
        n_vocab = nltk.FreqDist(n_grams)

        m_grams = nltk.ngrams(self.tokens_2, self.n-1)
        m_vocab = nltk.FreqDist(m_grams)

        model_2 = {}
        for n_gram, n_count in n_vocab.items():
            m_gram = n_gram[:-1]
            m_count = m_vocab[m_gram]
            model_2[n_gram] = (self.laplace + n_count ) / (m_count + self.laplace * len(self.vocab))
        for m_gram, m_count in m_vocab.items():
            out_of_vocab_n_gram = tuple(list(m_gram) + [UNK])
            model_2[out_of_vocab_n_gram] = self.laplace / (m_count + self.laplace * len(self.vocab))
        out_of_m_gram = tuple([UNK]*self.n)
        model_2[out_of_m_gram] = 1 / len(self.vocab)

        # for m_gram, m_count in m_vocab.items():
        #     for k in self.vocab.keys():
        #         new_n_gram = tuple(list(m_gram) + [k])
        #         model_2[new_n_gram] = (self.laplace + n_vocab.get(new_n_gram,0) ) / (m_count + self.laplace * len(self.vocab))

        return model_1, model_2

        # result = { n_gram: smoothed_count(n_gram, count) for n_gram, count in n_vocab.items() }
        # result[(UNK,)*self.n]= 1/(self.laplace * vocab_size)
        # return result

    def _create_model(self):
        """Create a probability distribution for the vocabulary of the training corpus.
        
        If building a unigram model, the probabilities are simple relative frequencies
        of each token with the entire corpus.

        Otherwise, the probabilities are Laplace-smoothed relative frequencies.

        Returns:
            A dict mapping each n-gram (tuple of str) to its probability (float).

        """
        if self.n == 1:
            vocab_1 = nltk.FreqDist(self.tokens_1)
            model_1 = {}
            for unigram, uni_count in vocab_1.items():
                model_1[(unigram,)] = (self.laplace + uni_count) / (len(self.tokens_1) + self.laplace * len(self.vocab) )
            # if (UNK,) not in model_1:   # remove this condition
            model_1[(UNK,)] = self.laplace / (len(self.tokens_1) + self.laplace * len(self.vocab) )

            # for unigram in self.vocab.keys():
            #     model_1[(unigram,)] = (self.laplace + vocab_1.get(unigram, 0)) / (len(self.tokens_1) + self.laplace * len(self.vocab) )

            vocab_2 = nltk.FreqDist(self.tokens_2)
            model_2 = {}
            for unigram, uni_count in vocab_2.items():
                model_2[(unigram,)] = (self.laplace + uni_count) / (len(self.tokens_2) + self.laplace * len(self.vocab) )
            # if (UNK,) not in model_2:
            model_2[(UNK,)] = self.laplace / (len(self.tokens_2) + self.laplace * len(self.vocab) )


            # for unigram in self.vocab.keys():
            #     model_2[(unigram,)] = (self.laplace + vocab_2.get(unigram, 0)) / (len(self.tokens_2) + self.laplace * len(self.vocab) )
            
            return model_1, model_2
        else:
            return self._smooth()

    def _convert_oov(self, ngram, model):
        """Convert, if necessary, a given n-gram to one which is known by the model.

        Starting with the unmodified ngram, check each possible permutation of the n-gram
        with each index of the n-gram containing either the original token or <UNK>. Stop
        when the model contains an entry for that permutation.

        This is achieved by creating a 'bitmask' for the n-gram tuple, and swapping out
        each flagged token for <UNK>. Thus, in the worst case, this function checks 2^n
        possible n-grams before returning.

        Returns:
            The n-gram with <UNK> tokens in certain positions such that the model
            contains an entry for it.

        """
        mask = lambda ngram, bitmask: tuple((token if flag == 1 else "<UNK>" for token,flag in zip(ngram, bitmask)))

        ngram = (ngram,) if type(ngram) is str else ngram
        for possible_known in [mask(ngram, bitmask) for bitmask in self.masks]:
            if possible_known in model:
                return possible_known

    def perplexity(self, test_data):
        """Calculate the perplexity of the model against a given test corpus.
        
        Args:
            test_data (list of str): sentences comprising the training corpus.
        Returns:
            The perplexity of the model as a float.
        
        """
        test_tokens = preprocess(test_data, self.n, remove_p=0.0)
        test_ngrams = list(nltk.ngrams(test_tokens, self.n))
        N = len(test_tokens)

        known_ngrams  = [self._convert_oov(ngram, self.model_1) for ngram in test_ngrams]
        probabilities = [self.model_1[ngram] for ngram in known_ngrams]
        # print(probabilities)
        pp_1 = math.exp((-1/N) * sum(map(math.log, probabilities)))

        known_ngrams  = [self._convert_oov(ngram, self.model_2) for ngram in test_ngrams]
        probabilities = [self.model_2[ngram] for ngram in known_ngrams]
        # print(probabilities)
        pp_2 = math.exp((-1/N) * sum(map(math.log, probabilities)))
        return pp_1, pp_2



if __name__ == '__main__':
    
    train_users = 160
    valid_users = 50
    splitted_data_path = './splitted_data'

    if not os.path.exists(splitted_data_path):

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
        assert train_users + valid_users == len(I)

        
        random.shuffle(I)
        random.shuffle(NI)
        train_labels = {}
        valid_labels = {}
        for i in range(train_users):
            train_labels[I[i]] = "I"
            train_labels[NI[i]] = "NI"

        for i in range(train_users, train_users+valid_users):
            valid_labels[I[i]] = "I"
            valid_labels[NI[i]] = "NI"



        Ironic_train_data = []
        Non_ironic_train_data = []
        valid_data = {}
        

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
            if USERCODE in train_labels.keys():
                if train_labels[USERCODE] == "I":
                    Ironic_train_data.extend(texts)
                else:
                    Non_ironic_train_data.extend(texts)
            else:
                assert USERCODE in valid_labels.keys()
                valid_data[USERCODE] = texts


        print(len(Ironic_train_data))
        print(len(Non_ironic_train_data))
        print(len(valid_data))

        splitted_data = [Ironic_train_data, Non_ironic_train_data, valid_data, valid_labels]
        pickle.dump(splitted_data, open(splitted_data_path, 'wb'), -1)

    else:
        splitted_data = pickle.load(open(splitted_data_path, 'rb'))
        Ironic_train_data, Non_ironic_train_data, valid_data, valid_labels = splitted_data
        print(len(Ironic_train_data))
        print(len(Non_ironic_train_data))
        print(len(valid_data))


    Ironic_train_data = remove_punc(Ironic_train_data)
    Non_ironic_train_data = remove_punc(Non_ironic_train_data)
    for key in valid_data.keys():
        valid_data[key] = remove_punc(valid_data[key])

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int)
    parser.add_argument("--lap", type=int)
    parser.add_argument("--remove_p", type=float)

    args = parser.parse_args()

    n_gram = args.n
    laplace = args.lap
    remove_p = args.remove_p
    print("n: ", n_gram, "\t laplace: ", laplace, "\t remove singleton probability: ", remove_p)
    print("Loading {}-gram model...".format(n_gram))
    lm = LanguageModel(Ironic_train_data, Non_ironic_train_data, n_gram, laplace=laplace, remove_p=remove_p)
    print("Vocabulary size: {}".format(len(lm.vocab)))

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

    print('accuracy: ', total_score / len(valid_data))
    print('I in mistake ratio: ', mistake_I / mistake_num)


