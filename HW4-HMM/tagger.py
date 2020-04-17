import numpy as np

from util import accuracy
from hmm import HMM

# TODO:
def model_training(train_data, tags):
    """
    Train HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - tags: (1*num_tags) a list of POS tags

    Returns:
    - model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
    """
    model = None
    ###################################################
    # Edit here

    S = len(tags)
    pi = np.zeros(S)
    obs_dict = {}
    state_dict = {}
    all_tags = []
    words = []

    for i in train_data:
        all_tags.extend(i.tags)
        words.extend(i.words)
    words = np.unique(words)
    length = len(words)
    A = np.zeros([S, S])
    B = np.zeros([S, length])

    for tag in range(S):
        pi[tag] = (all_tags.count(tags[tag])/len(all_tags))

    for word, index in zip(words, range(len(words))):
        obs_dict[word] = index

    for tag, index in zip(tags, range(len(tags))):
        state_dict[tag] = index

    for sentence in train_data:
        temp_tags = sentence.tags
        temp_words = sentence.words
        tag_list = []
        word_list = []
        for (word, tag) in zip(temp_words, temp_tags):
            tag_list.append(state_dict[tag])
            word_list.append(obs_dict[word])
        for (word, tag) in zip(word_list, tag_list):
            B[tag][word] += 1
        # for index in range(len(temp_tags)):
        #     temp_tags[index] = state_dict[temp_tags[index]]
        #     temp_words[index] = obs_dict[temp_words[index]]
        for i, j in zip(tag_list, tag_list[1:]):
            A[i][j] += 1
        # for index in range(len(temp_words)):
        #     B[temp_tags[index]][temp_words[index]] += 1

    A = (A.T / A.sum(axis=1)).T
    B = (B.T / B.sum(axis=1)).T

    model = HMM(pi, A, B, obs_dict, state_dict)

    ###################################################
    return model

# TODO:
def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - model: an object of HMM class

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    ###################################################
    # Edit here

    S = len(model.pi)
    for sentence in test_data:
        temp_words = sentence.words
        for word in temp_words:
            if word not in model.obs_dict.keys():
                model.obs_dict[word] = len(model.obs_dict)
                new_pro = np.full((1, S), pow(10, -6))
                model.B = np.column_stack((model.B, new_pro.T))
        tagging.append(model.viterbi(temp_words))
    ###################################################
    return tagging
