from typing import Union, List
import numpy as np
import torch,json
import torch.nn as nn

def get_numpy_word_embed(word2index: Union[dict,None],pretrained_wordemb_path: str) -> np.array:
    """
    read pretrained word embedding from file and conver it to np.array

    :param word2index: dict {word:index...}
    :param pretrained_wordemb_path:  txt file （word,num1,num2,...）
    :return: pretrained word embedding

    Example
    -------
    >>> word_embedding = get_numpy_word_embed(word2index,pretrained_wordemb_path)
    >>> embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word_embedding))
    """
    if word2index is None:
        return np.load(pretrained_wordemb_path)

    words_embed = {}
    with open(pretrained_wordemb_path, "r") as pretrained_wordemb_file:
        lines = pretrained_wordemb_file.readlines()
        for line in lines:
            line_list = line.split()
            word = line_list[0]
            embed = line_list[1:]
            embed = [float(num) for num in embed]
            words_embed[word] = embed
    dim = len(embed)
    index2word = {index: word for word,index in word2index.items()}
    id2emb = {}
    for index in range(len(word2index)):
        if index2word[index] in words_embed:
            id2emb[index] = words_embed[index2word[index]]
        else:
            id2emb[index] = [0.0] * dim

    word_embedings = np.array([id2emb[index] for index in range(len(word2index))])

    # set unk embedding which the averate embedding of all words
    word_embedings[0,:] = np.mean(word_embedings[1:,:],axis=0)
    np.save("glove.42B.300d.npy",word_embedings)
    return word_embedings


def get_word_count(data_source: Union[str,dict]):
    if isinstance(data_source,str):
        with open(data_source,"r") as f:
            word_count_dict = json.load(f)
        return  word_count_dict

    word_count_dict = dict()
    for sample in data_source:
        for word in sample["question"].lower().split():
            if word in word_count_dict:
                word_count_dict[word] += 1
            else:
                word_count_dict[word] = 1
    return word_count_dict


def logging_format(losses: List[str], show_iter: bool = True) -> str :
    # A simple function to return format of showing loss info
    # example : log_f % tuple([])
    if show_iter:
        basic_format = "[Ep: %.2f][Iter: %d]"
    else:
        basic_format = "[EP: %.2f]"
    loss_f = ""
    for loss in losses:
        loss_f += "[{} :%.3g]".format(loss)
    return basic_format+loss_f

