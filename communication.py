from typing import Tuple
from typing import Tuple, List
import numpy as np
from numpy.typing import ArrayLike
from itertools import product
from string import ascii_uppercase


def read_prob_matrix(fpath:str=None) -> ArrayLike:
    with open(fpath, 'rb') as f:
        prob_matrix = np.load(f)
    return prob_matrix

def sample_index(prob_matrix:ArrayLike=None) -> Tuple:
    word_idx = np.random.choice(prob_matrix.shape[0], replace=True, \
                p=prob_matrix.sum(axis=1))
    chip_idx = np.random.choice(prob_matrix.shape[1], replace=True, \
                p=prob_matrix.sum(axis=0))
    return word_idx, chip_idx

def sample_language(fpath:str, N:int=1) -> Tuple[List, List]:
    idx2colour = {idx:idx for idx in range(330)}
    words = [''.join(i) for i in product(ascii_uppercase, repeat=2)]
    idx2word = {idx:word for idx, word in enumerate(words)}
    
    prob_matrix = read_prob_matrix(fpath)
    
    word_list = []
    colour_list = []
    for _ in range(N):
        word_idx, chip_idx = sample_index(prob_matrix)
        word = idx2word[word_idx]
        colour = idx2colour[chip_idx]
        
        word_list.append(word)
        colour_list.append(colour)
        
    return word_list, colour_list

def output_language_file(words:list, 
                        colours:list, 
                        opath:str='./lan.txt',
                        lan_idx:int=1,
                        spk_idx:int=1) -> None:
    with open(opath, 'a') as f:
        for word, colour in zip(words, colours):
            print(str(lan_idx)+'\t'+str(spk_idx)+'\t'+str(colour)+'\t'+word, \
                    file=f)

if __name__ == '__main__':
    sample_words, sample_colours = sample_language('./tmp_test.npy', N=10)
    output_language_file(sample_words, sample_colours)
    sample_words, sample_colours = sample_language('./tmp_test.npy', N=20)
    output_language_file(sample_words, sample_colours, lan_idx=2, spk_idx=2)