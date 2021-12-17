from typing import Tuple
from typing import Tuple, List
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from itertools import product
from string import ascii_uppercase

class LanguageSampler(object):
    def __init__(self, fpath:str=None) -> None:
        super().__init__()
        self.fpath = fpath
        self.prob_matrix = None
        self.read_prob_matrix()

    def read_prob_matrix(self) -> ArrayLike:
        with open(self.fpath, 'rb') as f:
            self.prob_matrix = np.load(f)

    def sample_index(self) -> Tuple:
        word_idx = np.random.choice(self.prob_matrix.shape[0], replace=True, \
                    p=self.prob_matrix.sum(axis=1))
        chip_idx = np.random.choice(self.prob_matrix.shape[1], replace=True, \
                    p=self.prob_matrix.sum(axis=0))
        return word_idx, chip_idx

    def sample_language(self, N:int=1) -> Tuple[List, List]:
        idx2colour = {idx:idx for idx in range(330)}
        words = [''.join(i) for i in product(ascii_uppercase, repeat=2)]
        idx2word = {idx:word for idx, word in enumerate(words)}

        word_list = []
        colour_list = []
        for _ in range(N):
            word_idx, chip_idx = self.sample_index()
            word = idx2word[word_idx]
            colour = idx2colour[chip_idx]

            word_list.append(word)
            colour_list.append(colour)

        return word_list, colour_list

    @staticmethod
    def output_language_file(words:list, 
                            colours:list, 
                            opath:str='./lan.txt',
                            lan_idx:int=1,
                            spk_idx:int=1) -> None:
        with open(opath, 'a') as f:
            for word, colour in zip(words, colours):
                print(str(lan_idx)+'\t'+str(spk_idx)+'\t'+str(colour)+\
                    '\t'+word, file=f)
                
class MutualInfoCalculator(object):
    def __init__(self) -> None:
        super().__init__()
    
    @staticmethod
    def read_language_data(fpath:str="wcs/term.txt") -> pd.DataFrame:
        df = pd.read_csv(fpath, delimiter="\t", header=None)
        df.columns = ["language", "speaker", "chip", "word"]
        return df
    
    @staticmethod
    def read_colour_data(fpath:str="wcs/cnum-vhcm-lab-new.txt") -> pd.DataFrame:
        df = pd.read_csv(fpath, sep="\t", header=0, index_col="#cnum")
        df = df.sort_values(by="#cnum")[["L*", "a*", "b*"]].copy().reset_index()
        return df
    
    @staticmethod
    def get_px(fpath:str="wcs/term.txt") -> ArrayLike:
        raise NotImplementedError
    
    @staticmethod
    def get_pxGy(fpath:str="wcs/cnum-vhcm-lab-new.txt") -> ArrayLike:
        raise NotImplementedError
    
    def get_MI(self, flan:str=None, fclab:str=None) -> float:
        p_x = self.get_px(flan)
        p_xGy = self.get_pxGy(fclab)
        
        p_xy = p_xGy * p_x[:, np.newaxis]
        p_xy = p_xy / np.sum(p_xy)
        
        mi = np.nansum(p_xy * np.log2(p_xy / (p_xy.sum(axis=0, keepdims=True) *\
                        p_xy.sum(axis=1, keepdims=True))))
        
        return mi

if __name__ == '__main__':
    sampler = LanguageSampler('./tmp_test.npy')
    sample_words, sample_colours = sampler.sample_language(N=10)
    sampler.output_language_file(sample_words, sample_colours)
    sample_words, sample_colours = sampler.sample_language(N=20)
    sampler.output_language_file(sample_words, sample_colours, lan_idx=2, \
                                spk_idx=2)