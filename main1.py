from typing import List, Tuple
from os import listdir
from os.path import isfile, join
import glob, os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from ibhelpers import score_q_kl
from communication import MutualInfoCalculator


def read_lan_file_from_dir(dir_path:str) -> pd.DataFrame:
    paths = []
    random_seeds = []
    lan_ids = []
    num_words = []
    
    for fpath in glob.glob(dir_path + "**/[1-9]*.npy", recursive=True):
        paths.append(fpath)
        random_seeds.append(fpath.split('.')[-2].split('/')[-3])
        lan_ids.append(fpath.split('.')[-2].split('/')[-2])
        num_words.append(fpath.split('.')[-2].split('/')[-1])
        
    os.chdir('./')
    
    return pd.DataFrame(list(zip(paths, random_seeds, lan_ids, num_words)),
                        columns =['path', 'random seed', 'language id',
                                  'number of words']
                       )


def get_pxy(flan: str="./wcs/term.txt", 
            fclab:str="./wcs/cnum-vhcm-lab-new.txt"
           ) -> ArrayLike:
    mi = MutualInfoCalculator()
    p_x = mi.get_px(flan)
    p_xGy = mi.get_pxGy(fclab)

    p_xy = p_xGy * p_x[:, np.newaxis]
    p_xy = p_xy / np.sum(p_xy)
    
    return p_xy


def process_lan(lan_file:str, p_xy:ArrayLike) -> Tuple[float, float]: 
    q = np.load(lan_file)
    print(lan_file)
    num_rows = q.shape[0]
    if num_rows <330:
        num_duplicate = 330 - num_rows
        duplicate_vector = np.ones(num_rows, dtype=int)
        duplicate_vector[-1] = 331 - num_rows
        q = np.repeat(q, duplicate_vector, axis=0)
        
    if not q.shape[0] == q.shape[1]:
        print("Warning!")
        print(q.shape)
        print(p_xy.shape)
    
    return score_q_kl(p_xy, q)


def process_languages(lan_df:pd.DataFrame, output_file:str) -> None:
    p_xy = get_pxy()
    
    rate_list = []
    distortion_list = []
    for index, lan_item in lan_df.iterrows():
        result = process_lan(lan_item['path'], p_xy)
        rate_list.append(result[0])
        distortion_list.append(result[1])
    
    lan_df.insert(-1, "rate", rate_list, True)
    lan_df.insert(-1, "distortion", distortion_list, True)
    
    with open(output_file, 'w') as f:
        lan_df.to_csv(f)


def plot_curves(csv_path:str, out_file:str) -> None:
    # TODO: plot frontier first
    
    df = pd.read_csv(csv_path).sort_values(by=['number of words'])
    
    for lan_id in df['language id'].unique():
        lan_df = df[df['language id'] == lan_id]
        # select a colour for the language
        colour = list(matplotlib.colors.cnames.keys())[lan_id]
        lan_df.plot(x="rate", y="distortion")
    
    plt.savefig(out_file, format='pdf', bbox_inches='tight')


def plot_boxes(csv_path:str, out_file:str) -> None:
    df = pd.read_csv(csv_path).sort_values(by=['number of words'])
    boxplot = df.boxplot(column=['rate'], by=['number of words'])
    plt.savefig(out_file, format='pdf', bbox_inches='tight')


if __name__ == '__main__':
    LAN_DIR = './output/learnability/' 
    lan_df = read_lan_file_from_dir(LAN_DIR)
    OUT_PATH = './output/learnability/result.csv'
    process_languages(lan_df, OUT_PATH)
    plot_boxes(OUT_PATH)
    