import numpy as np
import pandas as pd
import csv
import os
from itertools import combinations
''' Function of this code 
根據  dir_grp(BD/HC) feature_types 回傳 量表資料夾
'''
selected_attender = []

def interscetion_type_of_grp(at_list):
    print('-----------------------------------------------')
    global selected_attender
    if not selected_attender:
        selected_attender = at_list
    else:
        selected_attender = list(set(selected_attender) & set(at_list))
    print('===============================================')

def count_attender(dir_grp,feature_type):
    print('Counting the type',feature_type)
    df = pd.read_csv('./attender_of_feature/'+feature_type+'_' + dir_grp +'_count_res.csv')
    tmp_attender = df['attender']
    print('attender size:',df.shape)
    tmp_attender = tmp_attender.values.tolist()
    interscetion_type_of_grp(tmp_attender)

def get_wanna_attender(dir_grp,feature_types):
    global selected_attender
    for feature_type in feature_types:
        count_attender(dir_grp,feature_type)
    return selected_attender

if __name__ == '__main__':
    # HC
    dir_grp = 'HC'
    feature_types = ['B_1','B_2','C','D','E']
    HC_res = get_wanna_attender(dir_grp,feature_types)
    print('HC res:',len(HC_res))
    # BD
    dir_grp = 'BD'
    BD_res = get_wanna_attender(dir_grp,feature_types)
    print('BD res:',len(BD_res))
