import numpy as np
import pandas as pd
import csv
import os
from feature_attender  import get_wanna_attender

''' Function of this code 
算每一種feature的最大最小值
並儲存在 ./feature_max_min/ 
以便load_data 將input做 zero-one normalized
'''
datatype_fn={
    #'A': 'gps_new_all_grp_feature.csv',
    'B_1': 'dass.csv',
    'B_2' : 'altman.csv',
    'C': 'Demotion_w.csv',
    'D': 'sleep_w.csv',
    'E': 'seven_emo_w.csv'
}
def check_if_miss(x):
    mask = np.isnan(x)
    if True in mask:
        return True
    else:
        return False

if __name__ == '__main__':
    feature_types = ['B_1','B_2','C','D','E']
    dir_save = './feature_max_min/'
    # load train(HC) and train_p(BD) feature then get max and min of feature
    for feature_type in feature_types:
        # HC
        print('type:',feature_type)
        dir_grp = 'HC'
        HC_attenders = get_wanna_attender(dir_grp,[feature_type])
        print('HC attenders',HC_attenders)
        print(len(HC_attenders))
        type_feature = []
        for at in HC_attenders:
            feature = np.loadtxt('./train/' + at + '/feature/' + datatype_fn[feature_type], delimiter=',')
            type_feature.append(feature)
        # BD
        dir_grp = 'BD'
        BD_attenders = get_wanna_attender(dir_grp,[feature_type])
        print('BD attenders',BD_attenders)
        print(len(BD_attenders))

        for at in BD_attenders:
            feature = np.loadtxt('./train_p/' + at + '/feature/' + datatype_fn[feature_type], delimiter=',')
        type_feature = np.array(type_feature)
        print('type_feature size',type_feature.shape)
        max_val_in_type = np.amax(type_feature, axis=0)
        np.savetxt(dir_save + feature_type + '_max.csv', max_val_in_type, delimiter=",")
        min_val_in_type = np.min(type_feature, axis=0)
        np.savetxt(dir_save + feature_type + '_min.csv', min_val_in_type, delimiter=",")