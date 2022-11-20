import numpy as np
import pandas as pd
import csv
import os
''' Function of this code 
算BD 和 HC 在量表資料夾中 各種feature中是否存在
並儲存在 ./attender_of_feature/
讓feature attender 可以確定 要拿哪些的 attender(量表資料)
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

def count_attender_of_feature(dir_train,feature_type):
    res = []
    res.append(['attender'])
    feature_fn = datatype_fn[feature_type]
    check_shape = 0

    for f in os.listdir(dir_train):
        # exclude .csv as attender
        if not os.path.isdir(dir_train+f):
            pass
        else:
            print('Checking',f)
            tmp = [f]
            # check
            if (not os.path.isdir(dir_train+f+'/feature/') ) or (not os.path.isfile(dir_train+f+'/feature/'+feature_fn) ):
                print('No such feature file')
                continue
            else:
                feature = np.loadtxt(dir_train+f+'/feature/'+feature_fn , delimiter=',')
                print('feature shape :',feature.shape)
                #check if all feature shape are same
                if check_shape == 0:
                    check_shape = len(feature)
                if check_shape != len(feature):
                    print('Oops! shape is not match!')
                #checking whether nan
                print('checking',feature_type)
                if check_if_miss(feature):
                    pass
                else:
                    res.append(tmp)
    print('res:',res)
    dir_grp = ''
    if dir_train == './train/':
        dir_grp = 'HC'
    elif dir_train == './train_p/':
        dir_grp = 'BD'
    print('feature type',feature_type)
    with open('./attender_of_feature/' + feature_type + '_' + dir_grp + '_count_res.csv', "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(res)

if __name__ == '__main__':
    dir_attend = 'attender_in_type/'
    all_type = ['B_1','B_2','C','D','E']

    # HC
    dir_ = './train/'
    for feature in all_type:
        count_attender_of_feature(dir_,feature)
    # BD
    dir_ = './train_p/'
    for feature in all_type:
        count_attender_of_feature(dir_,feature)
