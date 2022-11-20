
import numpy as np
import warnings
warnings.filterwarnings("ignore")
''' Function of this code 
load data 用 , 其中的 主要是要把目標量表資料分成 factors 和總分
使用的方式如Uni_Lasso_HAMD/YMRS 所述
'''
datatype_fn={
    #'A': 'gps_new_all_grp_feature.csv',
    'B_1': 'dass.csv',
    'B_2' : 'altman.csv',
    'C': 'Demotion_w.csv',
    'D': 'sleep_w.csv',
    'E': 'seven_emo_w.csv'
}

def hamd_partition_to_factor(hamd):
    global  hamd_factor
    new_hamd=[]
    for i in hamd:
        tmp=[]
        for problems in hamd_factor.values():
            class_score=0
            for problem in problems:
                class_score=class_score+i[problem]
            tmp.append(class_score)
        # sum of 21 questions
        tmp.append(np.sum(i[0:21]))
        new_hamd.append(tmp)
    new_hamd = np.array(new_hamd)
    return new_hamd

def ymrs_partition_to_factor(ymrs):
    global  ymrs_factor
    new_ymrs=[]
    for i in ymrs:
        tmp=[]
        for problems in ymrs_factor.values():
            class_score=0
            for problem in problems:
                class_score=class_score+i[problem]
            tmp.append(class_score)
        # sum of 11 questions
        tmp.append(np.sum(i[0:11]))
        new_ymrs.append(tmp)
    new_ymrs = np.array(new_ymrs)
    return new_ymrs

hamd_factor={
    'core':[0,1,6,7,9],
    'c_core':[0,1,2],
    'sleep':[3,4,5],
    'activity':[6,7],
    'psychic_anxiety':[8,9],
    'somatic_anxiety':[10,11,12],
    'delusopn':[1,4,19],
}

ymrs_factor={
    'psychotic_mania':[3,5,6,7,10],
    'irritable_mania':[1,4,8],
    'elated mania':[0,2,9],
}
def load_max_min(feature_types):
    max_dim = []

    min_dim = []

    for feature_type in feature_types:
        max_tmp = np.loadtxt( './feature_max_min/' + feature_type + '_max.csv', delimiter=',')
        for t in max_tmp:
            max_dim.append(t)
        min_tmp = np.loadtxt( './feature_max_min/' + feature_type + '_min.csv', delimiter=',')
        for k in  min_tmp:
            min_dim.append(k)
    max_dim = np.array(max_dim)
    min_dim = np.array(min_dim)
    print('max',max_dim.shape)
    print('min',max_dim.shape)
    return max_dim , min_dim

def Normalized(x,max_dim,min_dim):
    res = []
    for x_ in x:
        tmp_x = []
        for dim in range (len(x_)):
            nor_x = ( x_[dim] - min_dim[dim]) / (max_dim[dim] - min_dim[dim])
            tmp_x.append(nor_x)
        res.append(tmp_x)
    res = np.array(res)
    print('res:',res.shape)
    return res

# loading data
def loading_data(dir_grp,attenders,feature_types,target_scale,Normalize = True):
    dir_train = ''
    if dir_grp == 'HC':
        dir_train = './train/'
    elif dir_grp == 'BD':
        dir_train = './train_p/'
    x = []
    target = []
    max_val, min_val = load_max_min(feature_types)
    for at in attenders:
        # get each feature of features
        x_tmp = []
        for feature_type in feature_types:
            tmp_feature = np.loadtxt(dir_train + at +'/feature/'+ datatype_fn[feature_type] , delimiter=',')
            #print('extract feature_type', feature_type,len(tmp_feature))
            x_tmp = np.concatenate([x_tmp,tmp_feature])
        x.append(x_tmp)
        # get target scale
        target_tmp = np.loadtxt(dir_train + at +'/target_' + target_scale + '.csv' , delimiter=',')
        target.append(target_tmp)

    label = target
    label = np.array(label)
    if target_scale =='ymrs':
        label = ymrs_partition_to_factor(label)
    elif target_scale =='hamd':
        label = hamd_partition_to_factor(label)
    x = np.array(x)
    if Normalize:
        x = Normalized(x,max_val,min_val)
    print('x:',x.shape)
    print('label:',label.shape)
    return x ,label


if __name__ == '__main__':
    dir_grp = 'BD'
    feature_types = ['B_1','B_2','C','D','E']
    # attenders = get_wanna_attender(dir_grp,feature_types)
    # print(dir_grp,'attenders:',len(attenders),attenders)
    # x, label = loading_data(dir_grp,attenders,feature_types,target_scale='hamd')
    #
    # if two_grp_flag :
    #     if dir_grp == 'BD':
    #         dir_grp = 'HC'
    #     elif dir_grp == 'HC':
    #         dir_grp = 'BD'
    #     feature_types = ['B_1', 'B_2', 'C', 'D', 'E']
    #     attenders_another_grp = get_wanna_attender(dir_grp, feature_types)
    #     print(dir_grp, 'attenders:', len(attenders_another_grp))
    #     x_another_grp, label_another_grp = load_data(dir_grp, attenders, feature_types, target_scale='hamd')
    #     x = np.vstack((x,x_another_grp))
    #     label = np.vstack((label,label_another_grp))

    # print('res', x.shape, label.shape)