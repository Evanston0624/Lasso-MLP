

from io import StringIO
import csv
import os
import datetime
import math
from os import listdir
from os.path import isfile, join
from haversine import haversine
from scipy.signal import lombscargle
from sklearn import preprocessing
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from math import sin, cos, sqrt, atan2, radians
from sklearn.cluster import DBSCAN
import sklearn.cluster as skc
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
count_dis = 0

''' Function of this code 
(**華偉若要將data改為時段區間內每天的資料 應該要改此code)
1. 把BD和HC的量表對應到輸入資料 做feature extraction 依照其type的不同分成不同的csv檔 
    (BD輸出位於./train_p/量表/feature , HC 輸出位於./train/量表/feature)
2. 把資料 依照其是否是 於 10日內 以及 是否有超過各type的threshold 再進行 feature extraction 
'''

altman2scores = {
    '1':{ 
        "不覺得"    : 0,
        "偶爾"      : 1,
        "常常"      : 2,
        "多數時候"  : 3,
        "隨時"      : 4
    },
    '2':{ 
        "不覺得"    : 0,
        "偶爾"      : 1,
        "常常"      : 2,
        "多數時候"  : 3,
        "隨時"      : 4
    },
    '3':{ 
        "我的睡眠時間跟平常一樣"        : 0,
        "我 偶爾 睡得比平常少"          : 1,
        "我 常常 睡得比平常少"          : 2,
        "我 多數時候 睡得比平常少"      : 3,
        "我可以日夜不睡而不覺疲累"      : 4
    },
    '4':{ 
        "我的話量跟平常一樣"            : 0,
        "我 偶爾 比平常還多話"          : 1,
        "我 常常 比平常還多話"          : 2,
        "我 多數時候 比平常還多話"      : 3,
        "我 隨時 在說話"  : 4
    },
    '5':{ 
        "並沒有"    : 0,
        "偶爾"      : 1,
        "常常"      : 2,
        "多數時候"  : 3,
        "隨時"      : 4
    }
}

def mark_info_week( data, loc):
    weekday = []
    for row in data[ loc]:
        weekday.append( row.weekday())
    data[ 'weekday'] = weekday
    return

def get_repeat_idx( seq):
    repeat  =   []
    tmp = seq[ -1]
    for i in range( len(seq)-2, -1, -1):
        if seq[ i] == tmp:
            repeat.append( i)
        else:
            tmp = seq[ i]
    return repeat



def drop_repeat_pd( repeat, data):
    for i in repeat:
        data = data.drop( i)
    return data

###################################################################################


def altman_to_score( data, dict1):
    ans = []
    idx = 1
    # Q1
    s = data.iloc[0, idx + 1]
    keyword = s.split()[1]
    ans.append( dict1.get('1').get(keyword))
    # Q2
    s = data.iloc[0, idx + 2]
    keyword = s.split()[1]
    ans.append( dict1.get('2').get(keyword))
    # Q3
    s = data.iloc[0, idx + 3]
    keyword = s.split('，')[1]
    ans.append( dict1.get('3').get(keyword))
    # Q4
    s = data.iloc[0, idx + 4]
    keyword = s.split('，')[1]
    ans.append( dict1.get('4').get(keyword))
    # Q5
    s = data.iloc[0, idx + 5]
    keyword = s.split()[1]
    ans.append( dict1.get('5').get(keyword))
    return ans 

###############################################################################


def feature_selfscale_dass():
    global dir_train
    attend = []
    for f in os.listdir( dir_train):
        if os.path.isdir( dir_train + f):
            ### create feature folder
            try:
                if not os.path.exists( dir_train + f + '/' + 'feature'):
                    os.makedirs( dir_train + f + '/' + 'feature')
            except OSError:
                print ('Error: Creating directory. ' +  dir_train + f + '/' + 'feature')
            ### read file
            try:
                df =  dir_train + f + '/' + 'selfscale_dass.csv'
                dr =  pd.read_csv( df, index_col=False)
                ### get attendance
                if len( dr) > 0:
                    attend.append( f)
                    print( f)
                    ###
                    ans     =   []
                    num_Q_dass  =   21
                    for i in range( num_Q_dass):
                        ans.append( dr.iloc[0, 2 + i])
                    total   =   sum(ans)
                    ans.append( total)
                    ###
                    np.savetxt( dir_train + f + '/feature/' + 'dass.csv', ans, delimiter=",")
                    np.savetxt( dir_train + f + '/feature/' + 'dass_w.csv', np.array( [total]), delimiter=",")
            except (FileNotFoundError):
                pass
    np.savetxt( dir_train + 'dass_attend.csv', np.array( attend), delimiter=",", fmt="%s")
    return attend


###############################################################################


def feature_selfscale_altman():
    global dir_train
    global altman2scores
    attend = []
    for f in os.listdir( dir_train):
        if os.path.isdir( dir_train + f):
            ### create feature folder
            try:
                if not os.path.exists( dir_train + f + '/' + 'feature'):
                    os.makedirs( dir_train + f + '/' + 'feature')
            except OSError:
                print ('Error: Creating directory. ' +  dir_train + f + '/' + 'feature')
            ### read file
            try:
                df = dir_train + f + '/' + 'selfscale_altman.csv'
                dr =  pd.read_csv( df, index_col=False)
                ### get attendance
                if len( dr) > 0:
                    attend.append( f)
                    print( f)
                    ### transfer ans_text to score
                    ans = altman_to_score( dr, altman2scores)
                    ### get total score
                    total = sum( ans)
                    ans.append( total)
                    ###
                    np.savetxt( dir_train + f + '/feature/' + 'altman.csv', ans, delimiter=",")
                    np.savetxt( dir_train + f + '/feature/' + 'altman_w.csv', np.array( [ total]), delimiter=",")
            except (FileNotFoundError):
                pass
    np.savetxt( dir_train + 'altman_attend.csv', np.array( attend), delimiter=",", fmt="%s")
    return attend


###################################################################################


def get_Demo( data, days):
    ###
    ans     =   []
    idx_value = '2'
    short = 0
    ###
    if len( data) < days:
        short = days - len( data)
        days = len( data)
    ### get the latest days' Demotion
    for i in range( len( data)-days , len( data)):
        tmp = data[ idx_value].iloc[ i]
        ans.append( tmp - 3)

    mean    =   sum( ans) / len( ans)
    std     =   np.std( ans)
    # fill the short days
    # all data
    for i in range( short):
        ans.append( mean)
    ans.append( mean)
    ans.append( std)
    # week data
    ans_w = []
    ans_w.append( mean)
    ans_w.append( std)
    return ans, ans_w


def feature_Demotion():
    global  dir_train
    ################################
    attend = []
    ####
    for f in os.listdir( dir_train):
        if os.path.isdir( dir_train + f):
            # create feature folder
            try:
                if not os.path.exists( dir_train + f + '/' + 'feature'):
                    os.makedirs( dir_train + f + '/' + 'feature')
            except OSError:
                print ('Error: Creating directory. ' +  dir_train + f + '/' + 'feature')
            # read file
            try:
                df =  dir_train + f + '/' + 'info_Demotion.csv'
                dr =  pd.read_csv( df, index_col=False)
                print( len(dr))

                if len( dr) > 0:
                    print( len(attend))
                    print( f)
                    attend.append( f)
                    ### 
                    dateFormatter = "%Y-%m-%d %H:%M:%S"
                    idx_time = '19'
                    dr[ idx_time] = pd.to_datetime(dr[ idx_time]).dt.strftime( dateFormatter)
                    dr[ idx_time] = dr[ idx_time].astype('datetime64[ns]')
                    mark_info_week( dr, idx_time)
                    ### drop repeat
                    repeat = get_repeat_idx( list(dr.weekday))
                    dr = drop_repeat_pd( repeat, dr)

                    base = []
                    base_w = []
                    base_threshold = 3
                    #if dr.empty:
                    if len(dr) < base_threshold:
                        print(f, 'base is empty')
                        base =  np.empty(8)
                        base[:] = np.nan
                        base_w = np.empty(2)
                        base_w[:] = np.nan
                    else:
                        base, base_w  =  get_Demo( dr, 6)

                    np.savetxt( dir_train + f + '/feature/' + 'Demotion_w.csv', base_w, delimiter=",")
            except (FileNotFoundError):
                pass
    np.savetxt( dir_train + 'Demotion_attend.csv', np.array( attend), delimiter=",", fmt="%s")
    return attend


def get_media( data, days):
    ###
    ans     =   []
    idx_value = [ '5', '6', '7', '8', '9', '10', '11']
    short = 0
    ###
    if len( data) < days:
        short = days - len( data)
        days = len( data)
    ### get the latest days' Demotion
    for i in range( len( data)-days , len( data)):
        ans.append( data[ idx_value].iloc[ i].tolist())
    sum_cols = [sum( x) for x in zip( *ans)]
    # ans.append( data[i])
    mean    =   [x / len( ans) for x in sum_cols]
    std     =   np.std( ans, axis = 0)
    # 2d to 1d
    ans = [j for sub in ans for j in sub]
    ### fill the short days
    for i in range( short):
        ans = ans + mean
    # all data
    ans = ans + mean
    ans = ans + std.tolist()
    # week data
    ans_w = []
    ans_w = ans_w + mean
    ans_w = ans_w + std.tolist()
    return ans, ans_w



def feature_seven_emo():
    global dir_train
    ################################
    #threshold_media     =   6
    threshold_media = 1
    ################################
    attend = []
    ####
    for f in os.listdir( dir_train):
        if os.path.isdir( dir_train + f):
            # create feature folder
            try:
                if not os.path.exists( dir_train + f + '/' + 'feature'):
                    os.makedirs( dir_train + f + '/' + 'feature')
            except OSError:
                print ('Error: Creating directory. ' +  dir_train + f + '/' + 'feature')
            # read speech
            try:
                df      =  dir_train + f + '/' + 'info_speech.csv'
                dr_s    =  pd.read_csv( df, index_col=False)
                # read text
                df      =  dir_train + f + '/' + 'info_text.csv'
                dr_t    =  pd.read_csv( df, index_col=False)
                # read video
                df      =  dir_train + f + '/' + 'info_video.csv'
                dr_v    =  pd.read_csv( df, index_col=False)
                # combine all the media data
                dr = pd.concat( [ dr_t, dr_s, dr_v])
                print( 'len' + str( len(dr)))
                ########
                if len( dr) >= threshold_media:
                    print( len( attend))
                    print( f)
                    attend.append( f)
                    ### 
                    dateFormatter = "%Y-%m-%d %H:%M:%S"
                    idx_time = '19'
                    # sort the data by datetime
                    dr =  dr.sort_values( by = idx_time)
                    # format time and mark weekday
                    dr[ idx_time] = pd.to_datetime(dr[ idx_time]).dt.strftime( dateFormatter)
                    dr[ idx_time] = dr[ idx_time].astype('datetime64[ns]')
                    mark_info_week( dr, idx_time)
                    ###### get groups
                    ### group 1
                    print('group 1')
                    base = []
                    base_w = []
                    #if dr.empty:
                    base_threshold = 3
                    if len(dr) < base_threshold:
                        print(f, 'base is empty')
                        base = np.empty(49)
                        base[:] = np.nan
                        base_w = np.empty(14)
                        base_w[:] = np.nan
                    else:
                        base, base_w = get_media(dr, 5)
                    ###
                    all_group = base
                    all_group_w = base_w
                    print( len( all_group))
                    np.savetxt( dir_train + f + '/feature/' + 'seven_emo.csv', all_group, delimiter=",")
                    np.savetxt( dir_train + f + '/feature/' + 'seven_emo_w.csv', all_group_w, delimiter=",")
            except (FileNotFoundError):
                pass
    np.savetxt( dir_train + 'media_attend.csv', np.array( attend), delimiter=",", fmt="%s")
    return attend

def split_sleep( data):
    ### transfer record time
    idx_record = '19'
    dateFormatter = "%Y-%m-%d %H:%M:%S"
    data[ idx_record]   = pd.to_datetime( data[ idx_record]).dt.strftime( dateFormatter)
    data[ idx_record]   = data[ idx_record].astype('datetime64[ns]')
    mark_info_week( data, idx_record)
    ### transfer record value
    idx_value = '2'
    dateFormatter = "%Y-%m-%d %H:%M:%S"
    data[ idx_value]    = pd.to_datetime( data[ idx_value]).dt.strftime( dateFormatter)
    data[ idx_value]    = data[ idx_value].astype('datetime64[ns]')
    ###
    value_wake  =   []
    value_sleep =   []
    wd_wake     =   []
    wd_sleep    =   []
    record_wake =   []
    record_sleep=   []
    #
    type_wake   =   5
    type_sleep  =   8
    idx_type    =   '1'
    # split sleep & wake
    for i in range( len( data)):
        if data[ idx_type].iloc[i] == type_wake:
            value_wake.append(  data[ idx_value].iloc[ i])
            record_wake.append( data[ idx_record].iloc[ i])
            wd_wake.append(     data[ idx_value].iloc[ i].weekday())
        elif data[ idx_type].iloc[i] == type_sleep:
            value_sleep.append(     data[ idx_value].iloc[ i])
            record_sleep.append(    data[ idx_record].iloc[ i])
            wd_sleep.append(        data[ idx_value].iloc[ i].weekday())
    return value_wake, value_sleep, wd_wake, wd_sleep, record_wake, record_sleep

def del_repeat_sleep( data):
    #
    del_wake    = []
    #
    value_wake, value_sleep, wd_wake, wd_sleep, record_wake, record_sleep = split_sleep( data)
    if not record_wake:
        return  False,0,0,0,0,0
    tmp_record  =   record_wake[ -1] + datetime.timedelta( hours = 24)
    tmp_value   =   value_wake[ -1]  + datetime.timedelta( hours = 24)
    #
    for idx_wake in range( len( record_wake)-1, -1, -1):
        if  abs( tmp_record - record_wake[ idx_wake]) < datetime.timedelta( hours = 3):
            # CLOSE record time
            if abs( tmp_value - value_wake[ idx_wake]) < datetime.timedelta( hours = 20):
                # CLOSE value time
                # delete
                del_wake.append( idx_wake)
            else:
                # FAR value time
                print( '補值')
                tmp_record  = record_wake[ idx_wake]
                tmp_value   = value_wake[ idx_wake]
        else:
            # FAR record time
            tmp_record  = record_wake[ idx_wake]
            tmp_value   = value_wake[ idx_wake]
    ### delete repeat wake data
    for i in del_wake:
        print( 'delete : ' + str( i))
        del value_wake[ i]
        del wd_wake[ i]
        del record_wake[ i]
    return value_wake, value_sleep, wd_wake, wd_sleep, record_wake, record_sleep



def get_sleep_pair( data):
    value_wake, value_sleep, wd_wake, wd_sleep, record_wake, record_sleep = del_repeat_sleep( data)
    if value_wake == False:
        return ''
    ######### get pairs#########
    Sleep = pd.DataFrame( columns = [ 'sleep', 'wake', 'duration', 'weekday'])
    threshold_duration  =   15
    tmp_record  =   record_wake[ -1] + datetime.timedelta( hours = 24)
    #
    for idx_wake in range( len( value_wake)-1, -1, -1):
        tmp_wake    =   value_wake[ idx_wake]
        # if ( tmp_record - tmp_wake) > datetime.timedelta( hours = 2):
        if True:
            tmp_record  = tmp_wake
            # sleep time before wake time
            bool_ago    =   [ i < tmp_wake for i in value_sleep]
            if sum( bool_ago) == 0:
                continue
            # get the last sleep time
            idx_sleep   =   max( [ i for i, x in enumerate( bool_ago) if x])
            # determine whether a group or not
            duration    =   tmp_wake - value_sleep[ idx_sleep]
            if duration > datetime.timedelta( hours = threshold_duration):
                # 補值
                ################################################################
                ################################################################
                continue
            else:
                Sleep = Sleep.append(
                    {   'sleep'     : value_sleep[ idx_sleep], 
                        'wake'      : value_wake[ idx_wake], 
                        'duration'  : duration,
                        'weekday'   : wd_wake[ idx_wake]
                        },ignore_index=True)
    return Sleep


def get_sleep( data, days):
    ans     =   []
    idx_value = '2'
    short = 0
    dict_duration = {
        'short'      :   0,
        'medium'    :   1,
        'long'     :   2,
    }
    dict_midpoint = {
        'early'     :   0,
        'middle'    :   1,
        'late'      :   2,
    }
    dict_rg = {
        'unstable'    :   0,
        'medium'    :   1,
        'stable'  :   2,
    }
    print('start get_sleep')
    if len( data) < days:
        short = days - len( data)
        days = len( data)

    # get duration(hour) and the score of duration
    duration_type   = []
    durations       = []

    for i in range( len( data)-days , len( data)):
        d =   data[ 'duration'].iloc[i]
        durations.append( d.seconds/3600 )
        if d <= datetime.timedelta( hours = 5):
            tmp =   dict_duration[ 'short']
        elif  d > datetime.timedelta( hours = 7.5):
            tmp =   dict_duration[ 'long']
        else:
            tmp =   dict_duration[ 'medium']
        duration_type.append( tmp)

    duration_mean   =   sum( durations) / len( durations)
    duration_std    =   np.std( durations)

    duration_type_mean       =   sum( duration_type) / len( duration_type)
    duration_type_std        =   np.std( duration_type)

    for i in range( short):
        duration_type.append( duration_type_mean)
        durations.append( duration_mean)
    # get midpoint and the score of midpoint
    midpnts         =   []
    midpnts_type    =   []
    for i in range( len( data)-days , len( data)):
        s = data[ 'sleep'].iloc[ i]
        w = data[ 'wake'].iloc[ i]

        spad = 0
        if s.hour < 12:
            spad = 24
        wpad = 0
        if w.hour < 12:
            wpad = 24

        sleep = datetime.timedelta( hours   = s.hour,
                                    minutes = s.minute,
                                    seconds = s.second) + datetime.timedelta( hours = spad)
        wake = datetime.timedelta(  hours   = w.hour,
                                    minutes = w.minute,
                                    seconds = w.second) + datetime.timedelta( hours = wpad)
        mid   = ( sleep + wake).total_seconds() / 2 / 3600
        ###
        if mid      <=  ( 24 + 2 ) :
            tmp =   dict_midpoint[ 'early']
        elif  mid   >   ( 24 + 4 ) :
            tmp =   dict_midpoint[ 'late']
        else:
            tmp =   dict_midpoint[ 'middle']

        midpnts.append( mid)
        midpnts_type.append( tmp)
    ### fill the short days
    midpnts_mean   =   sum( midpnts) / len( midpnts)
    midpnts_std    =   np.std( midpnts)
    #
    midpnts_type_mean      =   sum( midpnts_type) / len( midpnts_type)
    midpnts_type_std       =   np.std( midpnts_type)
    #
    for i in range( short):
        midpnts_type.append( midpnts_type_mean)
        midpnts.append( midpnts_mean)

    # get regularity and the score of regularity
    rg      = []
    rg_type = []
    for i in range( len( data)-days , len( data)-1 ):
        s1 = data[ 'sleep'].iloc[ i]
        w1 = data[ 'wake'].iloc[ i]
        spad = 0
        if s1.hour < 12:
            spad = 24
        wpad = 0
        if w1.hour < 12:
            wpad = 24
        sleep1 = ( datetime.timedelta( hours   = s1.hour,
                                    minutes  = s1.minute,
                                    seconds  = s1.second) + datetime.timedelta( hours = spad)).total_seconds()
        wake1 = ( datetime.timedelta(  hours   = w1.hour,
                                    minutes  = w1.minute,
                                    seconds  = w1.second) + datetime.timedelta( hours = wpad)).total_seconds()
        s2 = data[ 'sleep'].iloc[ i+1]
        w2 = data[ 'wake'].iloc[ i+1]
        spad = 0
        if s2.hour < 12:
            spad = 24
        wpad = 0
        if w2.hour < 12:
            wpad = 24
        sleep2 = ( datetime.timedelta( hours   = s2.hour,
                                    minutes  = s2.minute,
                                    seconds  = s2.second) + datetime.timedelta( hours = spad)).total_seconds()
        wake2 = ( datetime.timedelta(  hours   = w2.hour,
                                    minutes  = w2.minute,
                                    seconds  = w2.second) + datetime.timedelta( hours = wpad)).total_seconds()
        a = max( [ sleep1, sleep2, wake1, wake2]) - min( [ sleep1, sleep2, wake1, wake2])
        u = abs( sleep1 - sleep2)
        l = abs( wake1 - wake2)
        r = 1 - ( ( u+l) / ( 24*3600))
        if      r  <= 0.5 :
            tmp =   dict_rg[ 'unstable']
        elif    r  >= 0.75 :
            tmp =   dict_rg[ 'stable']
        else:
            tmp =   dict_rg[ 'medium']
        rg.append( r)
        rg_type.append( tmp)
    ### fill the short days
    if len( rg) == 0:
        rg.append( 1)
        rg_type.append( 0)
        short = short -1
    rg_mean    =   sum( rg) / len( rg)
    rg_std     =   np.std( rg)
    rg_type_mean    =   sum( rg_type) / len( rg_type)
    rg_type_std     =   np.std( rg_type)
    for i in range( short):
        rg_type.append( rg_type_mean)
        rg.append( rg_mean)
    ############################################################
    ### get sleep feature
    ### all data
    ans.extend( durations)
    ans.append( duration_mean)
    ans.append( duration_std)
    ans.extend( duration_type)
    ans.append( duration_type_mean)
    ans.append( duration_type_std)
    #
    ans.extend( midpnts)
    ans.append( midpnts_mean)
    ans.append( midpnts_std)
    ans.extend( midpnts_type)
    ans.append( midpnts_type_mean)
    ans.append( midpnts_type_std)
    #
    ans.extend( rg)
    ans.append( rg_mean)
    ans.append( rg_std)
    ans.extend( rg_type)
    ans.append( rg_type_mean)
    ans.append( rg_type_std)
    ### week data
    ans_w = []
    ans_w.append( duration_mean)
    ans_w.append( duration_std)
    ans_w.append( duration_type_mean)
    ans_w.append( duration_type_std)
    #
    ans_w.append( midpnts_mean)
    ans_w.append( midpnts_std)
    ans_w.append( midpnts_type_mean)
    ans_w.append( midpnts_type_std)
    #
    ans_w.append( rg_mean)
    ans_w.append( rg_std)
    ans_w.append( rg_type_mean)
    ans_w.append( rg_type_std)
    return ans, ans_w



def feature_Sleep():
    global dir_train
    ################################
    #threshold_Sleep     = 4 * 2
    threshold_Sleep = 2 * 1
    ################################
    attend = []
    f_num=0
    ####
    for f in os.listdir( dir_train):
        if os.path.isdir( dir_train + f):
            # create feature folder
            try:
                if not os.path.exists( dir_train + f + '/' + 'feature'):
                    os.makedirs( dir_train + f + '/' + 'feature')
            except OSError:
                print ('Error: Creating directory. ' +  dir_train + f + '/' + 'feature')
            # read file
            if  f =='':
                pass
            else:
                try:
                    print('reading')
                    print( f)

                    df =  dir_train + f + '/' + 'info_sleeptime.csv'
                    dr =  pd.read_csv( df, index_col=False)
                    print('dr len')
                    print( len(dr))
                    ########
                    if len( dr) >= threshold_Sleep:
                        #
                        print('bigger than threshold')
                        pairs = get_sleep_pair( dr)
                        print('get threshlod')
                        if len( pairs)>0:
                            print( len(attend))
                            f_num=f_num+1
                            print(f_num)
                            print('pairs')
                            print(pairs)
                            attend.append( f)
                            ### group 1
                            print( 'group 1')
                            base_threshold = 3
                            base = []
                            base_w = []
                            if len(pairs) < base_threshold:
                                print('base is empty')
                                base = np.empty(14)
                                base[:] = np.nan
                                base_w = np.empty(12)
                                base_w[:] = np.nan
                            else:
                                base, base_w = get_sleep( pairs, 5)

                            all_group = base
                            all_group_w = base_w
                            print( len( all_group))
                            np.savetxt( dir_train + f + '/feature/' + 'sleep.csv', all_group, delimiter=",")
                            np.savetxt( dir_train + f + '/feature/' + 'sleep_w.csv', all_group_w, delimiter=",")
                except (FileNotFoundError):
                    pass
    np.savetxt( dir_train + 'sleep_attend.csv', np.array( attend), delimiter=",", fmt="%s")
    return attend

if __name__ == '__main__':
    # feature extraction for HC
    dir_train = "train/"
    attend_altman = feature_selfscale_altman()
    attend_dass = feature_selfscale_dass()
    attend_Demo = feature_Demotion()
    attend_sleep = feature_Sleep()
    attend_seven_emo = feature_seven_emo()
    # feature extraction for BD
    dir_train = "train_p/"
    attend_altman = feature_selfscale_altman()
    attend_dass = feature_selfscale_dass()
    attend_Demo = feature_Demotion()
    attend_sleep = feature_Sleep()
    attend_seven_emo = feature_seven_emo()