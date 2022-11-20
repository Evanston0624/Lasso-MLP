

from io import StringIO
import csv
import os
from collections import Counter
import datetime
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

################################################################################

#### get info
# 0=文字
# 1=語音
# 2=純情緒標記
# 3=Video
# 4=每日情緒
# 5=起床時間
# 6=每周自評量表
# 7=每周自評量表
# 8=睡覺時間



def get_train_info_date( tdr, record_user, record_date, types, idx_time):
    dir_train   = "train/"
    # target date
    end_date    = datetime.datetime.strptime(record_date, "%Y/%m/%d")
    # the day range
    day_range   =   10
    start_date  = end_date - datetime.timedelta( days = day_range)
    target = tdr[( ( tdr[ idx_time] > start_date) & 
                   ( tdr[ idx_time] < end_date  ))]
    df = dir_train + record_user + "_" + end_date.strftime("%Y_%m_%d")
    try:
        if not os.path.exists( df):
            os.makedirs( df)
    except OSError:
        print ('Error: Creating directory. ' +  df)
    ########
    target.to_csv( df + "/info_"+ types +".csv", index=False)
    print( df + " info_" + types + " complete")
    return




def get_train_info( record_user, record_date):
    dir_split   = "./split/"
    dir_info    = "info/"
    try:
        dr =  pd.read_csv( dir_split + dir_info + "user/" + record_user + ".csv", header=None,  skiprows=1, index_col=False)
        # reformat date
        idx_type = 1
        idx_time = 19
        dateFormatter = "%Y-%m-%d %H:%M:%S"
        dr[ idx_time] = pd.to_datetime(dr[ idx_time]).dt.strftime( dateFormatter)
        dr[ idx_time] = dr[ idx_time].astype('datetime64[ns]')
        # 0=文字
        tdr = dr[ dr[ idx_type] == 0]
        get_train_info_date( tdr, record_user, record_date, "text", idx_time)
        # 1=語音
        tdr = dr[ dr[idx_type] == 1]
        get_train_info_date( tdr, record_user, record_date, "speech", idx_time)
        # 2=純情緒標記
        tdr = dr[ dr[idx_type] == 2]
        get_train_info_date( tdr, record_user, record_date, "emotion", idx_time)
        # 3=Video
        tdr = dr[ dr[idx_type] == 3]
        get_train_info_date( tdr, record_user, record_date, "video", idx_time)
        # 4=每日情緒
        tdr = dr[ dr[idx_type] == 4]
        get_train_info_date( tdr, record_user, record_date, "Demotion", idx_time)
        # 5=起床時間
        # 8=睡覺時間
        tdr = dr[ ( (dr[idx_type] == 5) | ( dr[idx_type] == 8))]
        get_train_info_date( tdr, record_user, record_date, "sleeptime", idx_time)
        # 6=每周自評量表
        # 7=每周自評量表
        tdr = dr[ ( (dr[idx_type] == 6) | ( dr[idx_type] == 7))]
        get_train_info_date( tdr, record_user, record_date, "selfscale", idx_time)
    except (FileNotFoundError):
        pass
    return



def get_train_target( record_user, record_date, data):
    ########
    end_date    = datetime.datetime.strptime(record_date, "%Y/%m/%d")
    df = dir_train + record_user + "_" + end_date.strftime("%Y_%m_%d")
    try:
        if not os.path.exists( df):
            os.makedirs( df)
    except OSError:
        print ('Error: Creating directory. ' +  df)
    ########
    # YMRS
    YMRS    = np.array( data[ 3 : 15], dtype=np.uint8)
    np.savetxt( df + "/target_ymrs.csv", YMRS, delimiter=",")
    # HAMD
    HAMD    = np.array( data[ 16 :], dtype=np.uint8)
    np.savetxt( df + "/target_hamd.csv", HAMD, delimiter=",")
    print( df + " Target complete")
    return 



def read_selfscale( ):
    #
    dir_raw     = "./raw/"
    file_dass   = 'dass.csv'
    file_alt    = 'altman.csv'
    dateFormatter = "%Y/%m/%d"
    # DASS
    dass =  pd.read_csv( dir_raw + file_dass, header=None, skiprows=[0], index_col=False)
    idx_time = 0
    for i in range( len( dass)):
        print( 'Dass  ' + str(i))
        # format
        tmp = dass[ idx_time][ i].split()[0]
        #print('tmp',tmp)
        dass[ idx_time][ i] = datetime.datetime.strptime( tmp, dateFormatter)
    print('processing Altman')
    # Altman
    altman =  pd.read_csv( dir_raw + file_alt, header=None, skiprows=[0], index_col=False)
    idx_time = 0
    for i in range( len( altman)):
        print( 'Altman  ' + str(i))
        #print('tmp', tmp)
        tmp = altman[ idx_time][ i].split()[0]
        altman[ idx_time][ i] = datetime.datetime.strptime( tmp, dateFormatter)
    return dass, altman



def get_train_selfscale( dass, altman, record_user, record_date):
    #
    day_range   = 10
    # get date and data
    end_date    = datetime.datetime.strptime( record_date, "%Y/%m/%d")
    start_date  = end_date - datetime.timedelta( days = day_range)
    df = dir_train + record_user + "_" + end_date.strftime("%Y_%m_%d")
    try:
        if not os.path.exists( df):
            os.makedirs( df)
    except OSError:
        print ('Error: Creating directory. ' +  df)
    ############# DASS ############
    idx_time = 0
    idx_user = 1
    # get user
    udr = dass[ dass[ idx_user] == record_user]
    # get after start time
    sdr = udr[ udr[ idx_time] > start_date]
    # get before end time
    edr = sdr[ sdr[ idx_time] < end_date]
    # get latest one scale
    want = edr.tail( 1)
    want.to_csv( df + "/selfscale_dass.csv", index=False)
    print( df + " DASS complete")
    ############# Altman ############
    idx_time = 0
    idx_user = 1
    # get user
    udr = altman[ altman[ idx_user] == record_user]
    # get after start time
    sdr = udr[ udr[ idx_time] > start_date]
    # get before end time
    edr = sdr[ sdr[ idx_time] < end_date]
    # get last one
    want = edr.tail( 1)
    want.to_csv( df + "/selfscale_altman.csv", index=False)
    print( df + " Altman complete")
    return



####################################################################
####################################################################
####################################################################
####################################################################
####################################################################


##
file_info   = "INFO.csv"
file_hamd   = "scale_lab.csv"
file_user   = "users.csv"
file_lab    = "scale_lab.csv"
file_dass   = 'dass.csv'
file_alt    = 'altman.csv'

##
dir_raw     = "./raw/"
dir_split   = "./split/"
dir_info    = "info/"
dir_train   = "train/"


# create train directory
d = dir_train
try:
    if not os.path.exists(d):
        os.makedirs(d)
except OSError:
    print ('Error: Creating directory. ' +  d)


dass, altman = read_selfscale()


#### Read file
data_lab = []
with open(dir_raw + file_lab, newline='' ,encoding="gb2312",  errors='ignore') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        #print(row)
        data_lab.append(row)


user=[]
HC_list=['mhmcmd104', 'mhmcmd103', 'mhmcmd101', 'mhmcph04', 'mhmcph03', 'mhmcmd202', 'mhmcmd201', 'mhmcmd203', 'mhmcmd105', 'mhmcmd102', 'mhmcph02', 'mhmcph01', 'mhmcph05','nd8081026', 'ksstp93017', 'jeremychang8', 'NF6091011', 'casper980111', 'tmty','76084481','76084627','mhmcphd02','NF609101','76094169']
#  cut head
data_lab = data_lab[1:]
#
check_num=0
for i in range( 0, len( data_lab)):
    breakpoint = -1
    if i > breakpoint:
        print( i)
        rec_date =  data_lab[i][0].split(" ")[0]
        user_name = data_lab[i][1]
        if data_lab[i][1] not in user:
            user.append(data_lab[i][1])
        print("user:", data_lab[i][1], "date:", rec_date)
        #check if in HC list
        check = False
        for li in HC_list:
            if li in user_name:
                check = True
                if li == '76084627':
                    user_name = 'p76084627'
                elif li == '76084481':
                    user_name = 'p76084481'
                elif li == '76094169':
                    user_name = 'p76094169'
                break
        if check:
            r_date =  datetime.datetime.strptime(data_lab[i][2], "%Y/%m/%d")
            print(r_date)
            #threshold_date = datetime.datetime(year=2020, month=10, day=5)
            threshold_date = datetime.datetime(year=2018, month=10, day=5)
            if r_date > threshold_date:
                print('recording:',r_date)
                #O
                get_train_selfscale( dass, altman, user_name, data_lab[i][2] )
                #O
                get_train_target( user_name, data_lab[i][2], data_lab[i])
                check_num+=1
                #O
                get_train_info( user_name, data_lab[i][2])
        else:
            print('user',user_name,'is Not HC.')
print("processing data num:",check_num)
print('user',user)


