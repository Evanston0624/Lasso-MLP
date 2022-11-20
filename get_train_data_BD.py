

from io import StringIO
import csv
import os
from collections import Counter
import datetime
import numpy as np
import pandas as pd
import warnings
''' Function of this code
(**華偉可能要將前10天改成想要的天數) 
依照BD 評測量表的時間點的前十天 將split_p 中的資料 及 目標量表分數(HAMD/YMRS)
放入 ./train_p/"BD_phone"_"評測日期"
'''
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
    dir_train   = "train_p/"
    # target date
    end_date    = record_date
    # the day range
    day_range   =   10
    # print('type record date',type(record_date))
    start_date  = end_date - datetime.timedelta( days = day_range)
    target = tdr[( ( tdr[ idx_time] > start_date) & 
                   ( tdr[ idx_time] < end_date  ))]
    print('info find',len(target))
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
    dir_split   = "./split_p/"
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
        print('type record date', type(record_date))
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
    #end_date    = datetime.datetime.strptime(record_date, "%Y/%m/%d")
    end_date = record_date
    df = dir_train + record_user + "_" + end_date.strftime("%Y_%m_%d")
    print('df',df)
    try:
        if not os.path.exists( df):
            os.makedirs( df)
    except OSError:
        print ('Error: Creating directory. ' +  df)
    ########
    # YMRS
    ymrs = data[ 3 : 14]
    ymrs.append(data[38])
    print('ymrs',ymrs)
    YMRS    = np.array( ymrs, dtype=np.uint8)
    np.savetxt( df + "/target_ymrs.csv", YMRS, delimiter=",")
    # HAMD
    hamd = data[14:38]
    hamd.append(data[39])
    print('hamd',hamd)
    HAMD    = np.array( hamd, dtype=np.uint8)
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



def get_train_selfscale( dass, altman, phone, record_date,record_user):
    #
    day_range   = 10
    # get date and data
    end_date    = record_date
    start_date  = end_date - datetime.timedelta( days = day_range)
    df = dir_train + phone + "_" + end_date.strftime("%Y_%m_%d")
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
    print('find in dass',len(edr))
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
    print('find in altman', len(edr))
    # get last one
    want = edr.tail( 1)
    want.to_csv( df + "/selfscale_altman.csv", index=False)
    print( df + " Altman complete")
    return



####################################################################


##
file_info   = "INFO.csv"
file_hamd   = "scale_p.csv"
file_user   = "users.csv"
file_lab    = "scale_.csv"
file_dass   = 'dass.csv'
file_alt    = 'altman.csv'

##
dir_raw     = "./raw/"
dir_split   = "./split_p/"
dir_info    = "info/"
dir_train   = "train_p/"
dir_account = './raw/p/accounts.csv'

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
# read HY scale
scale_p = pd.read_csv( dir_raw + 'p/' + 'scale_p.csv',encoding=' utf-8' )
with open(dir_raw + 'p/' + 'scale_p.csv',encoding=' utf-8', newline='' ,  errors='ignore') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        #print(row)
        data_lab.append(row)
#0505
user_list = scale_p['Number'].tolist()

#print('user_list len',len(user_list),'user_list:',user_list)
# read account
# converts => set the 'certain column' 'type' as read
account = pd.read_csv(dir_account,encoding=' utf-8',converters ={4:str})

#print(mobile)

user=[]
phone_user=[]
#  cut head
data_lab = data_lab[1:]
#
check_num=0
for i in range( 0, len( scale_p)):
    breakpoint = -1
    if i > breakpoint:
        print( i)
        rec_date =  data_lab[i][0].split(" ")[0]
        user_name = data_lab[i][1]
        check = False
        if data_lab[i][1] not in user:
            user.append(data_lab[i][1])
        #get phone
        if user_name =='912383209':
            user_name ='0912383209'
        tmp = account[account['Account']==user_name]
        if len(tmp) == 0:
            print('user',user_name,'does not match mobile in accounts.csv!')
        else:
            phone = tmp.iloc[0]['mobile']
            check = True
            print('phone',str(phone))
            if phone =='0986355280':
                check=False
        #check if in HC list

        c = rec_date.split("/")[0]
        if(len(c)==4):
            r_date = datetime.datetime.strptime(rec_date, "%Y/%m/%d")
        else:
            r_date =  datetime.datetime.strptime(rec_date, "%m/%d/%Y")
         # change to right format
        r_date = datetime.datetime(year=r_date.year, month=r_date.month, day=r_date.day)
        threshold_date = datetime.datetime(year=2020, month=10, day=5)
        if r_date > threshold_date and check:
            if phone not in phone_user:
                phone_user.append(phone)
            print("user:", user_name, "date:", r_date,'phone:',phone)
            print('recording:',r_date)
            #O
            get_train_selfscale( dass, altman, phone, r_date,user_name)
            # #O
            get_train_target( phone, r_date, data_lab[i])
            # check_num+=1
            get_train_info( phone, r_date)

print("processing data num:",check_num)
print('user',phone_user)