import warnings
from io import StringIO
import csv
import os
from collections import Counter
import pandas as pd
warnings.filterwarnings("ignore")

''' Function of this code 
1. 把BD的INFO.csv資料 依照其type的不同分成不同的csv檔 (輸出位於./spilt_p/info/0~8) ** 但好像之後用不到
2. 將INFO.csv 透過將 id_user.xlsx和accounts mapping方式
把BD的INFO.csv資料 依照其"手機號碼"將不同病人的資料分到個別資料夾 (輸出位於./spilt_p/info/user)
'''
# file name of data in database
file_info   = "INFO.csv"
file_dass   = 'dass.csv'
file_altman = 'altman.csv'

# file name of users' information
file_user_p = "id_user.xlsx"
file_account= "accounts.csv"

# directories
dir_raw_p   = "./raw/"
dir_patient = "p/"
dir_split   = "./split_p/"
dir_info    = "info/"

# create info directory
d = dir_split + dir_info
try:
    if not os.path.exists(d):
        os.makedirs(d)
except OSError:
    print ('Error: Creating directory. ' +  d)

''' meaning of type
0=文字
1=語音
2=純情緒標記
3=Video
4=每日情緒
5=起床時間
6=每周自評量表
7=每周自評量表
8=睡覺時間
''' 
str_type = ["文字", "語音", "純情緒標記", "影片", "每日情緒", "睡眠時間1",
     "每周自評量表1", "每周自評量表2", "睡眠時間2"]

# create directory by data type
for i in range(0, len( str_type)):
    d = dir_split + dir_info + str(i)
    try:
        if not os.path.exists(d):
            os.makedirs(d)
    except OSError:
        print ('Error: Creating directory. ' +  d)

# Read info csv file
data_info = pd.read_csv( dir_raw_p + file_info,  encoding="utf8")

# Read account csv file
account = pd.read_csv( dir_raw_p + dir_patient + file_account, dtype=str)
account[ 'Account']  = account[ 'Account'].astype( str) 
account[ 'mobile']   = account[ 'mobile'].astype( str) 


# Read user_id csv file
# 0 user
# 1 type
# 19 date
id_user = pd.read_excel( dir_raw_p + dir_patient + file_user_p, header = None, dtype=str)
# change type to string
id_user[ 1] = id_user[ 1].astype( str)
# remove the space
for i in range( len( id_user)):
    id_user[ 1].iloc[ i] = id_user[ 1].iloc[ i].strip()


# insert phone to info

# get account's phone
acc_p = {}
# get all account
acc = Counter( data_info[ 'Account'].tolist())
for n in acc.keys():
    # get index in registration
    tmp = account[ account[ 'Account'] == n].index
    if len( tmp) > 0:
        # get phone
        acc_p[ n] = account[ 'mobile'].iloc[ tmp[0]]
    else:
        # no phone number
        acc_p[ n] = 'NAN'


tmp = ['NAN'] * len( data_info)
for i in range( len( data_info)):
    name    = data_info[ 'Account'].iloc[i]
    tmp[ i] = acc_p[ name]


# insert
data_info[ 'phone'] = tmp

# Count how many users
phones = data_info[ 'phone'].tolist()
c_phone = Counter( phones)


t = data_info[ 'type'].tolist()
c_type = Counter(t)

# Split data by info type
datas = data_info.values.tolist()
#
for types in c_type.keys():
    od = dir_split + dir_info
    # split data by type
    t = []
    for row in datas:
        if row[1] == types:
            t.append(row)
    # split data by user
    for p in c_phone.keys():
        of = od + str( types) + '/' + p + '.csv'
        with open( of, 'w', newline='', encoding="utf8") as csvfile:
            writer = csv.writer(csvfile)
            for row in t:
                if row[ 28] == p :
                    n = writer.writerow(row)


# Split data by user

# create directory
d = dir_split + dir_info + 'user/'
try:
    if not os.path.exists(d):
        os.makedirs(d)
except OSError:
    print ('Error: Creating directory. ' +  d)


# Split data by user
for p in c_phone.keys():
    of = dir_split + dir_info + "user/" + p + ".csv"
    # split data by user
    with open(of, 'w', newline='', encoding="utf8") as csvfile:
        writer = csv.writer(csvfile)
        for row in datas:
            if row[ 28] == p:
                n = writer.writerow(row)
