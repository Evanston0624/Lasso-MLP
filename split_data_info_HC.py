

from io import StringIO
import csv
import os
from collections import Counter
import pandas as pd

''' Function of this code 
1. 把HC的INFO.csv資料 依照其type的不同分成不同的csv檔 (輸出位於./spilt_p/info/0~8) ** 但好像之後用不到
2. 把HC的INFO.csv資料 依照其"帳號"將不同人的資料分到個別資料夾 (輸出位於./spilt/info/user)
'''
##
file_info   = "INFO.csv"

##
dir_split   = "./split/"
dir_raw     = "./raw/"
dir_info    = "info/"


# create directory
d = dir_split+dir_info
try:
    if not os.path.exists(d):
        os.makedirs(d)
except OSError:
    print ('Error: Creating directory. ' +  d)

# 0=文字
# 1=語音
# 2=純情緒標記
# 3=影片
# 4=每日情緒
# 5=睡眠時間_1
# 6=每周自評量表_1
# 7=每周自評量表_2
# 8=睡眠時間_2

s = ["文字", "語音", "純情緒標記", "影片", "每日情緒", "睡眠時間1",
     "每周自評量表1", "每周自評量表2", "睡眠時間2"]

# create directory by data type
for i in range( 0, len(s)):
    d = dir_split + dir_info + str(i)
    try:
        if not os.path.exists(d):
            os.makedirs(d)
    except OSError:
        print ('Error: Creating directory. ' +  d)

####################################################################

#  Read file
with open(dir_raw + file_info, newline='' ,encoding="ISO-8859-1") as csvfile:
    data_info = list( csv.reader(csvfile))


# 0 user
# 1 type
# 19 date
# cut the head row
data_info = data_info[1:]


# Count how many users
users = [ row[0] for row in data_info]
c_user = Counter(users)


# Count of types
t = [ row[1] for row in data_info]
c_type = Counter(t)


##############################################################


# output the sorted count as csv

of = dir_split + "info_count_by_type.csv"
with open(of, encoding='utf-8-sig', mode='w') as fp:
    fp.write('name, count\n')  
    for tag, count in c_type.most_common():
        n = fp.write('{},{}\n'.format(s[ int(tag)], count))  




of = dir_split + "info_count_by_user.csv"
with open(of, encoding='utf-8-sig', mode='w') as fp:
    n = fp.write('name, total')
    for i in s :
        n = fp.write(','+i)  
    n = fp.write('\n')  
    for tag, count in c_user.most_common():
        matches = [x for x in data_info if x[0]==tag]
        c = []
        for i in range(0, len(s)):
            c.append( len( [x for x in matches if int(x[1])==i]))
       # n = fp.write('{},{},{},{},{},{},{},{},{}\n'.format(tag, count, c[0], c[1], c[2], c[3], c[4], c[5], c[6]))  
        n = fp.write('{},{}'.format(tag, count))  
        for i in c:
            n = fp.write(', {}'.format(i)) 
        n = fp.write('\n')


# Split data by info type
for types in c_type.keys():
    od = dir_split + dir_info
    # split data by type
    t = []
    for row in data_info:
        if row[1] == types:
            t.append(row)
    # split data by user
    for name in c_user.keys():
        of = od + types + '/' + name + '.csv'
        with open(of, 'w', newline='', encoding="ISO-8859-1") as csvfile:
            writer = csv.writer(csvfile)
            for row in t:
                if row[0] == name :
                    n=writer.writerow(row)

####################################################################

# create directory
d = dir_split + dir_info + 'user/'
try:
    if not os.path.exists(d):
        os.makedirs(d)
except OSError:
    print ('Error: Creating directory. ' +  d)



# Split data by user
for user in c_user.keys():
    of = dir_split + dir_info + "user/" + user + ".csv"
    # split data by user
    with open(of, 'w', newline='', encoding="ISO-8859-1") as csvfile:
        writer = csv.writer(csvfile)
        for row in data_info:
            if row[0] == user:
                n=writer.writerow(row)
