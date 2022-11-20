import os
import pandas as pd
import csv
import shutil
import datetime
import numpy as np

dir_res="./obj_emo_res/"
dir_train="./train/"

data_num=0

def get_4_emo(_of_attend):
    sad = obj_of_attend['sad'].tolist()
    happy = obj_of_attend['happy'].tolist()
    neutral = obj_of_attend['neutral'].tolist()
    angry = obj_of_attend['angry'].tolist()
    res = []
    for i in range(obj_of_attend.shape[0]):
        tmp = []
        tmp.append(sad[i])
        tmp.append(happy[i])
        tmp.append(neutral[i])
        tmp.append(angry[i])
        res.append(tmp)
    return res
def get_4_emo_from_text(obj_of_attend):
    sad = obj_of_attend['object_Sadness'].tolist()
    happy = obj_of_attend['object_Happiness'].tolist()
    neutral = obj_of_attend['object_Neutral'].tolist()
    angry = obj_of_attend['object_Anger'].tolist()
    res = []
    for i in range(obj_of_attend.shape[0]):
        tmp = []
        tmp.append(sad[i])
        tmp.append(happy[i])
        tmp.append(neutral[i])
        tmp.append(angry[i])
        res.append(tmp)
    print('4 emo from text', res)
    return res

# Match audio res and account name to the ./train/feature/objemo


if __name__ == '__main__':
    #
    base_attend = []

    base_feature = []

    tmp_base_feature = []

    mapped_video = 0
    mapped_audio = 0
    mapped_text = 0

    len_3type_base = 0

    #
    ## check audio
    dr = pd.read_csv(dir_res + 'audio_from_vid_res_by_user.csv', index_col=False)
    #dr = pd.read_csv(dir_res + 'vid_res_by_user.csv', index_col=False)
    dr1 = pd.read_csv(dir_res + 'audio_res_by_user.csv', index_col=False)
    dr2 = pd.read_csv(dir_res + 'user_text_res.csv', index_col=False)
    dateFormatter = "%Y-%m-%d"
    dr['formatetime'] = pd.to_datetime(dr['formatetime']).dt.strftime(dateFormatter)
    dr['formatetime'] = dr['formatetime'].astype('datetime64[ns]')


    dr1['formatetime'] = pd.to_datetime(dr1['formatetime']).dt.strftime(dateFormatter)
    dr1['formatetime'] = dr1['formatetime'].astype('datetime64[ns]')

    # text
    dr2['Datetime'] = pd.to_datetime(dr2['Datetime']).dt.strftime(dateFormatter)
    dr2['Datetime'] = dr2['Datetime'].astype('datetime64[ns]')

    print("account num:",dr.shape[0])

    print("account num:",dr.shape[0])
    #print(df[2])
    for d in os.listdir(dir_train):
        if ".csv" not in d:
            len_3type_base = 0
            len_3type_weekday = 0
            len_3type_weekend = 0
            # map phone to account
            print(d)
            temp_res = [d]
            tmp = d.split("_")
            attend = tmp[0]
            print('attender:', attend)

            date = tmp[1] + '-' + tmp[2] + '-' + tmp[3]
            # str to datetime
            target_date = datetime.datetime.strptime(date, "%Y-%m-%d")
            obj_of_attend = dr[dr["Account"] == attend]
            # print("type of target:", type(target_date), "type of date:", type(obj_of_attend["formatetime"].iloc[0]))
            mask = (obj_of_attend['formatetime'] > target_date - datetime.timedelta(days=10)) & (
                        obj_of_attend['formatetime'] < target_date)
            obj_of_attend = obj_of_attend[mask]
            feature = []

            obj_of_attend_audio = dr1[dr1["Account"] == attend]

            mask = (obj_of_attend_audio['formatetime'] > target_date - datetime.timedelta(days=10)) & (
                    obj_of_attend_audio['formatetime'] < target_date)
            obj_of_attend_audio = obj_of_attend_audio[mask]

            # text
            obj_text_attend = dr2[dr2["Account"] == attend]

            mask = (obj_text_attend['Datetime'] > target_date - datetime.timedelta(days=10)) & (
                    obj_text_attend['Datetime'] < target_date)
            obj_text_attend = obj_text_attend[mask]

            # video
            if not obj_of_attend.empty:
                # base
                if len(obj_of_attend) > 0:
                    mapped_video += len(obj_of_attend)
                    len_3type_base += len(obj_of_attend)
                    print('vid expected len',len(obj_of_attend))
                    emo_extract_res = get_4_emo(obj_of_attend)
                    for emo in emo_extract_res:
                        tmp_base_feature.append(emo)
                    #print('get len',len(tmp_base_feature))

            # audio
            if not obj_of_attend_audio.empty:
                #base

                if len(obj_of_attend_audio) > 0:
                    mapped_audio += len(obj_of_attend_audio)
                    len_3type_base += len(obj_of_attend_audio)
                    print('audio expected len',len(obj_of_attend_audio))
                    #print('expected len', len(obj_of_attend_audio))
                    emo_extract_res = get_4_emo(obj_of_attend_audio)
                    for emo in emo_extract_res:
                        tmp_base_feature.append(emo)
                    #print('get len', len(tmp_base_feature))

            if not obj_text_attend.empty:
                # base
                if len(obj_text_attend) > 0:
                    mapped_text += len(obj_text_attend)
                    len_3type_base += len(obj_text_attend)
                    print('text expected len', len(obj_text_attend))
                    emo_extract_res = get_4_emo_from_text(obj_text_attend)
                    for emo in emo_extract_res:
                        tmp_base_feature.append(emo)

            obj_len = len(obj_of_attend_audio)+len(obj_of_attend)+len(obj_text_attend)
            # check if exceed threshold
            if len_3type_base >= 2 and (tmp_base_feature):
                print('base here')
                base_attend.append(d)
                t = np.array(tmp_base_feature)
                base_feature = np.mean(t, axis=0)
                std_feature = np.std(t, axis=0)
                print('mean',base_feature,'std',std_feature)
                base_feature = np.hstack((base_feature,std_feature) )
            else:
                base_feature = np.empty(8)
                base_feature[:] = np.nan

            all_group_w = base_feature

            print('size of obj emo feature',len(all_group_w),all_group_w)
            # reset
            tmp_base_feature = []

            np.savetxt( dir_train + d + '/feature/' + 'obj_4emo.csv', all_group_w, delimiter=",")