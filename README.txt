**跑code的順序

(**1.~8.每增加raw資料就要跑一次)
# 處理raw data資料
1. split_data_info_BD.py
2. split_data_HC.py

# 分配raw data(依照量表時間點的前10天)

3. get_train_data_BD.py
4. get_train_data_HC.py

# 跑feature
5. feature_info_BD_HC.py

# 之後load_data的時候要使用
7. count_attender_in_each_type.py
8. cal_max_val_in_each_dim.py

# run linear model(後面可解釋性(畫圖)的 code 參考就好)
9. Uni_Lasso_YMRS.ipynb 
10. Uni_Lasso_HAMD.ipynb
