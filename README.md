# Lasso-Multilayer perceptron
- The training and testing data are not included here.
- If you have any questions about the code, please email: g192e1654k@gmail.com

# Result
- ![image](https://github.com/Evanston0624/Lasso-MLP/edit/main/result/MAE-1.png)
- ![image](https://github.com/Evanston0624/Lasso-MLP/edit/main/result/MAE-2.png)

# Introduction

- This github using the variables selected by Lasso Regression, let the MLP performs a reduction of parameters to reduce the impact of overfitting. Before each input feature enters the model, it is multiplied by the corresponding weight \beta, and if \beta is 0, it means that the feature dimension is screened out, thereby reducing the amount of model parameters.

$$ y=g(\sum_{j=1}^q w{^0_j} f ((\sum_{i=1}^p \beta{i} w{_ij} x{_j})+ b{^h_j} ) b{^0}) $$


# Run the program

## Setting
- input = ./matched_audio/, ./matched_video/
- output = ./predict_audio_from_vid.csv, ./predict_audio.csv
- configuration = ./requirement.txt, ./conda_list.txt, ./package_version.txt

## run steps
- Every time you add raw data, you need to run steps 1~5
- step1. Data preprocessing
-  split_data_info_BD.py
-  split_data_HC.py

- step2. Allocate raw data (according to the first 10 days of the scale time point)
-  get_train_data_BD.py
-  get_train_data_HC.py

- step3. Feature extraction
-  feature_info_BD_HC.py

- step4. Calculate the number of subjects
-  count_attender_in_each_type.py

- step5. Data normalization
-  cal_max_val_in_each_dim.py

- step6. Run linear model
-  Uni_Lasso_YMRS.ipynb 
-  Uni_Lasso_HAMD.ipynb