import load_data
import feature_attender
# from sklearn.linear_model import LassoCV
# from sklearn import linear_model
# from sklearn.metrics import mean_absolute_error

dir_grp = 'HC'
feature_types = ['B_1','B_2','C','D','E']
attenders = feature_attender.get_wanna_attender(dir_grp,feature_types)

print(dir_grp,'attenders:',len(attenders),attenders)
x, label = load_data.loading_data(dir_grp,attenders,feature_types,target_scale='hamd')
print('res', x.shape, label.shape)

