import numpy as np
import pandas as pd

#Getting dataset
df=pd.read_csv('data.csv')
new_df=df.drop(['Unnamed: 0','game_season','knockout_match.1','match_event_id','location_x', 'location_y','home/away','match_id', 'team_id','team_name', 'date_of_game'],axis=1)
new_df.isnull().sum()

#Filling missing values
new_df['remaining_min'].fillna(df.remaining_min.mean(),inplace=True )
new_df['power_of_shot'].fillna(df.power_of_shot.mean(),inplace=True )
new_df['knockout_match'].fillna(df.knockout_match.mean(),inplace=True )
new_df['remaining_sec'].fillna(df.remaining_sec.mean(),inplace=True )
new_df['distance_of_shot'].fillna(df.distance_of_shot.mean(),inplace=True )
new_df['remaining_min.1'].fillna(df['remaining_min.1'].mean(),inplace=True )
new_df['power_of_shot.1'].fillna(df['power_of_shot.1'] .mean(),inplace=True )
#new_df['knockout_match.1'].fillna(df['knockout_match.1'].mean(),inplace=True )
new_df['knockout_match'].fillna(df.knockout_match.mean(),inplace=True )
new_df['remaining_sec.1'].fillna(df['remaining_sec.1'].mean(),inplace=True )
new_df['distance_of_shot.1'].fillna(df['distance_of_shot.1'] .mean(),inplace=True )

new_df.area_of_shot.fillna('Center(C)',inplace=True )
new_df.shot_basics.fillna('Mid Range',inplace=True )
new_df.range_of_shot.fillna('15 ft.',inplace=True )
new_df.type_of_shot.fillna('shot - 39',inplace=True )
new_df.type_of_combined_shot.fillna('shot - 3',inplace=True )
new_df['lat/lng'].fillna('42.982923, -71.446094',inplace=True )

#Encoding characters
from sklearn.preprocessing import LabelEncoder
lab_enc=LabelEncoder()
new_df.iloc[:,6]=lab_enc.fit_transform(new_df.iloc[:,6])
new_df.iloc[:,7]=lab_enc.fit_transform(new_df.iloc[:,7])
new_df.iloc[:,8]=lab_enc.fit_transform(new_df.iloc[:,8])
new_df.iloc[:,10]=lab_enc.fit_transform(new_df.iloc[:,10])
new_df.iloc[:,11]=lab_enc.fit_transform(new_df.iloc[:,11])
new_df.iloc[:,12]=lab_enc.fit_transform(new_df.iloc[:,12])

#splitting test and train
train = new_df.loc[df['is_goal'].notnull()]
X_train=train.loc[:, train.columns != 'is_goal']
X_train=X_train.drop(['shot_id_number'],axis=1)
Y_train=train[['is_goal']]
test = new_df.loc[df['is_goal'].isnull()]
X_test=test.loc[:, train.columns != 'is_goal']
X_test=X_test.drop(['shot_id_number'],axis=1)
Y_test=test[['is_goal']]

#Regularization
from sklearn.linear_model import LassoCV
reg = LassoCV()
reg.fit(X_train, Y_train)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X_train, Y_train))
coef = pd.Series(reg.coef_, index = X_train.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

imp_coef = coef.sort_values()
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")

#building model
'''
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,Y_train)
'''

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=80, max_features=3, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=5, min_samples_split=10,
            min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
clf.fit(X_train,Y_train)

# Create the parameter grid based on the results of random search 
from sklearn.model_selection import GridSearchCV
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X_train,Y_train)
grid_search.best_params_
best_grid = grid_search.best_estimator_


#predicting probability
Y_pred=clf.predict(X_test)
prob=clf.predict_proba(X_test)[:,1]
prediction = pd.DataFrame(prob, columns=['is_goal']).to_csv('prediction.csv')

