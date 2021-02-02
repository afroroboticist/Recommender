import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from category_encoders import OrdinalEncoder, OneHotEncoder
from sklearn.inspection import permutation_importance
from pdpbox.pdp import pdp_interact, pdp_interact_plot, pdp_isolate, pdp_plot

try:
	sys.path.append('/home/afroroboticist/.virtualenvs/movieRec-GPy9DBCD')

except:
	print('Error encountered')

def convert_popularity(data):
    try:
        return float(data)
    except ValueError:
        print(data)
        return 0

df = pd.read_csv('static/movie_data_training_set.csv')
movie_data = pd.read_csv('static/movies_metadata.csv')

condition = df['budget'] == 0
df.loc[condition, 'budget'] = df['budget'].mean()
df.drop(columns=['genre_list','pro_co'])
df['status'] = df['status'].fillna(value='Released')
df['popularity'] = df['popularity'].apply(convert_popularity)
df['popularity'] = df['popularity'].fillna(method='ffill')
df['video'] = df['video'].fillna(method='ffill')
df['country'] = df['country'].fillna(value='US')
categorical = [col for col in df.select_dtypes('object').columns]
numerical = []
for columns in df.columns:
    if columns in categorical:
        pass
    else:
        numerical.append(columns)
items_to_remove = ['score','vote_average','vote_count']
for item in items_to_remove:
    numerical.remove(item)

y = df['score']
# display(y)
X = df.drop(columns=['score','vote_average','vote_count'])
condition = (movie_data['id'].isin(df['id']))
catalogue = movie_data[condition==True]
catalogue_df = catalogue[['id','imdb_id','original_title','poster_path']].drop_duplicates(subset=['id'])
catalogue_final = catalogue_df.merge(y, on=catalogue_df.index).sort_values(by='score')


prep_numericals = Pipeline([
    ('simpleimputer', SimpleImputer()),
    ('standardscaler', StandardScaler())
])

prep_categoricals = Pipeline([
    ('onehotencoder', OneHotEncoder()),
    
])

full_pipeline = ColumnTransformer([
    ('num', prep_numericals, numerical),
    ('cat', OrdinalEncoder(), categorical)
    ])

second_pipeline = ColumnTransformer([
    ('num', prep_numericals, numerical),
    ('cat', OneHotEncoder(), categorical)
    ])

movie_data_prepared = full_pipeline.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(movie_data_prepared, 
	y, train_size=0.8, random_state=10)


model_lr = LinearRegression()
#model_lr_ohe = LinearRegression()
model_dtr = DecisionTreeRegressor()
y_train_mean = [y_train.mean()] * len(y_train)
baseline = mean_squared_error(y_train, y_train_mean, squared=False)

cv_results = cross_validate(model_lr, X_train, y_train, cv=3)
scores = cross_val_score(model_dtr, X_train, y_train,scoring="neg_mean_squared_error", cv=5)
tree_rmse_scores = np.sqrt(-scores)
print('Cross Validated Decision Tree Scores: ',tree_rmse_scores)
print('Mean Score for Decision Tree: ',tree_rmse_scores.mean())
print('Std for Scores for Decision Tree: ', tree_rmse_scores.std())
y_pred_val = model_lr.predict(X_val)
rmse = mean_squared_error(y_val, y_pred_val, squared=False)
zipped = list(zip(y_val,y_pred_val))
model_rfr = RandomForestRegressor()

rfr_scores = cross_val_score(model_rfr, X_train, y_train,scoring="neg_mean_squared_error", cv=5)
rfr_rmse = np.sqrt(-rfr_scores)
print('Cross Validated Decision Tree Scores: ',rfr_rmse)
print('Mean Score for Decision Tree: ',rfr_rmse.mean())
print('Std for Scores for Decision Tree: ', rfr_rmse.std())

model_gbrt = GradientBoostingRegressor(max_depth=3, n_estimators=250, learning_rate=0.1)
model_gbrt.fit(X_train, y_train)

errors = [mean_squared_error(y_val, y_pred) for y_pred in model_gbrt.staged_predict(X_val)]
best_n_estimators = np.argmin(errors) + 1

gbrt_scores = cross_val_score(model_gbrt, X_train, y_train,scoring="neg_mean_squared_error", cv=5)
gbrt_rmse = np.sqrt(-gbrt_scores)
print('Cross Validated Gradient Boost Scores: ',gbrt_rmse)
print('Mean Score for Gradient Boost : ',gbrt_rmse.mean())
print('Std for Scores for Gradient Boost Scores: ', gbrt_rmse.std())



perm_imp = permutation_importance(model_gbrt, X_val, y_val, n_repeats=5, n_jobs=-1, random_state=42)
#print(perm_imp)
data = {'importances_mean': perm_imp['importances_mean'], 
        'importances_std': perm_imp['importances_std']}

perm_importances = pd.DataFrame(data, index=X.columns)
#print(perm_importances)
perm_importances_sorted = perm_importances.sort_values(by='importances_mean')
display(perm_importances_sorted)
#perm_importances_sorted.tail(20)

feature = 'adult'
isolate = pdp_isolate(model=model_gbrt,
                     dataset=pd.DataFrame(X_val,columns=X.columns),
                      model_features = X.columns,
                      feature = feature,
                      n_jobs=2
                     )
pdp_plot(isolate,feature_name=feature )

features = ['adult','popularity']

interact = pdp_interact(model=model_gbrt,
                     dataset=pd.DataFrame(X_val,columns=X.columns),
                      model_features = X.columns,
                      features = features,
                      n_jobs=2)

pdp_interact_plot(interact, plot_type='grid', feature_names=features )