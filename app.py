import sys
sys.setrecursionlimit(500)
#########################################################
# Set the recursion limit because code kept maxing out  #
# when running on Heroku. Error was a recursion limit   #
# This fixed the problem. PLEASE DO NOT TOUCH           #
#########################################################
import pandas as pd 
import numpy as np 
import pickle
from flask import Flask, redirect, url_for, render_template
import os
from forms.forms import MovieChoice
from forms import forms
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from category_encoders import OrdinalEncoder, OneHotEncoder

app = Flask(__name__)
# Set manually or include secret key in your environment variables #
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY")#'mysecretkey'

# The list length variable determines how many movies +/- across your index should be returned #
LIST_LENGTH = 5
# Definition of pre trained models #
model_names = ['linear_regressor.mod','random_forest.mod','gradient_boost.mod']
MODELS = ['Linear Regressor','Random Forest Regressor','Gradient Boost Regressor']

build_models = []

# Download pretrained models and store them up in a list #
for models in model_names:
	model = pickle.load(open('static/'+models, 'rb'))
	build_models.append(model)
# Imported our data pipeline used in build_model.py #
PIPELINE = pickle.load(open('static/data_pipeline.lin','rb'))

CHOICES = []

# Read pre-wrangled data set #

df = pd.read_csv('static/movie_train.csv')

# Catalogue refers to a list of original movies with their untouched ratings to pick #
# Movies similar to our own movies from #

catalogue = pd.read_csv('static/catalogue_final.csv')
#########################################################
# This function is responsible for taking an observation#
# And converting it using the data pipeline into a      #
# form that can be understood by the model. Basically   #
# We're replicating what we did during training here    #
#########################################################

def prepare_data(data, model=None):
	X_prepd = PIPELINE.transform(data)
	print(X_prepd)
	return X_prepd

##################################################################
# The place score function is responsible for taking a predicted #
# Rating of a movie and searching through movie ratings in the   #
# Catalogue to find what movies best match the predicted rating  #
##################################################################


def place_score(catalogue, value):
  for idx in range(len(catalogue)):
    if (catalogue.iloc[idx,0] < value) and (catalogue.iloc[idx+1,0] > value):
      return idx
    elif idx == (len(catalogue)-2): return None
  return None

################################################
# All app routes are for the 3 major web pages #
################################################


@app.route('/')
def index():
	return redirect(url_for('home'))

@app.route('/home')
def home():
	return render_template('home.html')


@app.route('/pick_movie', methods=['GET','POST'])
def pick_movie():
	form = MovieChoice()
	global CHOICES
	if form.validate_on_submit():
		movie_choice = form.movie_choice.data
		model_choice = form.model_choice.data
		CHOICES=[(movie_choice, model_choice)]
		return redirect(url_for('display_choices'))

	return render_template('pick_movie.html', form=form)



@app.route('/display_choices')
def display_choices():
	recommended_movies = []
	charts = ['static/IMG/linear_regressor.png','static/IMG/random_forest.png'
	,'static/IMG/gradient_boost.png']
	print(CHOICES[0][1])
	if CHOICES[0][1] == 'Linear Regressor':
		model_choice = build_models[0]
		chart_choice = charts[0]
	elif CHOICES[0][1] == 'Random Forest Regressor':
		model_choice = build_models[1]
		chart_choice = charts[1]

	else:
		model_choice = build_models[2]
		chart_choice = charts[2]
	##############################
	# Extract ID of movie choice #
	##############################
	X_id = catalogue.loc[catalogue['original_title'] == CHOICES[0][0], 'id']
	print(X_id.iloc[0])
	#id = X_id['id']
	#print(id)
	# Collect Data to make prediction from training Data set #
	train_data = df[df['id'] == X_id.iloc[0]].drop(columns='Unnamed: 0')
	print(train_data.columns)
	# Pass Row of interest through data pipe line for consistency #
	prep_data = prepare_data(data=train_data)
	# Predict rating of movie of choice #
	predicted_value = model_choice.predict(prep_data)
	print('Predicted Value is: ', predicted_value)
	# Find index movie would occupy if it were inserted in the catalogue #
	idx = place_score(catalogue[['score']], predicted_value)
	# Set Beginning Index for Recommended Movies off of Catalogue #
	start_idx = idx - LIST_LENGTH
	if start_idx < 0:
		start_idx = 0
	end_idx = idx + LIST_LENGTH
	if end_idx > len(catalogue):
		end_idx = len(catalogue)

	for index in range(start_idx, end_idx):
		recommended_movies.append(catalogue.iloc[index, 4])
	print(recommended_movies)


	#print('Index occupied by value: ',idx)


	#print(type(model_choice))
	return render_template('display_choices.html', data=(CHOICES[0][0], recommended_movies, chart_choice))


@app.route('/about')
def about_page():
	return render_template('about.html')


if __name__ == "__main__":
	app.run(debug=True)
