import pandas as pd
from flask_wtf import FlaskForm
from wtforms import IntegerField, SubmitField, StringField, SelectField


data = pd.read_csv('static/catalogue_final.csv')

MOVIE_CHOICES = list(data['original_title'].sort_values())
MODELS = ['Linear Regressor','Random Forest Regressor','Gradient Boost Regressor']

class MovieChoice(FlaskForm):
    movie_choice = SelectField(label='Favorite Movie', choices=MOVIE_CHOICES)
    model_choice = SelectField(label='Model for Prediction', choices=MODELS)
    submit = SubmitField('SUBMIT')



   