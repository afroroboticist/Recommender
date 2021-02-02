import re
import pandas as pd 
import numpy as np

def place_score(value):
  for idx in range(len(df)):
    if (df.iloc[idx,0] > value) and (df.iloc[idx+1,0] < value):
      return idx
    elif idx == (len(df)-2): return None
  return None

all_genres=[]
def prune_genre(data):
    genre_list=[]
    pattern = 'name\': \'[a-zA-Z]+\''
    found = re.findall(pattern, data)
    if found:
        for i in range(len(found)):
            genre_list.append(found[i][8:-1])
            if found[i][8:-1] not in all_genres:
                all_genres.append(found[i][8:-1])
    else:
        return []
    return genre_list

all_companies = []
def prune_production_companies(data):
    company_list=[]
    pattern ='name\': \'[a-zA-Z\s_]+'
    found = re.findall(pattern, data)
    if found:
        for i in range(len(found)):
            company_list.append(found[i][8:-1])
            if found[i][8:-1] not in all_companies:
                all_companies.append(found[i][8:-1])
    else:
        return []
    return company_list




def wrangle(data):
    movie_data = data.copy()
    movie_data['vote_count'].fillna(value=movie_data['vote_count'].mean(), inplace=True)
    movie_data['vote_average'].fillna(value=movie_data['vote_average'].mean(), inplace=True)
    movie_data['production_countries'].fillna(method='ffill', inplace=True)
    movie_data['country'] = movie_data['production_countries'].apply(lambda X: X[17:19])
    movie_data.drop(columns='production_countries', inplace=True)
    movie_data['spoken_languages'].fillna(method='ffill',inplace=True)
    movie_data['genre_list'] = movie_data['genres'].apply(prune_genre)
    movie_data['production_companies'].fillna(method='backfill', inplace=True)
    movie_data.drop(columns=['production_companies'], inplace=True)
    for genre in all_genres:
        movie_data[genre] = 0
    for i in range(len(movie_data['genre_list'])):
        current_genre_list = movie_data.loc[i,'genre_list']
        for item in current_genre_list:
            movie_data.loc[i,item] = 1
    condition = (movie_data['adult'] != 'True') & (movie_data['adult'] != 'False')
    movie_data.loc[condition, 'adult'] = 'False'
    mean_vote = movie_data['vote_average'].mean()
    chosen_quantile = movie_data['vote_count'].quantile(0.85)
    
    ############################
    # Weighted Rating Function #
    ############################

    def weighted_rating(data, m=chosen_quantile, C=mean_vote):
        v = data['vote_count']
        R = data['vote_average']
        return (v/(v+m) * R) + (m/(m+v) * C)    
    chosen_data = movie_data.copy().loc[movie_data['vote_count'] >= chosen_quantile]
    chosen_data['score'] = chosen_data.apply(weighted_rating, axis=1)
    chosen_data.drop_duplicates(subset=['id'], inplace=True)
    columns_to_drop = ['imdb_id','original_title','overview','poster_path',
        'genres','homepage','belongs_to_collection','release_date','spoken_languages','tagline','title']
    final_dataset = chosen_data.drop(columns=columns_to_drop)
    final_dataset.index= final_dataset['id']
    final_dataset['status'].fillna(value='Released', inplace=True)
    final_dataset.drop(columns='id',inplace=True)
    def treat_budget(data):
        if data.isnumeric():
            return float(data)
        else:
            print(data)
            return 0.0
    final_dataset['budget'] = final_dataset['budget'].apply(treat_budget)







