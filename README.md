# Recommender
A mini movie recommender that uses movie ratings as its target vector

BRIEF INTRODUCTION TO PROJECT:
The dataset used for building this recommender system was gotten from GroupLens.org
They are a team of researchers from the University of Minnesota.
The idea behind this system is that movies of similar ratings would hold similar interests for viewers. 
Basically you’re more likely to be interested in a movie if it rates similar to a movie you really enjoy. This has actually proven to be true in a general sense.

CHALLENGES:
As expected, wrangling the data was the biggest challenge faced. 
A borrowed concept of weighted scoring was used. Credit to IMDB. 
The idea behind weighted scoring is to ensure that if a movie is rated 6.0 by a thousand people, it is weighted heavier than a movie that’s rated 6.0 by ten people only. This is crucial to the effectiveness of this system.
The new scoring system was developed into a new feature and used as my target vector.
The movie genres were stored as JSON objects. As a result I had to utilize regEx functions to extract them from these objects. Each movie had a combination of several genres and resulted in me having to manually One-Hot-Encode them into my dataset.

For the purpose of this project, I have only made three models available to be tested in developing your movie recommendations. I have also provided (hopefully) some residual plots showing how well these models perform on the validation data sets before going on to make predictions.

Despite having laid out the project such that your already-wrangled data is available, the code contains all that you need to do the wrangling yourself. 

Storing of the completely trained models and data pipelines were achieved using the python pickle library. All pre-trained models are re-called in the app.py file prior to deployment of the web app.

You can find all of my code and datasets here: https://github.com/afroroboticist/Recommender
You can also find the original datasets used for the project here: https://grouplens.org/datasets/movielens/

I thoroughly enjoyed working on this and hopefully it serves as a platform for more work:
Possible in-roads to further research:
- Utilization of Cosine-Similarities to link movies with their summaries.
- Inclusion of production crew and cast in the movie ranking metrics
