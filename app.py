import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import pandas as pd

url = 'https://raw.githubusercontent.com/kesavan100/movie-rec/main/Tamil_movies_dataset.csv'
movies_df = pd.read_csv(url)

print(movies_df.head())




# Define features and target
X = movies_df[['Genre', 'Year']]
y = movies_df['Rating']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing and pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('genre', OneHotEncoder(handle_unknown='ignore'), ['Genre']),
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Recommendation function
def recommend_movies(genre, min_rating, year, df=movies_df):
    ratings_to_predict = df[['Genre', 'Year']].copy()
    ratings_to_predict['PredictedRating'] = pipeline.predict(ratings_to_predict)

    recommendations = ratings_to_predict[
        (ratings_to_predict['Genre'].str.contains(genre, case=False)) &
        (ratings_to_predict['PredictedRating'] >= min_rating) &
        (df['Year'] >= year)
    ]

    recommendations = recommendations.merge(df[['MovieName', 'Genre', 'Year']], on=['Genre', 'Year'], how='left')
    recommendations = recommendations.sort_values(by='PredictedRating', ascending=False)
    return recommendations[['MovieName', 'Genre', 'PredictedRating', 'Year']].reset_index(drop=True)

# Streamlit App
st.title("Tamil Movie Recommendation System 🎥")

# Inputs
genre = st.text_input("Enter a genre (e.g., Crime, Action, Drama, Comedy):")
min_rating = st.slider("Select minimum rating (0-10):", min_value=0.0, max_value=10.0, step=0.1, value=5.0)
year = st.number_input("Enter the year:", min_value=1900, max_value=2100, value=2000, step=1)

# Recommendation button
if st.button("Get Recommendations"):
    if genre.strip():
        recommended_movies = recommend_movies(genre, min_rating, year)
        if not recommended_movies.empty:
            st.write("### Recommended Movies:")
            st.dataframe(recommended_movies)
        else:
            st.warning("No movies found matching your criteria.")
    else:
        st.error("Please enter a valid genre.")
