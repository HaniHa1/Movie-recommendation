#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install -r requirements.txt


# # Preprocess the data:

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
movies = pd.read_csv('data/ml-1m/movies.dat', sep='::', header=None, engine='python', names=['movieId', 'title', 'genres'])
ratings = pd.read_csv('data/ml-1m/ratings.dat', sep='::', header=None, engine='python', names=['userId', 'movieId', 'rating', 'timestamp'])
users = pd.read_csv('data/ml-1m/users.dat', sep='::', header=None, engine='python', names=['userId', 'gender', 'age', 'occupation', 'zip'])

# Preprocess movie metadata
movies['genres'] = movies['genres'].str.replace('|', ' ')

# Merge ratings and movies data
ratings = ratings.merge(movies[['movieId', 'title']], on='movieId')


# # Implement Collaborative Filtering:

# In[ ]:


# Create user-item matrix
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
matrix = csr_matrix(user_movie_matrix.values)

# Fit KNN model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model_knn.fit(matrix)

# Function to get movie recommendations based on user-item interactions
def collaborative_filtering(user_id, n_recommendations=10):
    user_index = user_movie_matrix.index.get_loc(user_id)
    distances, indices = model_knn.kneighbors(matrix[user_index], n_neighbors=n_recommendations+1)
    
    recommendations = []
    for i in range(1, len(distances.flatten())):
        idx = indices.flatten()[i]
        movie_id = user_movie_matrix.columns[idx]
        recommendations.append(movies[movies['movieId'] == movie_id]['title'].values[0])
    
    return recommendations


# # Implement Content-Based Filtering:

# In[ ]:


# Fit TF-IDF model
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations based on movie metadata
def content_based_filtering(movie_title, n_recommendations=10):
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n_recommendations+1]
    
    movie_indices = [i[0] for i in sim_scores]
    recommendations = movies['title'].iloc[movie_indices].tolist()
    
    return recommendations


# In[ ]:


def hybrid_recommendations(user_id, movie_title, n_recommendations=10):
    collab_recs = collaborative_filtering(user_id, n_recommendations)
    content_recs = content_based_filtering(movie_title, n_recommendations)
    
    # Combine recommendations
    combined_recs = list(set(collab_recs + content_recs))
    
    return combined_recs[:n_recommendations]


# # Create Visualizations

# In[ ]:


# Distribution of Ratings
plt.figure(figsize=(10, 6))
sns.countplot(ratings['rating'])
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# Most Rated Movies
most_rated = ratings.groupby('title').size().sort_values(ascending=False)[:10]
plt.figure(figsize=(10, 6))
sns.barplot(x=most_rated.values, y=most_rated.index, palette='viridis')
plt.title('Top 10 Most Rated Movies')
plt.xlabel('Number of Ratings')
plt.ylabel('Movie Title')
plt.show()


# # Example

# In[ ]:


user_id = 1
movie_title = 'Toy Story (1995)'

print("Collaborative Filtering Recommendations:")
print(collaborative_filtering(user_id))

print("\nContent-Based Filtering Recommendations:")
print(content_based_filtering(movie_title))

print("\nHybrid Recommendations:")
print(hybrid_recommendations(user_id, movie_title))


# # Future Enhancements
# 
Model Tuning: Use grid search or randomized search to tune hyperparameters for both collaborative and content-based models.
Neural Network Models: Implement neural network-based collaborative filtering using TensorFlow or Keras.
Large-Scale Data Processing: Use Apache Spark for handling larger datasets and distributed processing.
Advanced Evaluation: Implement more advanced evaluation metrics like precision, recall, F1-score, and AUC-ROC.