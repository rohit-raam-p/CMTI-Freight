import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# User-item interaction matrix
R = np.array([
    [4, 0, 3, 5],
    [5, 4, 0, 0],
    [0, 0, 5, 4],
    [1, 5, 4, 0]
])

# Item feature matrix
F = np.array([
    [1, 0, 1],  # Item 1
    [0, 1, 1],  # Item 2
    [1, 1, 0],  # Item 3
    [0, 0, 1]   # Item 4
])

# Content-based scores
def content_based_scores(F, user_profile):
    return cosine_similarity(F, user_profile.reshape(1, -1)).flatten()

# Collaborative filtering scores
def collaborative_scores(R, user_index):
    user_similarity = cosine_similarity(R)
    scores = np.dot(user_similarity[user_index], R)
    return scores

# Hybrid recommendation
def hybrid_recommendation(R, F, user_index, alpha=0.5, top_n=2):
    user_ratings = R[user_index]
    user_profile = np.dot(user_ratings, F) / np.sum(user_ratings)

    cb_scores = content_based_scores(F, user_profile)
    cf_scores = collaborative_scores(R, user_index)

    hybrid_scores = alpha * cb_scores + (1 - alpha) * cf_scores
    recommended_items = np.argsort(hybrid_scores)[-top_n:][::-1]

    return recommended_items

# Generate recommendations
recommendations = hybrid_recommendation(R, F, user_index=0, top_n=2)
print("Hybrid Recommendations for User 1:", recommendations)
