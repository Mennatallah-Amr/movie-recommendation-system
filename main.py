# main.py

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# -------------------------
# Load Data
# -------------------------
ratings = pd.read_csv("u.data", sep="\t", names=["user_id","item_id","rating","timestamp"])
movies = pd.read_csv(
    "u.item",
    sep="|",
    names=["item_id","title"]+[str(i) for i in range(22)],
    encoding="latin-1"
)

data = pd.merge(ratings, movies[["item_id","title"]], on="item_id")

# Train/Test split
train, test = train_test_split(data, test_size=0.2, random_state=42)

train_matrix = train.pivot_table(index="user_id", columns="title", values="rating")
test_matrix  = test.pivot_table(index="user_id", columns="title", values="rating")

# Fill NaN with 0 for similarity
train_matrix_filled = train_matrix.fillna(0)

user_similarity = pd.DataFrame(
    cosine_similarity(train_matrix_filled),
    index=train_matrix_filled.index,
    columns=train_matrix_filled.index
)

# -------------------------
# Functions
# -------------------------
def recommend_movies(user_id, user_item_matrix, user_similarity, top_n=5):
    sim_scores = user_similarity.loc[user_id].drop(user_id)
    sim_scores = sim_scores.reindex(user_item_matrix.index).fillna(0)

    recommendation_scores = user_item_matrix.fillna(0).T.dot(sim_scores)

    watched = user_item_matrix.loc[user_id]
    recommendation_scores = recommendation_scores[watched.isna()]

    return recommendation_scores.sort_values(ascending=False).head(top_n)


def precision_at_k(user_id, train_matrix, test_matrix, user_similarity, k=5, threshold=3.5):
    top_recs = recommend_movies(user_id, train_matrix, user_similarity, top_n=k)
    recommended_movies = set(top_recs.index)

    if user_id not in test_matrix.index:
        return None

    user_ratings_test = test_matrix.loc[user_id]
    relevant_movies = set(user_ratings_test[user_ratings_test >= threshold].dropna().index)

    if len(recommended_movies) == 0:
        return 0

    hits = len(recommended_movies & relevant_movies)
    return (hits / k)*100


# -------------------------
# Streamlit UI
# -------------------------
st.title("🎬 Movie Recommendation System")

# User selection
user_id = st.number_input("Enter a User ID", min_value=1, max_value=int(train["user_id"].max()), value=1)

if st.button("Get Recommendations"):
    recs = recommend_movies(user_id, train_matrix, user_similarity, top_n=5)
    recs_df = recs.reset_index()
    recs_df.columns = ["title", "score"]
    st.subheader(f"Top 5 Recommendations for User {user_id}:")
    st.table(recs_df)


if st.button("Evaluate Precision@5"):
    precision = precision_at_k(user_id, train_matrix, test_matrix, user_similarity, k=5, threshold=3.5)
    st.subheader(f"Precision@5 for User {user_id}: {precision}%")
