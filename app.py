import streamlit as st
import pandas as pd
import pickle
from transformers import pipeline
from sklearn.utils import shuffle

# Load the cleaned movie dataset
with open('new_df.pkl', 'rb') as f:
    new_df = pickle.load(f)

# Load the emotion classification model
classifier = pipeline("text-classification", 
                      model="j-hartmann/emotion-english-distilroberta-base", 
                      top_k=None)

def map_sentiment_to_genres(label):
    label = label.upper()
    mood_to_genres = {
        "JOY": ["comedy", "adventure", "family", "animation", "fantasy"],
        "SADNESS": ["drama", "romance", "tragedy", "emotional"],
        "ANGER": ["action", "war", "revenge", "thriller"],
        "FEAR": ["horror", "thriller", "mystery"],
        "SURPRISE": ["fantasy", "mystery", "sci-fi"],
        "DISGUST": ["crime", "psychological"],
        "NEUTRAL": ["documentary", "history"],
    }
    return mood_to_genres.get(label, ["drama"])  # Default fallback

def recommend_movies_based_on_mood(user_input):
    # Get list of mood predictions
    results = classifier(user_input)

    if isinstance(results[0], list):
        results = results[0]  # multi-label list

    # Sort by confidence
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    top_result = results[0]
    label = top_result['label'].upper()

    st.markdown(f"### 🧠 Detected Emotion: `{label}` ({top_result['score']:.2f})")

    mood_genres = map_sentiment_to_genres(label)
    st.markdown(f"### 🎭 Target Genres: {', '.join(mood_genres)}")

    # Filter movies
    filtered_df = new_df[new_df['genres'].apply(
        lambda genres: isinstance(genres, list) and any(g in genres for g in mood_genres))]

    if filtered_df.empty:
        st.warning("No matching movies found for this mood. Try a different emotion or expand genre mapping.")
        return pd.DataFrame(columns=["title", "vote_average"])

    # Return top 5 by vote_average
    recommendations = filtered_df.sort_values(by='vote_average', ascending=False)[['title', 'vote_average']].head(5)
    return recommendations

# ------------------- Streamlit UI -------------------
st.title("🎬 Emotion-Based Movie Recommender")
st.write("Tell us how you're feeling, and we'll suggest movies that match your mood!")

user_input = st.text_input("How are you feeling today?")

if user_input:
    recommendations = recommend_movies_based_on_mood(user_input)
    if not recommendations.empty:
        st.markdown("## 🎥 Top Movie Recommendations:")
        st.dataframe(recommendations.reset_index(drop=True))
