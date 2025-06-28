import streamlit as st
import pickle
from transformers import pipeline
from sklearn.utils import shuffle

# Load preprocessed movie data
with open('movie_data.pkl', 'rb') as f:
    movie_df = pickle.load(f)

# Load sentiment classifier from HuggingFace
classifier = pipeline("sentiment-analysis")

# Map sentiment to genre-related keywords
def map_sentiment_to_genres(label):
    if label == "POSITIVE":
        return ["comedy", "adventure", "family", "fantasy", "animation"]
    elif label == "NEGATIVE":
        return ["romance", "drama", "tragedy", "emotional", "history"]
    else:
        return ["thriller", "mystery", "crime", "documentary"]

# Recommend movies based on detected sentiment
def recommend_movies(user_input):
    result = classifier(user_input)[0]
    sentiment = result['label']
    confidence = result['score']

    st.write(f"🧠 Detected Sentiment: **{sentiment}** (Confidence: `{confidence:.2f}`)")

    mood_genres = [g.lower() for g in map_sentiment_to_genres(sentiment)]

    filtered_df = movie_df[movie_df['tags'].apply(lambda tags: any(genre in tags for genre in mood_genres))]

    if filtered_df.empty:
        st.warning("😔 No matching movies found. Try different mood wording.")
        return None

    recommendations = shuffle(filtered_df[['title']]).head(5)
    return recommendations

# Streamlit UI
st.set_page_config(page_title="Mood-Based Movie Recommender 🎬", layout="centered")
st.title("🎭 Mood-Based Movie Recommender")
st.markdown("Get personalized movie recommendations based on your current **emotions**!")

user_input = st.text_input("💬 How are you feeling today?")

if st.button("🎬 Recommend Movies") and user_input:
    recommendations = recommend_movies(user_input)
    if recommendations is not None:
        st.subheader("📽️ Top Picks for Your Mood:")
        for idx, row in recommendations.iterrows():
            st.markdown(f"✅ **{row['title']}**")
