import streamlit as st
from content_based_filtering import recommend
from scipy.sparse import load_npz
import pandas as pd

# Load paths
transformed_data_path = "data/transformed_data.npz"
cleaned_data_path = "data/cleaned_data.csv"

# Load data
data = pd.read_csv(cleaned_data_path)
transformed_data = load_npz(transformed_data_path)

# Set page config
st.set_page_config(page_title="Spotify Recommender", page_icon="🎧", layout="centered")

st.markdown("""
    <style>
        /* General body and container styles */
        .block-container {
            padding-top: 2rem;
            background-color: #121212;
            color: #FFFFFF;
        }

        /* Title */
        h1 {
            color: #1DB954;
        }

        /* Input fields styling */
        .stTextInput > div > div > input {
            background-color: #1e1e1e;
            color: #FFFFFF;
            border: 1px solid #333333;
            border-radius: 8px;
            padding: 0.5rem;
        }

        /* Select box styling */
        .stSelectbox > div > div > div {
            background-color: #1e1e1e !important;
            color: #FFFFFF !important;
            border-radius: 8px;
        }

        /* Form styling */
        .stForm {
            border: 1px solid #333333;
            padding: 25px;
            border-radius: 12px;
            background-color: #181818;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }

        /* Button styling */
        button[kind="primary"] {
            background-color: #1DB954 !important;
            color: white !important;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            transition: background-color 0.3s ease;
        }

        button[kind="primary"]:hover {
            background-color: #1ed760 !important;
        }

        /* Divider line */
        hr {
            border-top: 1px solid #2a2a2a;
        }

        /* Markdown audio block spacing */
        audio {
            margin-top: 10px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)


# Main title
st.markdown("<h1 style='text-align: center; color: #1DB954;'>🎧 Spotify Song Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Discover songs similar to your favorites</p>", unsafe_allow_html=True)

# Input section
with st.form("recommendation_form"):
    st.markdown("### 🔍 Search for a Song")
    song_name = st.text_input("Enter the name of a song:")
    k = st.select_slider("Number of recommendations", options=[5, 10, 15, 20], value=10)
    submit_button = st.form_submit_button(label="Get Recommendations")

# When button clicked
if submit_button:
    if song_name.strip() == "":
        st.warning("Please enter a song name.")
    else:
        try:
            recommendations = recommend(song_name.lower(), data, transformed_data, k)

            if recommendations.empty:
                st.warning("No matching song found. Please check your spelling.")
            else:
                st.success(f"Recommendations for **{song_name.title()}**")
                
                for ind, recommendation in recommendations.iterrows():
                    song_display = recommendation['name'].title()
                    artist_display = recommendation['artist'].title()
                    preview_url = recommendation['spotify_preview_url']

                    if ind == 0:
                        st.markdown(f"## 🔊 Currently Playing: **{song_display}** by **{artist_display}**")
                        st.audio(preview_url)
                        st.write("---")
                    else:
                        col1, col2 = st.columns([0.1, 0.9])
                        with col1:
                            st.markdown(f"### {ind}")
                        with col2:
                            st.markdown(f"**{song_display}** by **{artist_display}**")
                            st.audio(preview_url)
                        st.write("---")

        except Exception as e:
            st.error(f"Error: {str(e)}")
