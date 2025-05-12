import streamlit as st
from content_based_filtering import content_recommendation
from scipy.sparse import load_npz
import pandas as pd
from collaborative_filtering import collaborative_recommendation
from numpy import load

# Load datasets
songs_data = pd.read_csv("data/cleaned_data.csv")
transformed_data = load_npz("data/transformed_data.npz")
track_ids = load("data/track_ids.npy", allow_pickle=True)
filtered_data = pd.read_csv("data/collab_filtered_data.csv")
interaction_matrix = load_npz("data/interaction_matrix.npz")

# Streamlit Page Configuration
st.set_page_config(page_title="Spotify Recommender", page_icon="🎧", layout="centered")

# Custom CSS Styling
st.markdown("""
    <style>
        h1 { color: #1DB954; }
        .stTextInput > div > div > input {
            background-color: #1e1e1e;
            color: #FFFFFF;
            border: 1px solid #333333;
            border-radius: 8px;
            padding: 0.5rem;
        }
        .stSelectbox > div > div > div {
            background-color: #1e1e1e !important;
            color: #FFFFFF !important;
            border-radius: 8px;
        }
        .stForm {
            border: 2px solid #FF69B4;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            background-color: #181818;
            box-shadow: 0 4px 15px rgba(255, 105, 180, 0.3);
        }
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
        hr { border-top: 1px solid #2a2a2a; }
        audio {
            margin-top: 10px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Title & Subtitle
st.markdown("<h1 style='text-align: center;'>🎧 Spotify Song Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Discover songs similar to your favorites</p>", unsafe_allow_html=True)

# --- Input Form ---
with st.form("recommendation_form"):
    st.markdown("### 🔍 Search for a Song")
    song_name = st.text_input("Enter a song name:")
    artist_name = st.text_input("Enter the artist name:")
    k = st.select_slider("Number of recommendations", options=[5, 10, 15, 20], value=10)
    filtering_type = st.selectbox("Select the type of filtering:", ['Content-Based Filtering', 'Collaborative Filtering'])
    submit_button = st.form_submit_button(label="Get Recommendations")

# --- Recommendation Logic ---
if submit_button:
    song_name = song_name.lower().strip()
    artist_name = artist_name.lower().strip()

    if not song_name or not artist_name:
        st.warning("Please enter both song name and artist name.")
    else:
        try:
            if filtering_type == "Content-Based Filtering":
                if ((songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)).any():
                    st.success(f"Recommendations for **{song_name.title()}** by **{artist_name.title()}**")
                    recommendations = content_recommendation(song_name, songs_data, transformed_data, k)
                    # content_recommendation(song_name=song_name, songs_data=songs_data, transformed_data=transformed_data, k=k)
                else:
                    st.warning(f"❌ Couldn't find '{song_name}' by '{artist_name}' in the dataset.")
                    recommendations = None
            else:
                if ((filtered_data["name"] == song_name) & (filtered_data["artist"] == artist_name)).any():
                    st.success(f"Recommendations for **{song_name.title()}** by **{artist_name.title()}**")
                    recommendations = collaborative_recommendation(song_name, artist_name, track_ids, filtered_data, interaction_matrix, k)
                else:
                    st.warning(f"❌ Couldn't find '{song_name}' by '{artist_name}' in the dataset.")
                    recommendations = None

            # --- Display Results ---
            if recommendations is not None and not recommendations.empty:
                now_playing = recommendations.iloc[0]
                st.markdown("## 🎶 Now Playing")
                with st.container():
                    st.markdown(f"""
                        <div style='
                            border: 2px solid #1DB954;
                            border-radius: 12px;
                            padding: 20px;
                            margin: 15px 0;
                            background-color: #181818;
                            box-shadow: 0 4px 15px rgba(0, 255, 100, 0.3);
                        '>
                            <h3 style='color:#1DB954;margin-bottom:5px;'>🎵 {now_playing['name'].title()}</h3>
                            <p style='margin-top:0;margin-bottom:10px;'>by <strong>{now_playing['artist'].title()}</strong></p>
                    """, unsafe_allow_html=True)

                    if pd.notna(now_playing['spotify_preview_url']):
                        st.audio(now_playing['spotify_preview_url'])
                    else:
                        st.markdown("<p style='color:gray;'>No preview available</p>", unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

                # Additional Songs
                st.markdown("## 🎧 You Might Also Like")
                for idx, row in recommendations.iloc[1:].iterrows():
                    with st.container():
                        st.markdown(f"""
                            <div style='
                                border: 1px solid #333333;
                                border-radius: 12px;
                                padding: 20px;
                                margin: 15px 0;
                                background-color: #181818;
                                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                            '>
                                <h4 style='color:#1DB954;margin-bottom:5px;'>🎵 {row['name'].title()}</h4>
                                <p style='margin-top:0;margin-bottom:10px;'>by <strong>{row['artist'].title()}</strong></p>
                        """, unsafe_allow_html=True)

                        if pd.notna(row['spotify_preview_url']):
                            st.audio(row['spotify_preview_url'])
                        else:
                            st.markdown("<p style='color:gray;'>No preview available</p>", unsafe_allow_html=True)

                        st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")
