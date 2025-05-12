import pandas as pd
import dask.dataframe as dd
from scipy.sparse import csr_matrix, save_npz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# File Paths
TRACK_IDS_SAVE_PATH = "data/track_ids.npy"
FILTERED_DATA_SAVE_PATH = "data/collab_filtered_data.csv"
INTERACTION_MATRIX_SAVE_PATH = "data/interaction_matrix.npz"
SONGS_DATA_PATH = "data/cleaned_data.csv"
USER_HISTORY_PATH = "data/User_Listening_History.csv"


def save_dataframe(data: pd.DataFrame, path: str) -> None:
    """Save a pandas DataFrame to CSV."""
    data.to_csv(path, index=False)


def save_sparse_matrix(matrix: csr_matrix, path: str) -> None:
    """Save a SciPy sparse matrix in NPZ format."""
    save_npz(path, matrix)


def filter_songs(songs: pd.DataFrame, track_ids: list[str], save_path: str) -> pd.DataFrame:
    """Filter and save songs based on track IDs."""
    filtered = songs[songs["track_id"].isin(track_ids)].sort_values(by="track_id").reset_index(drop=True)
    save_dataframe(filtered, save_path)
    return filtered


def create_interaction_matrix(
    history: dd.DataFrame, track_ids_path: str, matrix_save_path: str
) -> csr_matrix:
    """Generate and save a sparse interaction matrix."""
    history = history.copy()
    history['playcount'] = history['playcount'].astype(np.float64)
    history = history.categorize(columns=['user_id', 'track_id'])

    user_codes = history['user_id'].cat.codes
    track_codes = history['track_id'].cat.codes
    track_ids = history['track_id'].cat.categories.values

    np.save(track_ids_path, track_ids, allow_pickle=True)

    history = history.assign(user_idx=user_codes, track_idx=track_codes)
    grouped = history.groupby(['track_idx', 'user_idx'])['playcount'].sum().reset_index().compute()

    matrix = csr_matrix(
        (grouped['playcount'], (grouped['track_idx'], grouped['user_idx']))
    )
    save_sparse_matrix(matrix, matrix_save_path)
    return matrix


def collaborative_recommendation(
    song_name: str,
    artist_name: str,
    track_ids: np.ndarray,
    songs_df: pd.DataFrame,
    interaction_matrix: csr_matrix,
    k: int = 5
) -> pd.DataFrame:
    """Recommend songs using collaborative filtering."""
    song_name, artist_name = song_name.lower(), artist_name.lower()
    match = songs_df[(songs_df["name"].str.lower() == song_name) &
                     (songs_df["artist"].str.lower() == artist_name)]

    if match.empty:
        raise ValueError("No matching song found for recommendation.")

    input_track_id = match['track_id'].values[0]
    index = np.where(track_ids == input_track_id)[0]

    if len(index) == 0:
        raise ValueError("Track ID not found in interaction matrix.")

    index = index[0]
    input_vector = interaction_matrix[index]
    similarity = cosine_similarity(input_vector, interaction_matrix).ravel()

    top_indices = np.argsort(similarity)[-k - 1:][::-1]
    top_track_ids = track_ids[top_indices]
    top_scores = similarity[top_indices]

    scores_df = pd.DataFrame({"track_id": top_track_ids, "score": top_scores})
    recommendations = (
        songs_df[songs_df["track_id"].isin(top_track_ids)]
        .merge(scores_df, on="track_id")
        .sort_values(by="score", ascending=False)
        .drop(columns=["track_id", "score"])
        .reset_index(drop=True)
    )
    return recommendations


def main() -> None:
    """Main execution function."""
    user_history = dd.read_csv(USER_HISTORY_PATH)
    unique_ids = user_history["track_id"].unique().compute().tolist()

    songs_df = pd.read_csv(SONGS_DATA_PATH)
    filtered_songs = filter_songs(songs_df, unique_ids, FILTERED_DATA_SAVE_PATH)

    create_interaction_matrix(user_history, TRACK_IDS_SAVE_PATH, INTERACTION_MATRIX_SAVE_PATH)


if __name__ == "__main__":
    main()
