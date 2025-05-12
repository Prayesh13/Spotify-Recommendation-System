from flask import Flask, render_template, request
from scipy.sparse import load_npz
import pandas as pd
from content_based_filtering import recommend

app = Flask(__name__)

# Load data
data = pd.read_csv("data/cleaned_data.csv")
transformed_data = load_npz("data/transformed_data.npz")

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    song_name = ''
    k = 10
    error_message = ''

    if request.method == 'POST':
        song_name = request.form.get('song_name', '').strip()
        k = int(request.form.get('num_recs', 10))
        if not song_name:
            error_message = "Please enter a song name."
        else:
            try:
                recommendations = recommend(song_name.lower(), data, transformed_data, k)
                if recommendations.empty:
                    error_message = "No matching song found. Please check your spelling."
            except Exception as e:
                error_message = str(e)

    return render_template("index.html", song_name=song_name, recommendations=recommendations, error=error_message)

if __name__ == '__main__':
    app.run(debug=True)
