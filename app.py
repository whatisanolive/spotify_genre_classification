"""Streamlit app: predict a track's genre from its audio features with every trained model.

Run from this folder:  streamlit run app.py
The models are created by spotify_genre_classification.ipynb (saved to models/).
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_DIR = Path(__file__).parent / "models"

# Slider settings per feature: (label, step, help text).
# Ranges come from the training data (metadata.joblib), so sliders never go outside what the models saw.
FEATURE_UI = {
    "danceability":     ("Danceability", 0.01, "How suitable the track is for dancing (0–1)"),
    "energy":           ("Energy", 0.01, "Perceived intensity and activity (0–1)"),
    "loudness":         ("Loudness (dB)", 0.1, "Average loudness in decibels"),
    "speechiness":      ("Speechiness", 0.01, "Presence of spoken words (0–1)"),
    "acousticness":     ("Acousticness", 0.01, "Confidence the track is acoustic (0–1)"),
    "instrumentalness": ("Instrumentalness", 0.01, "Likelihood the track has no vocals (0–1)"),
    "liveness":         ("Liveness", 0.01, "Likelihood of a live audience (0–1)"),
    "valence":          ("Valence", 0.01, "Musical positiveness (0–1)"),
    "tempo":            ("Tempo (BPM)", 1.0, "Estimated beats per minute"),
    "duration_ms":      ("Duration (seconds)", 1.0, "Track length"),  # slider in seconds, converted to ms
}


@st.cache_resource
def load_artifacts():
    meta = joblib.load(MODEL_DIR / "metadata.joblib")
    models = {name: joblib.load(MODEL_DIR / Path(path).name) for name, path in meta["model_files"].items()}
    finder_path = MODEL_DIR / "track_finder.joblib"
    finder = joblib.load(finder_path) if finder_path.exists() else None
    return meta, models, finder


def slider_for(feature, stats, value=None):
    label, step, help_text = FEATURE_UI[feature]
    lo, hi, default = stats["min"], stats["max"], stats["median"] if value is None else value
    if feature == "duration_ms":
        lo, hi, default = lo / 1000, min(hi / 1000, 900.0), default / 1000
    lo, hi = float(round(lo, 2)), float(round(hi, 2))
    default = float(min(max(round(default, 2), lo), hi))
    val = st.slider(label, lo, hi, default, step=step, help=help_text, key=feature)
    return val * 1000 if feature == "duration_ms" else val


st.set_page_config(page_title="Spotify Genre Classifier", page_icon="🎵", layout="wide")

if not (MODEL_DIR / "metadata.joblib").exists():
    st.error("No models found. Run `spotify_genre_classification.ipynb` first to train and save them to `models/`.")
    st.stop()

meta, models, finder = load_artifacts()
features, genres = meta["features"], meta["genres"]

st.title("🎵 Spotify Genre Classifier")
st.write(
    "Set the audio features with the sliders and every trained model predicts the genre: "
    + ", ".join(f"**{g}**" for g in genres) + "."
)

# Sidebar: sliders, optionally prefilled with a genre's average profile
with st.sidebar:
    st.header("Audio features")
    preset = st.selectbox("Start from a genre's average profile", ["Dataset median"] + genres)
    if st.session_state.get("_preset") != preset:
        # Changing the preset resets the sliders to that profile
        for f in features:
            st.session_state.pop(f, None)
        st.session_state["_preset"] = preset
    preset_values = meta["genre_means"][preset] if preset in genres else {}
    inputs = {f: slider_for(f, meta["feature_stats"][f], preset_values.get(f)) for f in features}

X = pd.DataFrame([inputs], columns=features)

# Predictions from every model
rows, probas = [], {}
for name, model in models.items():
    pred = model.predict(X)[0]
    row = {"Model": name, "Predicted genre": pred}
    if hasattr(model, "predict_proba"):
        p = pd.Series(model.predict_proba(X)[0], index=model.classes_)
        probas[name] = p
        row["Confidence"] = p[pred]
    row["Test accuracy"] = meta["results"][name]["Accuracy"]
    row["Test F1 (macro)"] = meta["results"][name]["F1"]
    rows.append(row)
pred_df = pd.DataFrame(rows).sort_values("Test F1 (macro)", ascending=False).reset_index(drop=True)

best_name = pred_df.loc[0, "Model"]
votes = pred_df["Predicted genre"].value_counts()

c1, c2, c3 = st.columns(3)
c1.metric(f"Best model ({best_name})", pred_df.loc[0, "Predicted genre"])
c2.metric("Majority vote", votes.index[0], f"{votes.iloc[0]}/{len(pred_df)} models", delta_color="off")
c3.metric("Models in agreement", "All" if len(votes) == 1 else f"{votes.iloc[0]} of {len(pred_df)}")

st.subheader("Predictions by model")
st.dataframe(
    pred_df.style.format({"Confidence": "{:.1%}", "Test accuracy": "{:.3f}", "Test F1 (macro)": "{:.3f}"}),
    width="stretch", hide_index=True,
)

st.subheader("Class probabilities")
proba_df = pd.DataFrame(probas).T[genres]
st.bar_chart(proba_df.T, horizontal=True, stack=False, height=380)
with st.expander("Probability table"):
    st.dataframe(proba_df.style.format("{:.1%}").highlight_max(axis=1, color="#b7e4c7"), width="stretch")

col_similar, col_top = st.columns(2)

with col_similar:
    st.subheader("Songs that sound like this")
    if finder is None:
        st.info("Re-run the notebook to create `models/track_finder.joblib`.")
    else:
        dist, idx = finder["nn"].kneighbors(finder["scaler"].transform(X[features]))
        similar = finder["tracks"].iloc[idx[0]][["track_name", "artists", "track_genre", "popularity"]]
        similar = similar.assign(distance=dist[0]).set_axis(
            ["Track", "Artist", "Genre", "Popularity", "Distance"], axis=1)
        st.dataframe(similar.style.format({"Distance": "{:.2f}"}), width="stretch", hide_index=True)
        st.caption("Real tracks with the closest audio features to the sliders (smaller distance = more similar).")

with col_top:
    st.subheader("Top songs by genre")
    best_pred = pred_df.loc[0, "Predicted genre"]
    top_genre = st.selectbox("Genre (defaults to the best model's prediction)", genres, index=genres.index(best_pred))
    if "top_tracks" not in meta:
        st.info("Re-run the notebook to save the top songs per genre.")
    else:
        n = st.slider("Number of songs", 5, len(meta["top_tracks"][top_genre]), 10)
        top = pd.DataFrame(meta["top_tracks"][top_genre]).head(n)
        top = top.set_axis(["Track", "Artist", "Album", "Popularity"], axis=1)
        top.index = range(1, len(top) + 1)
        st.dataframe(top, width="stretch")

with st.expander("Input sent to the models"):
    st.dataframe(X, hide_index=True)

st.caption(
    "Models: scikit-learn Pipelines (StandardScaler + classifier) trained on the Hugging Face "
    "maharshipandya/spotify-tracks-dataset. Random Forest (tuned) used GridSearchCV: "
    + ", ".join(f"{k.replace('clf__', '')}={v}" for k, v in meta["best_rf_params"].items())
)
