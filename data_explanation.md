# How the Data Is Obtained in FA2

## Short answer

The dataset is stored in the FA2 folder as **`spotify_tracks.csv`**, and the notebook
reads it from there with `pd.read_csv`. It originally comes from Hugging Face. If the
CSV file is ever missing, the notebook downloads the data from Hugging Face and saves
the CSV again, so it always works.

```python
DATA_FILE = "spotify_tracks.csv"

if os.path.exists(DATA_FILE):
    raw = pd.read_csv(DATA_FILE)
else:
    raw = load_dataset("maharshipandya/spotify-tracks-dataset", split="train").to_pandas()
    raw = raw.drop(columns=["Unnamed: 0"], errors="ignore")
    raw.to_csv(DATA_FILE, index=False)
```

The Streamlit app (`app.py`) doesn't use the dataset at all. It only loads the trained
models saved in `models/`.

---

## 1. Where the data comes from

| Item | Detail |
|---|---|
| Platform | Hugging Face Hub (an online repository of ML datasets and models) |
| Dataset name | `maharshipandya/spotify-tracks-dataset` |
| Link | https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset |
| Original source | Collected from the Spotify Web API (audio features of tracks) |
| Local copy | `spotify_tracks.csv` in the FA2 folder (19 MB) |
| Size | 114,000 rows (tracks), 125 genres, 1,000 tracks per genre |
| Columns | 20: `track_id`, `artists`, `album_name`, `track_name`, `popularity`, `duration_ms`, `explicit`, `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `time_signature`, `track_genre` |

---

## 2. How the loading code works, step by step

1. **`DATA_FILE = "spotify_tracks.csv"`**: the name of the local dataset file. It's a
   relative path, so it refers to the file in the same folder as the notebook.
2. **`os.path.exists(DATA_FILE)`**: checks whether the CSV is present.
3. **If it exists**: `pd.read_csv(DATA_FILE)` reads it into a pandas DataFrame. This is
   what normally happens. No internet is needed.
4. **If it doesn't exist**, the fallback runs:
   - `load_dataset("maharshipandya/spotify-tracks-dataset", split="train")` uses the
     Hugging Face `datasets` library to download the data. The string is the dataset's ID
     on Hugging Face (`username/dataset-name`).
   - `split="train"`: this dataset has only one split on Hugging Face, called `train`,
     which holds all 114,000 rows. It is **not** our training set; our own train/test
     split is made later with `train_test_split`.
   - `.to_pandas()` converts it into a pandas DataFrame.
   - `.drop(columns=["Unnamed: 0"])` removes an extra index column from the original
     file that has no meaning.
   - `raw.to_csv(DATA_FILE, index=False)` saves it as `spotify_tracks.csv`, so the next
     run reads the local file.

Either way, the result is the same DataFrame: `(114000, 20)`, meaning 114,000 rows and
20 columns.

---

## 3. How the CSV was created

The CSV was created by downloading the dataset once from Hugging Face with
`load_dataset(...)`, removing the `Unnamed: 0` index column, and saving it with
`to_csv`. This is exactly what the fallback code does. It was checked after saving:
reading the CSV back gives the same 114,000 rows, 20 columns and column types, and the
notebook gives identical results with the CSV as with the direct download.

---

## 4. Why store the data in the project folder

- **Self-contained:** the dataset is part of the project, so it's clear what data the
  models were trained on.
- **Works offline:** after the CSV exists, no internet connection is needed.
- **Faster and simpler:** `pd.read_csv` is a standard, familiar way to load data.
- **No broken paths:** the relative path `spotify_tracks.csv` works on any computer, as
  long as the notebook is run from the FA2 folder.
- **Safe fallback:** if the CSV isn't submitted or is deleted, the notebook gets the
  data from Hugging Face automatically.

---

## 5. What happens to the data after loading

The full 114,000 rows are reduced to a smaller dataset for the 5-genre problem
(section 3 of the notebook):

| Step | Code | Result |
|---|---|---|
| 1. Remove duplicate songs | `raw.drop_duplicates(subset="track_id", keep="first")` | 114,000 → 89,741 rows (24,259 removed) |
| 2. Keep only the 5 genres | `df[df["track_genre"].isin(GENRES)]` | 4,346 rows |
| 3. Sample up to 1,500 per genre | `g.sample(n=min(1500, len(g)), random_state=42)` | 4,346 rows (all kept) |

**Why remove duplicates?** The same song appears in the dataset under several genres
(for example, one track can be listed under both `edm` and `dance`). If duplicates
stayed, the same feature values could end up in both the training and test sets with
different labels, which would make evaluation unreliable.

**Why are there fewer than 1,500 per genre?** Each genre has only 1,000 tracks in the
original dataset, and removing duplicates leaves fewer:

| Genre | Tracks |
|---|---|
| heavy-metal | 997 |
| country | 946 |
| classical | 867 |
| hip-hop | 842 |
| edm | 694 |
| **Total** | **4,346** |

Because 1,500 isn't available, `min(1500, available)` keeps every track. The classes are
slightly unbalanced, so the notebook uses a **stratified** train/test split and
**macro-averaged** metrics (each genre counts equally).

**Missing values:** only 1 row has missing text fields (artist, album, track name).
None of the 10 audio features used for modelling have missing values, so no imputation
was needed.

---

## 6. Data flow summary

```
Hugging Face Hub (online)
        |
        |  downloaded once with load_dataset(...), saved with to_csv(...)
        v
spotify_tracks.csv in the FA2 folder   (114,000 rows x 20 columns, 19 MB)
        |
        |  pd.read_csv(...)
        v
raw DataFrame: 114,000 rows x 20 columns
        |
        |  drop duplicates -> filter 5 genres -> sample
        v
df: 4,346 rows (5 genres), 10 audio features used
        |
        |  train_test_split (80/20, stratified)
        v
Train models -> save to models/*.joblib
        |
        v
app.py loads models/*.joblib   (the app doesn't use the dataset)
```

---

## 7. Likely questions from the teacher

**Q: Where is your dataset?**
In the FA2 folder as `spotify_tracks.csv`. It comes from the Hugging Face dataset
`maharshipandya/spotify-tracks-dataset`, which was originally collected from the
Spotify Web API.

**Q: How did you get the CSV?**
I downloaded it from Hugging Face with the `datasets` library (`load_dataset`) and saved
it with `to_csv`. The notebook still contains that code as a fallback.

**Q: What if the CSV file is missing?**
The notebook detects that with `os.path.exists` and downloads the data from Hugging Face
instead, then saves the CSV again. That one run needs internet.

**Q: Is `split="train"` your training data?**
No. It's the name of the only split this dataset has on Hugging Face; it contains all
114,000 rows. Our own 80/20 train/test split is made later with `train_test_split`.

**Q: Does the Streamlit app need the dataset?**
No. The app loads the saved models and `metadata.joblib` (feature names, slider ranges,
genre averages, scores) from the `models/` folder. Everything it needs was computed by
the notebook.

**Q: Why not 1,500 tracks per genre as planned?**
The dataset has only 1,000 per genre, and fewer after removing duplicates. So all
available tracks (4,346 total) are used.
