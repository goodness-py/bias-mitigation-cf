import os
import pandas as pd
from surprise import Dataset

def get_movielens_path():
    """Return path to MovieLens-1M data directory after downloading if needed."""
    data = Dataset.load_builtin("ml-1m", prompt=False)
    data.build_full_trainset()
    base = os.path.expanduser("~/.surprise_data/ml-1m/ml-1m")
    return base

def load_ratings(data_dir):
    """Load ratings.dat → DataFrame with columns [user_id, item_id, rating, timestamp]."""
    path = os.path.join(data_dir, "ratings.dat")
    df = pd.read_csv(
        path,
        sep="::",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    df["rating"] = df["rating"].astype(float)
    df["timestamp"] = df["timestamp"].astype(int)
    return df


def load_users(data_dir):
    """Load users.dat → DataFrame with columns [user_id, gender, age, occupation, zip]."""
    path = os.path.join(data_dir, "users.dat")
    df = pd.read_csv(
        path,
        sep="::",
        header=None,
        names=["user_id", "gender", "age", "occupation", "zip"],
        engine="python",
    )
    df["user_id"] = df["user_id"].astype(int)
    return df


GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def load_items(data_dir):
    """Load movies.dat → DataFrame with [item_id, title, genres, + binary genre columns]."""
    path = os.path.join(data_dir, "movies.dat")
    df = pd.read_csv(
        path,
        sep="::",
        header=None,
        names=["item_id", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )
    df["item_id"] = df["item_id"].astype(int)
    # Expand pipe-separated genres into binary columns
    for genre in GENRES:
        df[genre] = df["genres"].str.contains(genre, regex=False).astype(int)
    return df


def build_gender_map(users_df):
    """Return dict {user_id: 'M'|'F'}."""
    return dict(zip(users_df["user_id"], users_df["gender"]))


def load_all():
    """
    Main entry point.
    Returns:
        ratings_df  : pd.DataFrame  [user_id, item_id, rating, timestamp]
        users_df    : pd.DataFrame  [user_id, age, gender, occupation, zip]
        items_df    : pd.DataFrame  [item_id, title, ...]
        gender_map  : dict          {user_id -> 'M'|'F'}
        data_dir    : str           path to ml-1m folder
    """
    data_dir = get_movielens_path()
    ratings_df = load_ratings(data_dir)
    users_df = load_users(data_dir)
    items_df = load_items(data_dir)
    gender_map = build_gender_map(users_df)
    return ratings_df, users_df, items_df, gender_map, data_dir


if __name__ == "__main__":
    ratings, users, items, gender_map, _ = load_all()
    print(f"Ratings : {len(ratings):,} rows")
    print(f"Users   : {len(users):,}  ({sum(v=='M' for v in gender_map.values())} M / {sum(v=='F' for v in gender_map.values())} F)")
    print(f"Items   : {len(items):,}")
