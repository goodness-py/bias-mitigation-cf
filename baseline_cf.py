import numpy as np
import pandas as pd
from collections import defaultdict
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split

def build_surprise_dataset(ratings_df):
    """Convert a ratings DataFrame to a surprise Dataset."""
    reader = Reader(rating_scale=(1, 5))
    return Dataset.load_from_df(
        ratings_df[["user_id", "item_id", "rating"]], reader
    )

def train_baseline(ratings_df, k=50, test_size=0.2, random_state=42):
    """
    Train user-based KNN (cosine) on ratings_df.

    Returns:
        algo : fitted KNNBasic
        trainset : surprise Trainset
        testset : list of (uid, iid, r_ui) tuples
        predictions: list of surprise Prediction objects
    """
    data = build_surprise_dataset(ratings_df)
    trainset, testset = train_test_split(
        data, test_size=test_size, random_state=random_state
    )

    sim_options = {"name": "cosine", "user_based": True, "min_support": 1}
    algo = KNNBasic(k=k, sim_options=sim_options, verbose=False)
    algo.fit(trainset)

    predictions = algo.test(testset)
    return algo, trainset, testset, predictions


def get_similarity_matrix(algo, trainset):
    """
    Return the full user-user cosine similarity matrix as a DataFrame.
    Rows/columns are inner user IDs (integers 0…n_users-1).
    """
    sim = algo.compute_similarities() 
    inner_ids = list(range(trainset.n_users))
    raw_ids = [trainset.to_raw_uid(i) for i in inner_ids]
    df = pd.DataFrame(sim, index=raw_ids, columns=raw_ids)
    return df


def get_top_n(predictions, n=10):
    """
    Return dict {user_id: [(item_id, estimated_rating), ...]} for each user,
    containing their top-N items (unseen items only, highest est first).
    """
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return dict(top_n)


def get_neighbourhoods(algo, trainset, k=None):
    """
    Return dict {raw_user_id: [raw_neighbour_ids]} using the model's k-NN selection.
    If k is None, uses the model's own k value.
    """
    if k is None:
        k = algo.k

    neighbours = {}
    n_users = trainset.n_users

    for inner_uid in range(n_users):
        raw_uid = trainset.to_raw_uid(inner_uid)
        all_neighbours = algo.get_neighbors(inner_uid, k=k)
        neighbours[raw_uid] = [trainset.to_raw_uid(n) for n in all_neighbours]

    return neighbours


if __name__ == "__main__":
    from data_loader import load_all

    ratings, users, items, gender_map, _ = load_all()
    algo, trainset, testset, predictions = train_baseline(ratings)

    from surprise import accuracy
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    print(mae)
    print(rmse)
