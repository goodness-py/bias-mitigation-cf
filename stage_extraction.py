import numpy as np
import pandas as pd
from baseline_cf import get_similarity_matrix, get_neighbourhoods, get_top_n

def extract_stage1_input(ratings_df, gender_map):
    """
    Stage 1 — Input representation.
    Returns per-user rating counts annotated with gender.
    """
    user_stats = (
        ratings_df.groupby("user_id")
        .agg(n_ratings=("rating", "count"), mean_rating=("rating", "mean"))
        .reset_index()
    )
    user_stats["gender"] = user_stats["user_id"].map(gender_map)
    return user_stats


def extract_stage2_similarity(algo, trainset):
    """
    Stage 2 — Similarity matrix.
    Returns full n_users × n_users similarity DataFrame.
    """
    return get_similarity_matrix(algo, trainset)


def extract_stage3_top_similar(algo, trainset, top_k=50):
    """
    Stage 3 — Top-50 similar users per user.
    Returns dict {raw_user_id: [raw_similar_user_ids]} (top_k entries each).
    """
    top_similar = {}
    for inner_uid in range(trainset.n_users):
        raw_uid = trainset.to_raw_uid(inner_uid)
        neighbours = algo.get_neighbors(inner_uid, k=top_k)
        top_similar[raw_uid] = [trainset.to_raw_uid(n) for n in neighbours]
    return top_similar


def extract_stage4_neighbourhood(algo, trainset):
    """
    Stage 4 — Selected k-neighbourhood per user (model's actual k).
    Returns dict {raw_user_id: [raw_neighbour_ids]}.
    """
    return get_neighbourhoods(algo, trainset)


def extract_stage5_predictions(predictions):
    """
    Stage 5 — Predicted ratings.
    Returns DataFrame [user_id, item_id, true_rating, est_rating].
    """
    rows = [
        {"user_id": uid, "item_id": iid, "true_rating": r_ui, "est_rating": est}
        for uid, iid, r_ui, est, _ in predictions
    ]
    return pd.DataFrame(rows)


def extract_stage6_ranking(predictions, n=10):
    """
    Stage 6 — Top-N ranked recommendations per user.
    Returns dict {user_id: [(item_id, est_rating), ...]}.
    """
    return get_top_n(predictions, n=n)


def extract_all_stages(algo, trainset, predictions, ratings_df, gender_map, top_k=50, top_n=10):
    """
    Run all stage extractions and return a dict keyed by stage name.

    Keys:
        stage1_input       : pd.DataFrame
        stage2_similarity  : pd.DataFrame  (n_users × n_users)
        stage3_top_similar : dict
        stage4_neighbourhood: dict
        stage5_predictions : pd.DataFrame
        stage6_ranking     : dict
    """
    return {
        "stage1_input":        extract_stage1_input(ratings_df, gender_map),
        "stage2_similarity":   extract_stage2_similarity(algo, trainset),
        "stage3_top_similar":  extract_stage3_top_similar(algo, trainset, top_k=top_k),
        "stage4_neighbourhood": extract_stage4_neighbourhood(algo, trainset),
        "stage5_predictions":  extract_stage5_predictions(predictions),
        "stage6_ranking":      extract_stage6_ranking(predictions, n=top_n),
    }


if __name__ == "__main__":
    from data_loader import load_all
    from baseline_cf import train_baseline

    ratings, users, items, gender_map, _ = load_all()
    algo, trainset, testset, predictions = train_baseline(ratings)
    stages = extract_all_stages(algo, trainset, predictions, ratings, gender_map)

    print("Stage artefacts extracted:")
    for name, artefact in stages.items():
        if isinstance(artefact, pd.DataFrame):
            print(f"  {name}: DataFrame {artefact.shape}")
        elif isinstance(artefact, dict):
            print(f"  {name}: dict with {len(artefact)} keys")
