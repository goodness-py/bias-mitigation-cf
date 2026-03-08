import pandas as pd

def merge_demographics(ratings_df, users_df):
    """Merge ratings with user gender/age/occupation."""
    return ratings_df.merge(
        users_df[["user_id", "age", "gender", "occupation"]],
        on="user_id",
        how="left",
    )

def compute_representation_stats(enriched_df):
    """
    Compute Stage 2 bias statistics at the input data level.

    Returns a dict with:
        total_ratings       : int
        group_counts        : dict {gender -> count}
        group_proportions   : dict {gender -> fraction}
        avg_ratings_per_user: dict {gender -> mean ratings per user}
        unique_items        : dict {gender -> unique item count}
        activity_gap        : float  (M_avg - F_avg ratings per user)
        item_coverage_gap   : float  (|M_items - F_items| / total_items)
    """
    total = len(enriched_df)
    group_counts = enriched_df.groupby("gender").size().to_dict()
    group_props = {g: c / total for g, c in group_counts.items()}

    avg_per_user = (
        enriched_df.groupby(["gender", "user_id"])
        .size()
        .reset_index(name="n_ratings")
        .groupby("gender")["n_ratings"]
        .mean()
        .to_dict()
    )

    unique_items = (
        enriched_df.groupby("gender")["item_id"].nunique().to_dict()
    )

    m_avg = avg_per_user.get("M", 0)
    f_avg = avg_per_user.get("F", 0)
    activity_gap = m_avg - f_avg

    total_items = enriched_df["item_id"].nunique()
    m_items = unique_items.get("M", 0)
    f_items = unique_items.get("F", 0)
    item_coverage_gap = abs(m_items - f_items) / total_items if total_items else 0.0

    return {
        "total_ratings": total,
        "group_counts": group_counts,
        "group_proportions": group_props,
        "avg_ratings_per_user": avg_per_user,
        "unique_items": unique_items,
        "activity_gap": activity_gap,
        "item_coverage_gap": item_coverage_gap,
    }


def preprocess(ratings_df, users_df):
    """
    Full preprocessing pipeline.

    Returns:
        enriched_df : pd.DataFrame  (ratings merged with demographics)
        stats       : dict          (representation statistics)
    """
    enriched_df = merge_demographics(ratings_df, users_df)
    stats = compute_representation_stats(enriched_df)
    return enriched_df, stats


if __name__ == "__main__":
    from data_loader import load_all

    ratings, users, items, gender_map, _ = load_all()
    enriched, stats = preprocess(ratings, users)

    print("=== Stage 2: Input Data Bias ===")
    print(f"Total ratings      : {stats['total_ratings']:,}")
    print(f"Group counts       : {stats['group_counts']}")
    print(f"Group proportions  : { {g: f'{v:.1%}' for g,v in stats['group_proportions'].items()} }")
    print(f"Avg ratings/user   : { {g: f'{v:.1f}' for g,v in stats['avg_ratings_per_user'].items()} }")
    print(f"Unique items       : {stats['unique_items']}")
    print(f"Activity gap       : {stats['activity_gap']:.2f} ratings/user")
    print(f"Item coverage gap  : {stats['item_coverage_gap']:.4f}")
