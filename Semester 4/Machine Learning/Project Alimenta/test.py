import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from train import (
    build_model,
    recommend_by_ingredients,
    recommend_by_svd
)


# ============================================================
# LOAD DATA
# ============================================================
def load_data():
    print("Loading data...")
    raw_recipes = pd.read_csv('RAW_recipes.csv')
    recipes = raw_recipes.dropna(
        subset=['description', 'ingredients', 'nutrition']
    ).copy()

    nutrition_cols = [
        'calories', 'fat', 'sugar',
        'sodium', 'protein', 'sat_fat', 'carbs'
    ]
    recipes[nutrition_cols] = pd.DataFrame(
        recipes['nutrition']
        .apply(lambda x: eval(x))
        .tolist(),
        index=recipes.index
    )
    recipes['ingredients_parsed'] = recipes[
        'ingredients'
    ].apply(lambda x: eval(x))
    recipes['name'] = recipes['name'].str.strip().str.title()

    train_df = pd.read_csv('interactions_train.csv')
    test_df  = pd.read_csv('interactions_test.csv')
    val_df   = pd.read_csv('interactions_validation.csv')

    return recipes, train_df, test_df, val_df


# ============================================================
# RMSE EVALUATION
# ============================================================
def evaluate_rmse(model_data, test_df, sample_size=1000):
    print("\n--- RMSE Evaluation ---")

    user_factors = model_data['user_factors']
    item_factors = model_data['item_factors']

    # Use pre-encoded u and i columns!!
    test_clean = test_df[test_df['rating'] > 0].copy()
    test_clean = test_clean.dropna(
        subset=['u', 'i']
    ).reset_index(drop=True)

    # Filter valid indices
    max_u = user_factors.shape[0] - 1
    max_i = item_factors.shape[0] - 1

    test_clean = test_clean[
        (test_clean['u'] <= max_u) &
        (test_clean['i'] <= max_i)
    ].reset_index(drop=True)

    print(f"Valid test samples: {len(test_clean)}")

    if len(test_clean) == 0:
        print("❌ No valid test samples!!")
        return None, None

    if len(test_clean) > sample_size:
        test_clean = test_clean.sample(
            n=sample_size, random_state=42
        )

    predictions = []
    actuals     = []

    for _, row in test_clean.iterrows():
        u_idx = int(row['u'])
        i_idx = int(row['i'])
        pred  = np.dot(
            user_factors[u_idx],
            item_factors[i_idx]
        )
        pred = np.clip(pred, 1, 5)
        predictions.append(pred)
        actuals.append(row['rating'])

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae  = np.mean(
        np.abs(np.array(actuals) - np.array(predictions))
    )

    print(f"✅ RMSE:    {rmse:.4f}")
    print(f"✅ MAE:     {mae:.4f}")
    print(f"✅ Samples: {len(predictions)}")

    return rmse, mae
# ============================================================
# PRECISION@K
# ============================================================
def evaluate_precision(model_data, sample_size=200):
    print("\n--- Precision@10 Evaluation ---")
    recipes = model_data['recipes']

    sample = recipes.sample(
        n=sample_size, random_state=42
    )
    precision_scores = []

    for _, row in sample.iterrows():
        try:
            ings = row['ingredients_parsed'][:3]
            recs = recommend_by_ingredients(
                ings, model_data, n=10
            )
            relevant = 0
            for _, rec in recs.iterrows():
                try:
                    rec_ings = set(eval(rec['ingredients']))
                    user_ings = set(ings)
                    if len(rec_ings & user_ings) > 0:
                        relevant += 1
                except:
                    continue
            precision_scores.append(relevant / 10)
        except:
            continue

    avg_precision = np.mean(precision_scores)
    print(f"✅ Avg Precision@10: {avg_precision:.4f}")
    return avg_precision


# ============================================================
# COVERAGE
# ============================================================
def evaluate_coverage(model_data, sample_size=500):
    print("\n--- Coverage Evaluation ---")
    recipes = model_data['recipes']

    sample = recipes.sample(
        n=sample_size, random_state=42
    )
    covered = set()

    for _, row in sample.iterrows():
        try:
            ings = row['ingredients_parsed'][:3]
            recs = recommend_by_ingredients(
                ings, model_data, n=10
            )
            covered.update(recs['name'].tolist())
        except:
            continue

    coverage = len(covered) / len(recipes) * 100
    print(f"✅ Unique recipes recommended: {len(covered)}")
    print(f"✅ Catalogue Coverage:         {coverage:.2f}%")
    return coverage


# ============================================================
# MODEL COMPARISON
# ============================================================
def compare_models(model_data, test_df):
    print("\n--- Model Comparison ---")

    knn_precision = evaluate_precision(model_data)
    rmse, mae     = evaluate_rmse(model_data, test_df)

    print("\n" + "=" * 45)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 45)
    print(f"{'Model':<20} {'Metric':<15} {'Score':<10}")
    print("-" * 45)
    print(
        f"{'KNN (Ingredients)':<20} "
        f"{'Precision@10':<15} "
        f"{knn_precision:.4f}"
    )
    if rmse is not None:
        print(
            f"{'SVD (Ratings)':<20} "
            f"{'RMSE':<15} "
            f"{rmse:.4f}"
        )
        print(
            f"{'SVD (Ratings)':<20} "
            f"{'MAE':<15} "
            f"{mae:.4f}"
        )
    else:
        print(
            f"{'SVD (Ratings)':<20} "
            f"{'RMSE':<15} "
            f"N/A — no overlap"
        )
    print("=" * 45)

    return {
        'knn_precision': knn_precision,
        'svd_rmse':      rmse,
        'svd_mae':       mae
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    recipes, train_df, test_df, val_df = load_data()

    print("\nBuilding model...")
    model_data = build_model(recipes, train_df)

    print("\n" + "=" * 45)
    print("FULL EVALUATION REPORT")
    print("=" * 45)

    metrics = compare_models(model_data, test_df)
    evaluate_coverage(model_data, sample_size=500)

    print("\n✅ EVALUATION COMPLETE!!")

