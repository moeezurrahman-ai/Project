import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# ============================================================
# BUILD FULL MODEL
# ============================================================
def build_model(recipes, train_df):
    print("Building TF-IDF + KNN model...")

    recipes = recipes.copy()
    # Recipe Quality Filter
    recipes = recipes[
        # Must have at least 1 ingredient!!
        (recipes['n_ingredients'] >= 1) &
        # Must have cooking time > 0!!
        (recipes['minutes'] > 0) &
        # Remove unreasonably long recipes!!
        (recipes['minutes'] <= 600) &
        # Remove zero/negative calorie recipes!!
        (recipes['calories'] > 0) &
        # Remove absurdly high calories!!
        (recipes['calories'] <= 10000) &
        # Must have at least 1 step!!
        (recipes['n_steps'] >= 1)
        ].copy()

    recipes['ingredients_str'] = recipes[
        'ingredients_parsed'
    ].apply(lambda x: ' '.join(x))

    # TF-IDF
    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words='english'
    )
    tfidf_matrix = tfidf.fit_transform(
        recipes['ingredients_str']
    )
    print(f"✅ TF-IDF matrix: {tfidf_matrix.shape}")

    # KNN on ingredients
    knn_model = NearestNeighbors(
        n_neighbors=20,
        metric='cosine',
        algorithm='brute',
        n_jobs=-1
    )
    knn_model.fit(tfidf_matrix)
    print("✅ Ingredient KNN trained!!")

    # Nutrition normalization
    nutrition_cols = [
        'calories', 'fat', 'sugar',
        'sodium', 'protein', 'sat_fat', 'carbs'
    ]
    scaler = MinMaxScaler()
    nutrition_scaled = scaler.fit_transform(
        recipes[nutrition_cols]
    )
    nutrition_df = pd.DataFrame(
        nutrition_scaled,
        columns=nutrition_cols,
        index=recipes.index
    )

    # KNN on nutrition
    knn_nutrition = NearestNeighbors(
        n_neighbors=20,
        metric='euclidean',
        algorithm='brute',
        n_jobs=-1
    )
    knn_nutrition.fit(nutrition_scaled)
    print("✅ Nutrition KNN trained!!")

    # ── SVD Model ──
    print("Building SVD model on ratings...")
    svd_model, user_factors, item_factors, \
    user_encoder, item_encoder = build_svd(train_df)
    print("✅ SVD model trained!!")

    return {
        'recipes':          recipes.reset_index(drop=True),
        'tfidf':            tfidf,
        'tfidf_matrix':     tfidf_matrix,
        'nutrition_df':     nutrition_df,
        'nutrition_scaled': nutrition_scaled,
        'scaler':           scaler,
        'nutrition_cols':   nutrition_cols,
        'knn_model':        knn_model,
        'knn_nutrition':    knn_nutrition,
        'svd_model':        svd_model,
        'user_factors':     user_factors,
        'item_factors':     item_factors,
        'user_encoder':     user_encoder,
        'item_encoder':     item_encoder,
        'train_df':         train_df
    }


# ============================================================
# SVD
# ============================================================
def build_svd(train_df, n_components=50):
    train_df = train_df[train_df['rating'] > 0].copy()

    # Use pre-encoded u and i columns!!
    n_users = train_df['u'].max() + 1
    n_items = train_df['i'].max() + 1

    rows = train_df['u'].values
    cols = train_df['i'].values
    vals = train_df['rating'].values

    user_item = csr_matrix(
        (vals, (rows, cols)),
        shape=(n_users, n_items)
    )

    svd = TruncatedSVD(
        n_components=n_components,
        random_state=42
    )
    user_factors = svd.fit_transform(user_item)
    item_factors = svd.components_.T

    # Encoders now use u and i directly!!
    user_encoder = dict(
        zip(train_df['user_id'], train_df['u'])
    )
    item_encoder = dict(
        zip(train_df['recipe_id'], train_df['i'])
    )

    print(f"✅ SVD matrix: {user_item.shape}")
    print(f"✅ User factors: {user_factors.shape}")
    print(f"✅ Item factors: {item_factors.shape}")

    return svd, user_factors, item_factors, \
           user_encoder, item_encoder

# ============================================================
# RECOMMEND — INGREDIENTS (KNN)
# ============================================================
def recommend_by_ingredients(user_ingredients,
                              model_data, n=10):
    tfidf       = model_data['tfidf']
    tfidf_matrix = model_data['tfidf_matrix']
    knn_model   = model_data['knn_model']
    recipes     = model_data['recipes']

    user_str = ' '.join(user_ingredients)
    user_vec = tfidf.transform([user_str])

    distances, indices = knn_model.kneighbors(
        user_vec, n_neighbors=n
    )

    results = recipes.iloc[indices[0]].copy()
    results['match_score'] = (1 - distances[0]).round(3)

    return results[[
        'name', 'minutes', 'n_ingredients',
        'calories', 'protein', 'fat', 'carbs',
        'ingredients', 'tags', 'description',
        'steps', 'match_score'
    ]]


# ============================================================
# RECOMMEND — NUTRITION (KNN)
# ============================================================
def recommend_by_nutrition(user_nutrition,
                            model_data, n=10):
    recipes          = model_data['recipes']
    nutrition_scaled = model_data['nutrition_scaled']
    scaler           = model_data['scaler']
    knn_nutrition    = model_data['knn_nutrition']
    nutrition_cols   = model_data['nutrition_cols']

    user_dict = {
        'calories': user_nutrition.get('max_calories', 500),
        'fat':      user_nutrition.get('max_fat', 30),
        'sugar':    user_nutrition.get('max_sugar', 50),
        'sodium':   user_nutrition.get('max_sodium', 500),
        'protein':  user_nutrition.get('min_protein', 20),
        'sat_fat':  user_nutrition.get('max_sat_fat', 20),
        'carbs':    user_nutrition.get('max_carbs', 50)
    }

    user_vec = scaler.transform(
        pd.DataFrame([user_dict])[nutrition_cols]
    )

    distances, indices = knn_nutrition.kneighbors(
        user_vec, n_neighbors=n * 3
    )

    candidates = recipes.iloc[indices[0]].copy()
    mask = np.ones(len(candidates), dtype=bool)

    if 'max_calories' in user_nutrition:
        mask &= candidates['calories'].values <= \
                user_nutrition['max_calories']
    if 'min_protein' in user_nutrition:
        mask &= candidates['protein'].values >= \
                user_nutrition['min_protein']
    if 'max_fat' in user_nutrition:
        mask &= candidates['fat'].values <= \
                user_nutrition['max_fat']
    if 'max_carbs' in user_nutrition:
        mask &= candidates['carbs'].values <= \
                user_nutrition['max_carbs']

    filtered = candidates[mask]
    if filtered.empty:
        filtered = candidates.sort_values(
            by=['protein', 'calories'],
            ascending=[False, True]
        )

    return filtered.head(n)[[
        'name', 'minutes', 'n_ingredients',
        'calories', 'protein', 'fat', 'carbs',
        'ingredients', 'tags', 'description', 'steps'
    ]]


# ============================================================
# RECOMMEND — SVD (Rating Based)
# ============================================================
def recommend_by_svd(user_id, model_data, n=10):
    user_encoder = model_data['user_encoder']
    item_encoder = model_data['item_encoder']
    user_factors = model_data['user_factors']
    item_factors = model_data['item_factors']
    recipes      = model_data['recipes']
    train_df     = model_data['train_df']

    if user_id not in user_encoder:
        return pd.DataFrame()

    u_idx = user_encoder[user_id]
    user_vec = user_factors[u_idx].reshape(1, -1)

    # Predict ratings for all items
    scores = cosine_similarity(
        user_vec, item_factors.T
    ).flatten()

    # Remove already rated
    rated = train_df[
        train_df['user_id'] == user_id
    ]['recipe_id'].tolist()
    rated_idx = [
        item_encoder[r]
        for r in rated
        if r in item_encoder
    ]
    scores[rated_idx] = -1

    top_indices = scores.argsort()[::-1][:n]
    item_decoder = {v: k for k, v in item_encoder.items()}
    top_recipe_ids = [
        item_decoder[i] for i in top_indices
    ]

    results = recipes[
        recipes['id'].isin(top_recipe_ids)
    ].copy()
    results['svd_score'] = scores[top_indices[:len(results)]]

    return results[[
        'name', 'minutes', 'n_ingredients',
        'calories', 'protein', 'fat', 'carbs',
        'ingredients', 'tags', 'description',
        'steps', 'svd_score'
    ]]


# ============================================================
# RECOMMEND — HYBRID
# ============================================================
def hybrid_recommend(user_ingredients, user_nutrition,
                     model_data, n=10):
    ingredient_recs = recommend_by_ingredients(
        user_ingredients, model_data, n=50
    ).reset_index(drop=True)

    if ingredient_recs.empty:
        return pd.DataFrame()

    mask = np.ones(len(ingredient_recs), dtype=bool)

    if 'max_calories' in user_nutrition:
        mask &= ingredient_recs['calories'].values <= \
                user_nutrition['max_calories']
    if 'min_protein' in user_nutrition:
        mask &= ingredient_recs['protein'].values >= \
                user_nutrition['min_protein']
    if 'max_fat' in user_nutrition:
        mask &= ingredient_recs['fat'].values <= \
                user_nutrition['max_fat']
    if 'max_carbs' in user_nutrition:
        mask &= ingredient_recs['carbs'].values <= \
                user_nutrition['max_carbs']

    filtered = ingredient_recs[mask]

    if filtered.empty:
        return ingredient_recs.head(n)

    return filtered.head(n)


# ============================================================
# RECOMMEND — SIMILAR RECIPE
# ============================================================
def get_similar_recipes(recipe_name, model_data, n=10):
    recipes      = model_data['recipes']
    knn_model    = model_data['knn_model']
    tfidf_matrix = model_data['tfidf_matrix']

    match = recipes[
        recipes['name'].str.lower() == recipe_name.lower()
    ]

    if match.empty:
        # Try partial match
        match = recipes[
            recipes['name'].str.lower().str.contains(
                recipe_name.lower(), na=False
            )
        ]

    if match.empty:
        return pd.DataFrame()

    idx = match.index[0]
    recipe_vec = tfidf_matrix[idx]

    distances, indices = knn_model.kneighbors(
        recipe_vec, n_neighbors=n + 1
    )

    similar = recipes.iloc[indices[0][1:]].copy()
    similar['similarity'] = (1 - distances[0][1:]).round(3)

    return similar[[
        'name', 'minutes', 'n_ingredients',
        'calories', 'protein', 'fat', 'carbs',
        'ingredients', 'tags', 'description',
        'steps', 'similarity'
    ]]