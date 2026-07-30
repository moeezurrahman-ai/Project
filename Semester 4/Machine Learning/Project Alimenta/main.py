import streamlit as st
import pandas as pd
import numpy as np
import ast
import plotly.express as px
import plotly.graph_objects as go
from train import (
    build_model,
    recommend_by_ingredients,
    recommend_by_nutrition,
    recommend_by_svd,
    hybrid_recommend,
    get_similar_recipes
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Machine Learning",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
.recipe-card {
    background-color: #1e1e2e;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    border-left: 4px solid #ff6b6b;
}
.stTabs [data-baseweb="tab"] {
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INIT
# ============================================================
#if 'favourites'       not in st.session_state:
#   st.session_state.favourites       = []
#if 'history'          not in st.session_state:
#    st.session_state.history          = []
#if 'liked_ingredients' not in st.session_state:
#    st.session_state.liked_ingredients = []
#if 'search_count'     not in st.session_state:
#    st.session_state.search_count     = 0
# ============================================================
# SESSION STATE INIT
# ============================================================
if 'logged_in'         not in st.session_state:
    st.session_state.logged_in         = False
if 'current_user'      not in st.session_state:
    st.session_state.current_user      = None
if 'favourites'        not in st.session_state:
    st.session_state.favourites        = []
if 'history'           not in st.session_state:
    st.session_state.history           = []
if 'liked_ingredients' not in st.session_state:
    st.session_state.liked_ingredients = []
if 'search_count'      not in st.session_state:
    st.session_state.search_count      = 0

# ============================================================
# USER DATABASE (just a demo account)
# ============================================================
USERS = {
    "project":   {"password": "machine learning",    "user_id": 22095, "name": "project"}
}

# ============================================================
# LOGIN PAGE
# ============================================================
def show_login():
    st.markdown("""
    <style>
    .login-container {
        max-width: 450px;
        margin: 80px auto;
        padding: 40px;
        background: linear-gradient(135deg, #2d1200, #3d1a00);
        border-radius: 20px;
        border: 1px solid #ff6b35;
        box-shadow: 0 20px 60px rgba(255, 107, 53, 0.3);
    }
    .login-title {
        text-align: center;
        font-family: 'Playfair Display', serif;
        font-size: 2.5rem;
        background: linear-gradient(90deg, #ff6b35, #ffd700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .login-subtitle {
        text-align: center;
        color: #fff8f0;
        opacity: 0.7;
        margin-bottom: 30px;
        font-size: 0.95rem;
    }
    .demo-account {
        background: #1f0d00;
        border-radius: 10px;
        padding: 12px 16px;
        border: 1px solid #ff6b3555;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Centered login form!!
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            '<div class="login-title">Project Alimenta</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="login-subtitle">'
            'Personalised Food Recommendation Engine'
            '</div>',
            unsafe_allow_html=True
        )

        st.divider()

        username = st.text_input(
            "👤 Username",
            placeholder="Enter your username"
        )
        password = st.text_input(
            "🗝️ Password",
            type="password",
            placeholder="Enter your password"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        login_btn = st.button(
            "Login",
            use_container_width=True
        )

        if login_btn:
            if username in USERS and \
               USERS[username]['password'] == password:
                st.session_state.logged_in    = True
                st.session_state.current_user = username
                st.success(
                    f"✅ Welcome back, "
                    f"{USERS[username]['name']}!🎉"
                )
                st.rerun()
            else:
                st.error(
                    "❌ Invalid username or password!"
                )

        # Demo accounts hint!!
        st.markdown("""
        <div class="demo-account">
            <p style="color:#ffd700; font-weight:600;
                      margin-bottom:8px;">
                💡 Demo Account:
            </p>
            <p style="color:#fff8f0; font-size:0.85rem;
                      margin:0;">
                👤 Username: project 
                🗝️ Password: machine learning
            </p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
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
    return build_model(recipes, train_df)


# ============================================================
# HELPERS
# ============================================================
def show_recipe_card(row, score_col=None):
    with st.expander(
        f"🍽️ {row['name']} — "
        f"{int(row['calories'])} cal | "
        f"{int(row['protein'])}g protein | "
        f"⏱️ {int(row['minutes'])} mins"
    ):
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**📊 Nutrition Facts**")
            st.table(pd.DataFrame({
                'Nutrient': [
                    'Calories', 'Protein', 'Fat', 'Carbs'
                ],
                'Amount': [
                    f"{row['calories']} kcal",
                    f"{row['protein']}g",
                    f"{row['fat']}g",
                    f"{row['carbs']}g"
                ]
            }))
            if score_col and score_col in row.index:
                st.metric(
                    "Match Score",
                    f"{row[score_col]:.2%}"
                )

        with col2:
            st.markdown("**🥕 Ingredients**")
            try:
                ings = ast.literal_eval(row['ingredients']) \
                    if isinstance(row['ingredients'], str) \
                    else row['ingredients']
                for ing in ings:
                    st.markdown(f"- {ing}")
            except:
                st.write(row['ingredients'])

        st.markdown("**📝 Description**")
        st.write(row.get('description', 'N/A'))

        st.markdown("**👨‍🍳 Steps**")
        try:
            steps = ast.literal_eval(row['steps']) \
                if isinstance(row['steps'], str) \
                else row['steps']

            if isinstance(steps, list) and len(steps) > 0:
                for i, step in enumerate(steps, 1):
                    st.markdown(
                        f"**Step {i}:** {step}"
                    )
                    if i < len(steps):
                        st.markdown("---")
            else:
                st.write("No steps available!!")
        except Exception as e:
            # Try manual string cleaning!!
            try:
                raw = row.get('steps', '')
                raw = str(raw).strip()
                raw = raw.strip("[]")
                parts = raw.split("', '")
                parts = [
                    p.strip().strip("'\"")
                    for p in parts
                    if p.strip()
                ]
                for i, step in enumerate(parts, 1):
                    st.markdown(f"**Step {i}:** {step}")
                    if i < len(parts):
                        st.markdown("---")
            except:
                st.write(row.get('steps', 'N/A'))


def show_nutrition_chart(df, title):
    chart_df = df.head(10).copy()
    chart_df['name_short'] = chart_df['name'].str[:20]

    fig = px.bar(
        chart_df,
        x='name_short',
        y=['calories', 'protein', 'fat', 'carbs'],
        title=title,
        barmode='group',
        color_discrete_map={
            'calories': '#ff6b6b',
            'protein':  '#4ecdc4',
            'fat':      '#ffe66d',
            'carbs':    '#a8e6cf'
        }
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig, use_container_width=True)


def show_macro_pie(row):
    fig = go.Figure(data=[go.Pie(
        labels=['Protein', 'Fat', 'Carbs'],
        values=[row['protein'], row['fat'], row['carbs']],
        hole=0.4,
        marker_colors=['#4ecdc4', '#ffe66d', '#a8e6cf']
    )])
    fig.update_layout(
        height=250,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        margin=dict(t=0, b=0, l=0, r=0)
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# APP ROUTING — Login vs Main App!!
# ============================================================
if not st.session_state.logged_in:
    show_login()
else:
    # ── HEADER ──
    col1, col2 = st.columns([8, 1])
    with col1:
        st.title("Project Alimenta")
        st.markdown(
            "Personalised Food Recommendation Engine"
            ""
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        user_name = USERS[
            st.session_state.current_user
        ]['name']
        st.markdown(
            f"👤 **{user_name}**",
        )
        if st.button("🚪 Logout"):
            st.session_state.logged_in    = False
            st.session_state.current_user = None
            st.session_state.favourites   = []
            st.session_state.history      = []
            st.session_state.liked_ingredients = []
            st.session_state.search_count = 0
            st.rerun()

    st.divider()

    # ── LOAD MODEL ──
    with st.spinner("🔄 Loading model..."):
        model_data = load_model()
    st.success("✅ Model loaded!")
    st.divider()

    # ── SIDEBAR ──
    st.sidebar.title("⚙️ Your Preferences")

    # Show logged in user in sidebar!!
    st.sidebar.markdown(
        f"👤 Logged in as: **"
        f"{USERS[st.session_state.current_user]['name']}"
        f"**"
    )
    st.sidebar.divider()

    mode = st.sidebar.radio(
        "Recommendation Mode",
        [
            "🥕 By Ingredients",
            "🥗 By Nutrition",
            "🍽️ Hybrid",
            "🔍 Similar to Recipe",
            "⭐ By Rating History (SVD)"
        ]
    )

    st.sidebar.divider()

    # Dynamic inputs
    if mode in ["🥕 By Ingredients", "🍽️ Hybrid"]:
        st.sidebar.markdown("### 🥕 Ingredients")
        suggested = ', '.join(
            st.session_state.liked_ingredients[:5]
        ) if st.session_state.liked_ingredients else ''
        ingredients_input = st.sidebar.text_input(
            "Enter ingredients (comma separated)",
            value=suggested,
            placeholder="e.g. chicken, garlic, lemon"
        )
        if suggested:
            st.sidebar.caption(
                "💡 Pre-filled from your history!!"
            )

    if mode in ["🥗 By Nutrition", "🍽️ Hybrid"]:
        st.sidebar.markdown("### 🥗 Nutrition Goals")
        max_calories = st.sidebar.slider(
            "Max Calories", 100, 2000, 500, 50
        )
        min_protein = st.sidebar.slider(
            "Min Protein (g)", 0, 100, 20, 5
        )
        max_fat = st.sidebar.slider(
            "Max Fat (g)", 0, 100, 30, 5
        )
        max_carbs = st.sidebar.slider(
            "Max Carbs (g)", 0, 100, 50, 5
        )

    if mode == "🔍 Similar to Recipe":
        st.sidebar.markdown("### 🔍 Recipe Name")
        recipe_name_input = st.sidebar.text_input(
            "Enter recipe name",
            placeholder="e.g. Chicken Soup"
        )

    if mode == "⭐ By Rating History (SVD)":
        # Auto-fill user_id from logged in user!!
        auto_user_id = USERS[
            st.session_state.current_user
        ]['user_id']
        st.sidebar.markdown("### 👤 Your User ID")
        st.sidebar.info(
            f"🤖 Using your profile ID: **{auto_user_id}**"
        )
        svd_user_id = auto_user_id

    st.sidebar.divider()
    st.sidebar.markdown("### 🏷️ Filters")

    cuisine_tags = [
        'None',
        'italian', 'mexican', 'asian', 'american',
        'indian', 'mediterranean', 'french', 'greek',
        'japanese', 'chinese', 'african', 'middle-eastern',
        'breakfast', 'lunch', 'dinner', 'desserts',
        'snacks', 'appetizers', 'beverages', 'soups',
        'salads', 'sandwiches',
        'vegetarian', 'vegan', 'low-carb', 'low-fat',
        'low-sodium', 'low-calorie', 'high-protein',
        'gluten-free', 'diabetic', 'healthy',
        '30-minutes-or-less', '15-minutes-or-less',
        '60-minutes-or-less',
        'chicken', 'beef', 'pork', 'seafood',
        'fish', 'pasta', 'rice', 'eggs',
        'vegetables', 'fruit'
    ]
    selected_cuisine = st.sidebar.selectbox(
        "Cuisine / Diet / Type", cuisine_tags
    )
    time_filter = st.sidebar.slider(
        "Max Cooking Time (mins)", 5, 300, 120, 5
    )
    n_recommendations = st.sidebar.slider(
        "Number of Recommendations", 5, 20, 10, 1
    )

    recommend_btn = st.sidebar.button(
        "Recommend",
        use_container_width=True
    )

    if st.sidebar.button(
        "🗑️ Clear My History",
        use_container_width=True
    ):
        st.session_state.history           = []
        st.session_state.liked_ingredients = []
        st.session_state.search_count      = 0
        st.rerun()

    # ============================================================
    # RESULTS
    # ============================================================
    if recommend_btn:
        user_nutrition = {}
        if mode in ["🥗 By Nutrition", "🍽️ Hybrid"]:
            user_nutrition = {
                'max_calories': max_calories,
                'min_protein':  min_protein,
                'max_fat':      max_fat,
                'max_carbs':    max_carbs
            }

        results   = pd.DataFrame()
        score_col = None

        if mode == "🥕 By Ingredients":
            user_ingredients = [
                i.strip()
                for i in ingredients_input.split(',')
                if i.strip()
            ]
            if not user_ingredients:
                st.warning("Enter the ingredients!")
                st.stop()
            with st.spinner("🔍 Finding recipes..."):
                results = recommend_by_ingredients(
                    user_ingredients, model_data,
                    n=n_recommendations * 3
                )
            score_col = 'match_score'
            st.session_state.liked_ingredients.extend(
                user_ingredients
            )
            st.session_state.liked_ingredients = list(set(
                st.session_state.liked_ingredients
            ))

        elif mode == "🥗 By Nutrition":
            with st.spinner("🔍 Finding recipes..."):
                results = recommend_by_nutrition(
                    user_nutrition, model_data,
                    n=n_recommendations * 3
                )

        elif mode == "🍽️ Hybrid":
            user_ingredients = [
                i.strip()
                for i in ingredients_input.split(',')
                if i.strip()
            ]
            if not user_ingredients:
                st.warning("Please enter ingredients!")
                st.stop()
            with st.spinner("🔍 Finding best matches..."):
                results = hybrid_recommend(
                    user_ingredients, user_nutrition,
                    model_data, n=n_recommendations * 3
                )
            score_col = 'match_score'
            st.session_state.liked_ingredients.extend(
                user_ingredients
            )

        elif mode == "🔍 Similar to Recipe":
            if not recipe_name_input.strip():
                st.warning(
                    "Please enter a recipe name!"
                )
                st.stop()
            with st.spinner("🔍 Finding similar recipes..."):
                results = get_similar_recipes(
                    recipe_name_input.strip(),
                    model_data,
                    n=n_recommendations * 3
                )
            score_col = 'similarity'
            if results.empty:
                st.error(
                    f"❌ Recipe '{recipe_name_input}' "
                    f"not found!"
                )
                st.stop()

        elif mode == "⭐ By Rating History (SVD)":
            with st.spinner(
                "🤖 SVD predicting your preferences..."
            ):
                results = recommend_by_svd(
                    svd_user_id, model_data,
                    n=n_recommendations * 3
                )
            score_col = 'svd_score'
            if results.empty:
                st.error(
                    f"❌ No rating history found "
                    f"for your profile!"
                )
                st.stop()

        # ── Filters ──
        if selected_cuisine != 'None' and \
           not results.empty:
            mask = results['tags'].str.contains(
                selected_cuisine, case=False, na=False
            )
            if mask.any():
                results = results[mask]
            else:
                st.warning(
                    f"⚠️ No '{selected_cuisine}' recipes "
                    f"found — showing best matches!"
                )

        if not results.empty:
            results = results[
                results['minutes'] <= time_filter
            ]

        results = results.head(
            n_recommendations
        ).reset_index(drop=True)

        if results.empty:
            st.error(
                "❌ No recipes found!! "
                "Try adjusting your filters!"
            )
            st.stop()

        st.session_state.search_count += 1
        st.session_state.history.append({
            'search': st.session_state.search_count,
            'mode':   mode,
            'count':  len(results)
        })

        st.success(
            f"✅ Found {len(results)} recipes for you!"
        )

        tab1, tab2, tab3, tab4 = st.tabs([
            "🍽️ Recipes",
            "📊 Nutrition Charts",
            "⭐ Favourites",
            "🕐 My History"
        ])

        with tab1:
            st.subheader("🍽️ Your Recommendations")
            for _, row in results.iterrows():
                col1, col2 = st.columns([9, 1])
                with col1:
                    show_recipe_card(row, score_col)
                with col2:
                    if st.button(
                        "⭐",
                        key=f"fav_{row['name']}"
                    ):
                        if row['name'] not in \
                                st.session_state.favourites:
                            st.session_state.favourites\
                                .append(row['name'])
                            st.success("Added!!")

            st.divider()
            csv = results[[
                'name', 'minutes', 'calories',
                'protein', 'fat', 'carbs'
            ]].to_csv(index=False)
            st.download_button(
                "📥 Download Recommendations as CSV",
                data=csv,
                file_name="recommendations.csv",
                mime="text/csv",
                use_container_width=True
            )

        with tab2:
            st.subheader("📊 Nutrition Comparison")
            show_nutrition_chart(
                results, "Nutritional Breakdown"
            )
            st.divider()
            st.subheader(
                "🥧 Macro Breakdown — Top Recipe"
            )
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(
                    f"**{results.iloc[0]['name']}**"
                )
                st.metric(
                    "Calories",
                    f"{results.iloc[0]['calories']} kcal"
                )
                st.metric(
                    "Protein",
                    f"{results.iloc[0]['protein']}g"
                )
                st.metric(
                    "Fat",
                    f"{results.iloc[0]['fat']}g"
                )
                st.metric(
                    "Carbs",
                    f"{results.iloc[0]['carbs']}g"
                )
            with col2:
                show_macro_pie(results.iloc[0])

        with tab3:
            st.subheader("Your Liked Recipes")
            if not st.session_state.favourites:
                st.info(
                    "No favourites yet! "
                    "Click on any recipe!"
                )
            else:
                for fav in st.session_state.favourites:
                    st.markdown(f"- 🍽️ {fav}")
                if st.button("🗑️ Clear Favourites"):
                    st.session_state.favourites = []
                    st.rerun()
                fav_df = pd.DataFrame(
                    st.session_state.favourites,
                    columns=['Recipe Name']
                )
                st.download_button(
                    "📥 Download Favourites",
                    data=fav_df.to_csv(index=False),
                    file_name="favourites.csv",
                    mime="text/csv"
                )

        with tab4:
            st.subheader("🕐 Your Search History")
            if not st.session_state.history:
                st.info("No searches yet!!")
            else:
                st.markdown(
                    f"**Total searches:** "
                    f"{st.session_state.search_count}"
                )
                if st.session_state.liked_ingredients:
                    st.markdown(
                        f"**Your favourite ingredients:** "
                        f"{', '.join(st.session_state.liked_ingredients)}"
                    )
                history_df = pd.DataFrame(
                    st.session_state.history
                )
                st.dataframe(
                    history_df,
                    use_container_width=True
                )

    else:
        st.info(
            "Set your preferences in the sidebar "
            "and click **Get Recommendations!**"
        )
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Recipes", "226,658")
        with col2:
            st.metric("Ingredient Index", "4,105")
        with col3:
            st.metric("Machine Learning Model", "KNN + SVD")
        with col4:
            st.metric("Mode", "5")