import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
import matplotlib.pyplot as plt

# --- Load Data ---
recommendations = pd.read_csv("recommendations.csv")
games = pd.read_csv("games.csv")
users = pd.read_csv("users.csv")

print(recommendations.info())
print(games.info())
print(users.info())

# --- Eksplorasi Data ---
print("Distribusi Rating Game:")
print(games['rating'].value_counts())

print("\nRata-rata Positive Ratio per Rating:")
print(games.groupby('rating')['positive_ratio'].mean().sort_values(ascending=False))

print("\nPersentase Dukungan Platform:")
print((games[['win', 'mac', 'linux', 'steam_deck']].mean() * 100).round(2))

print("\nDistribusi Review Rekomendasi (%):")
print(recommendations['is_recommended'].value_counts(normalize=True).mul(100).round(2))

# --- Preprocessing ---
recommendations['date'] = pd.to_datetime(recommendations['date'])
games['date_release'] = pd.to_datetime(games['date_release'])
games[['win', 'mac', 'linux', 'steam_deck']] = games[['win', 'mac', 'linux', 'steam_deck']].astype(bool)
recommendations['is_recommended'] = recommendations['is_recommended'].astype(bool)

# Merge data
merged_df = recommendations.merge(games, on='app_id', how='left').merge(users, on='user_id', how='left')

# Drop rows dengan data penting yang hilang
important_columns = ['title', 'date_release', 'price_final', 'positive_ratio', 'products', 'reviews']
cleaned_df = merged_df.dropna(subset=important_columns)

print(cleaned_df.info())
print(cleaned_df.head())
print(f"Total rows: {cleaned_df.shape[0]}")
print(f"Total columns: {cleaned_df.shape[1]}")

# --- Simpan hasil preprocessing ke CSV (khusus jika menggunakan Google Colab) ---
# from google.colab import files
# cleaned_df.to_csv("cleaned_recommendations.csv", index=False)
# files.download("cleaned_recommendations.csv")

# --- Content-based Filtering ---
content_features = games[['rating', 'positive_ratio', 'win', 'mac', 'linux', 'steam_deck']].copy()
content_features['rating'] = content_features['rating'].fillna("Unknown")
content_features['rating_encoded'] = content_features['rating'].astype('category').cat.codes

feature_matrix = content_features[['rating_encoded', 'positive_ratio', 'win', 'mac', 'linux', 'steam_deck']]
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(feature_matrix)

similarity_matrix = cosine_similarity(scaled_features)

def recommend_similar_games(game_id, top_n=5):
    try:
        idx = games[games['app_id'] == game_id].index[0]
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        game_indices = [i[0] for i in sim_scores]
        return games.iloc[game_indices][['app_id', 'title']]
    except IndexError:
        return "Game ID not found."

print("\nRekomendasi mirip dengan 'Super Blackjack Battle 2 Turbo Edition - The Card Warriors':")
print(recommend_similar_games(545200))

# --- Evaluasi dengan Precision@K ---
def precision_at_k(user_game_df, k=5):
    hits = 0
    total = 0

    for user_id in user_game_df['user_id'].unique():
        liked_games = user_game_df[(user_game_df['user_id'] == user_id) & (user_game_df['is_recommended'] == True)]

        if liked_games.empty:
            continue

        anchor_game_id = liked_games['app_id'].iloc[0]
        recommended_games = recommend_similar_games(anchor_game_id, top_n=k)
        if isinstance(recommended_games, str):
            continue

        recommended_game_ids = set(recommended_games['app_id'])
        liked_game_ids = set(liked_games['app_id'])

        hits += len(recommended_game_ids & liked_game_ids)
        total += k

    precision = hits / total if total > 0 else 0
    return precision

precision = precision_at_k(cleaned_df, k=5)
print(f"\nPrecision@5 dari model Content-Based Filtering: {precision:.4f}")

# --- Collaborative Filtering menggunakan Surprise ---
cf_data = recommendations[['user_id', 'app_id', 'is_recommended']].copy()
cf_data['is_recommended'] = cf_data['is_recommended'].astype(int)

reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(cf_data[['user_id', 'app_id', 'is_recommended']], reader)

model = SVD()
cross_validate(model, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

trainset = data.build_full_trainset()
model.fit(trainset)

def recommend_for_user(user_id, n=5):
    all_game_ids = games['app_id'].unique()
    reviewed_games = cf_data[cf_data['user_id'] == user_id]['app_id'].values
    unseen_games = [app_id for app_id in all_game_ids if app_id not in reviewed_games]

    if not trainset.knows_user(user_id):
        print(f"User ID {user_id} tidak ditemukan dalam data pelatihan.")
        return pd.DataFrame()

    predictions = [model.predict(user_id, app_id) for app_id in unseen_games]
    top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    top_app_ids = [pred.iid for pred in top_predictions]
    return games[games['app_id'].isin(top_app_ids)][['app_id', 'title']]

print("\nRekomendasi game untuk user_id = 253880:")
print(recommend_for_user(253880))

# --- Visualisasi Rekomendasi Mirip ---
example_game_id = 545200
example_game_title = games[games['app_id'] == example_game_id]['title'].values[0]

idx = games[games['app_id'] == example_game_id].index[0]
similarity_scores = list(enumerate(similarity_matrix[idx]))
similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]
scores = [s[1] for s in similarity_scores]
titles = games.iloc[[s[0] for s in similarity_scores]]['title'].values

plt.figure(figsize=(8, 4))
plt.barh(titles[::-1], scores[::-1], color='skyblue')
plt.xlabel('Similarity Score')
plt.title(f'Game Mirip dengan: {example_game_title}')
plt.tight_layout()
plt.show()

# --- Visualisasi Rekomendasi User ---
def visualize_recommendations(user_id, n=5):
    reviewed_games = cf_data[cf_data['user_id'] == user_id]['app_id'].values
    unseen_games = [app_id for app_id in games['app_id'].unique() if app_id not in reviewed_games]
    predictions = [model.predict(user_id, app_id) for app_id in unseen_games]
    top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]

    titles = [games[games['app_id'] == int(pred.iid)]['title'].values[0] for pred in top_predictions]
    scores = [pred.est for pred in top_predictions]

    plt.figure(figsize=(8, 5))
    plt.barh(titles[::-1], scores[::-1], color='mediumseagreen')
    plt.xlabel('Skor Prediksi (est)')
    plt.title(f'Top {n} Rekomendasi Game untuk User {user_id}')
    plt.tight_layout()
    plt.show()

visualize_recommendations(253880, n=5)
