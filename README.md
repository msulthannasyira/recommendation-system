# steam-games-recommendation-system
Analisis prediksi harga kripto bitcoin di masa depan

- Nama Lengkap: Muhammad Sulthan Nasyira
- Alur Belajar: Machine Learning Engineer
- Cohort ID: MC589D5Y2486
- Coding Camp Email Username:   mc589d5y2486@student.devacademy.id
- Email Terdaftar:   sulthanasyirah@gmail.com
- Group Belajar:   MC-49

## 1. Domain Proyek
### Latar Belakang
Industri game digital mengalami pertumbuhan pesat dalam beberapa tahun terakhir, baik dari sisi jumlah pemain maupun jumlah game yang dirilis setiap tahunnya. Platform distribusi game seperti Steam, Epic Games Store, dan GOG menjadi wadah utama bagi para pengembang untuk mempublikasikan game mereka. Namun, lonjakan jumlah game ini menimbulkan tantangan bagi para pemain dalam menemukan game yang benar-benar sesuai dengan preferensi mereka.

Menurut laporan Newzoo (2023), terdapat lebih dari 1 juta game yang tersedia secara online, dengan rata-rata lebih dari 10.000 game baru dirilis setiap tahunnya. Kondisi ini membuat proses pencarian game yang relevan menjadi semakin kompleks dan memakan waktu. Oleh karena itu, dibutuhkan sistem rekomendasi yang cerdas dan efisien untuk membantu pengguna dalam menyaring dan menemukan game yang sesuai dengan selera mereka.

### Mengapa Masalah Ini Penting dan Bagaimana Cara Menyelesaikannya
Sistem rekomendasi yang baik tidak hanya meningkatkan kenyamanan dan pengalaman pengguna, tetapi juga memiliki dampak langsung terhadap tingkat kepuasan dan loyalitas pemain terhadap platform. Selain itu, sistem ini juga dapat mendorong peningkatan penjualan game dengan menampilkan rekomendasi yang relevan dan menarik bagi setiap pengguna.

Proyek ini bertujuan untuk membangun sistem rekomendasi game berbasis Machine Learning dengan menggunakan dua pendekatan utama: Content-Based Filtering dan Collaborative Filtering. Pendekatan Content-Based Filtering akan merekomendasikan game berdasarkan kesamaan atribut dengan game yang disukai pengguna, sementara Collaborative Filtering akan menggunakan pola interaksi antar pengguna dan game untuk memberikan rekomendasi. Dataset yang digunakan mencakup informasi tentang game, pengguna, serta ulasan dan rating yang diberikan oleh pengguna terhadap game.

### Referensi
- Newzoo. (2023). Global games market report 2023 (Free version). https://newzoo.com/resources/trend-reports/newzoo-global-games-market-report-2023-free-version
- Kutalia, M. T. (2020, November 1). Trend Micro CTF 2020 — Keybox writeup. Medium. https://tatocaster.medium.com/trend-micro-ctf-2020-keybox-writeup-cf5a69d2d091

## 2. Business Understanding
### Problem Statements
- Bagaimana sistem dapat merekomendasikan game yang mirip dengan game yang dimainkan pengguna?
- Bagaimana sistem dapat menyarankan game yang mungkin disukai pengguna berdasarkan histori pengguna lain?
  
### Goals
- Membangun sistem rekomendasi game berbasis fitur (content-based)
- Membangun sistem rekomendasi berbasis histori pengguna (collaborative)
- Memberikan interpretasi atas rekomendasi yang diberikan

### Solution Statement

Untuk membangun sistem rekomendasi yang efektif dan adaptif terhadap preferensi pengguna, proyek ini mengadopsi dua pendekatan utama dalam pengembangan sistem rekomendasi, yaitu Content-Based Filtering dan Collaborative Filtering. Kedua pendekatan ini digunakan secara terpisah atau dapat dikombinasikan dalam sistem rekomendasi hybrid untuk meningkatkan akurasi dan relevansi hasil rekomendasi.

#### Pendekatan Content-Based Filtering
Pendekatan Content-Based Filtering bekerja dengan cara menganalisis karakteristik atau fitur dari masing-masing game. Dalam proyek ini, fitur yang digunakan antara lain adalah average rating (rating rata-rata dari pengguna), positive ratio (rasio ulasan positif), serta dukungan terhadap berbagai platform seperti Windows, Mac, dan Linux. Dengan memanfaatkan fitur-fitur ini, sistem akan mencari game lain yang memiliki profil atau karakteristik yang serupa dengan game yang sebelumnya disukai oleh pengguna.

Untuk mengukur tingkat kemiripan antar game, digunakan metode cosine similarity, yaitu teknik pengukuran kesamaan antar dua vektor dalam ruang berdimensi tinggi. Vektor-vektor ini merepresentasikan fitur game yang telah dinormalisasi. Semakin tinggi nilai cosine similarity antara dua game, maka semakin mirip pula kedua game tersebut, sehingga lebih mungkin untuk direkomendasikan kepada pengguna.

#### Pendekatan Collaborative Filtering
Berbeda dengan pendekatan content-based yang berfokus pada fitur game, pendekatan Collaborative Filtering bertumpu pada pola interaksi antara pengguna dan game. Dalam proyek ini, digunakan algoritma Singular Value Decomposition (SVD) yang tersedia melalui library Surprise, sebuah pustaka populer untuk membangun dan mengevaluasi sistem rekomendasi berbasis interaksi pengguna.

Algoritma SVD berfungsi dengan memetakan pengguna dan item (dalam hal ini, game) ke dalam ruang laten berdimensi rendah, untuk menemukan pola tersembunyi dari perilaku pengguna. Dengan cara ini, sistem dapat mengenali preferensi pengguna secara implisit dan membandingkannya dengan preferensi pengguna lain yang memiliki pola interaksi serupa. Hasilnya adalah rekomendasi game yang disesuaikan dengan selera pengguna berdasarkan perilaku kolektif pengguna lain yang mirip.

## 3. Data Understanding
### Dataset
Dataset yang digunakan terdiri dari tiga file utama, masing-masing merepresentasikan entitas yang berbeda dalam sistem rekomendasi game:
- `games.csv` berisi metadata game seperti judul, platform, tanggal rilis, harga, dan rating konten

- `users.csv` menyimpan informasi agregat pengguna, termasuk jumlah produk yang dimiliki dan total ulasan yang ditulis

- `recommendations.csv` merekam interaksi pengguna terhadap game, mencakup informasi seperti apakah game direkomendasikan, lama bermain, serta seberapa membantu atau lucu sebuah ulasan dinilai.

### Sumber Data
dapat diunduh melalui kaggle: https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam

### Jumlah Data Total
Struktur Dataset (dari .info())

```python
print(recommendations.info())
print(games.info())
print(users.info())
```

| Dataset               | Jumlah Baris         |
|-----------------------|----------------------|
| `games.csv`           | 2.544 game           |
| `users.csv`           | 715.303 pengguna     |
| `recommendations.csv` | 2.057.740 review     |

### Fitur Pada Dataset

#### `games.csv`

| Fitur          | Deskripsi                                               |
|----------------|---------------------------------------------------------|
| `app_id`       | ID unik untuk setiap game                               |
| `title`        | Nama/judul game                                         |
| `rating`       | Rating konten game (misal: Teen, Mature)                |
| `positive_ratio` | Persentase ulasan positif dari total review          |
| `win`          | Indikator boolean untuk dukungan platform Windows       |
| `mac`          | Indikator boolean untuk dukungan platform macOS         |
| `linux`        | Indikator boolean untuk dukungan platform Linux         |
| `steam_deck`   | Apakah game kompatibel dengan Steam Deck                |
| `date_release` | Tanggal rilis game                                      |
| `user_reviews` | Jumlah review dari pengguna                             |
| `price_final`  | Harga akhir game setelah diskon                         |
| `price_original` | Harga asli sebelum diskon                            |
| `discount`     | Persentase diskon yang diberikan                        |

#### `users.csv`

| Fitur     | Deskripsi                                       |
|-----------|-------------------------------------------------|
| `user_id` | ID unik pengguna                                |
| `products`| Jumlah game yang dimiliki oleh pengguna         |
| `reviews` | Total ulasan yang ditulis oleh pengguna         |

#### `recommendations.csv`

| Fitur           | Deskripsi                                              |
|------------------|--------------------------------------------------------|
| `user_id`        | ID pengguna yang memberikan review                     |
| `app_id`         | ID game yang direview                                  |
| `is_recommended` | Apakah pengguna merekomendasikan game (boolean)        |
| `hours`          | Lama waktu bermain (jam)                               |
| `helpful`        | Jumlah penilaian helpful terhadap review               |
| `funny`          | Jumlah penilaian lucu terhadap review                  |
| `date`           | Tanggal ulasan dibuat                                  |
| `review_id`      | ID unik untuk masing-masing review                     |

### Exploratory Data Analysis (EDA) + Insight
```python
rating_counts = games['rating'].value_counts()

print("Distribusi Rating Game:")
print(rating_counts)

rating_ratio = games.groupby('rating')['positive_ratio'].mean().sort_values(ascending=False)

print("\nRata-rata Positive Ratio per Rating:")
print(rating_ratio)

platform_support = games[['win', 'mac', 'linux', 'steam_deck']].mean() * 100

print("\nPersentase Dukungan Platform:")
print(platform_support.round(2))

recommend_counts = recommendations['is_recommended'].value_counts(normalize=True) * 100

print("\nDistribusi Review Rekomendasi (%):")
print(recommend_counts.round(2))
```

#### Distribusi rating game 
Distribusi jumlah game berdasarkan kolom `rating`

```
Positive                   664
Very Positive              641
Mixed                      631
Mostly Positive            449
Mostly Negative             97
Overwhelmingly Positive     45
Negative                    12
Very Negative                4
Overwhelmingly Negative      1
```

- Mostly Positive dan Very Positive adalah dua kategori rating terbanyak.
- Hal ini menunjukkan bahwa mayoritas game mendapatkan respons yang baik dari pengguna.

#### Positive Ratio per Rating
Rata-rata nilai `positive_ratio` untuk tiap kategori `rating`:

```
rating
Overwhelmingly Positive    96.200000
Positive                   91.257530
Very Positive              88.728549
Mostly Positive            74.663697
Mixed                      57.711569
Mostly Negative            31.030928
Overwhelmingly Negative    13.000000
Negative                   12.416667
Very Negative              11.500000
Name: positive_ratio, dtype: float64
```
- Game dengan rating Overwhelmingly Positive dan Very Positive memiliki rasio ulasan positif > 89%.
- Sementara itu, game dengan rating Negative dan Overwhelmingly Negative rata-ratanya hanya di kisaran 23–34%.

#### Dukungan Platform
Persentase game yang mendukung masing-masing platform:

```
win           98.00
mac           26.18
linux         18.47
steam_deck    99.96
dtype: float64
```

Dalam bentuk persentase:
- Windows: 98.00%
- macOS: 26.18%
- Linux: 18.47%
- Steam Deck: 99.96%

Artinya hampir semua game mendukung Windows, sementara dukungan untuk Linux dan macOS masih terbatas. Dukungan Steam Deck cukup signifikan.

#### Distribusi Review yang Direkomendasikan

Distribusi `is_recommended` pada data:

```
True     85.8
False    14.2
Name: proportion, dtype: float64
```

Persentase:

- Direkomendasikan (True): 85,8%
- Tidak direkomendasikan (False): 14.2%

menunjukkan kecenderungan positif dari komunitas pengguna.

## 4. Data Preparation
### Konversi Tipe Data
Proses:
- Kolom tanggal dikonversi ke format `datetime` agar bisa diproses sebagai waktu.
- Kolom-kolom biner (`win`,`mac`, `linux`, `steam_deck`, dan `is_recommended`) dikonversi ke tipe `bool`.

```python
recommendations['date'] = pd.to_datetime(recommendations['date'])
games['date_release'] = pd.to_datetime(games['date_release'])

games[['win', 'mac', 'linux', 'steam_deck']] = games[['win', 'mac', 'linux', 'steam_deck']].astype(bool)
recommendations['is_recommended'] = recommendations['is_recommended'].astype(bool)
```

Hal ini dilakukan agar tipe data sesuai dengan konteks analisis dan memudahkan proses manipulasi, filtering, dan visualisasi.

### Penggabungan Dataset (Merge)
Proses:
- Dataset recommendations digabungkan dengan dataset `games` berdasarkan `app_id`.
- Kemudian hasilnya digabungkan lagi dengan dataset `users` berdasarkan `user_id`.

```python
merged_df = recommendations.merge(games, on='app_id', how='left')
merged_df = merged_df.merge(users, on='user_id', how='left')
```

Proses ini dilakukan untuk mendapatkan data yang lengkap dari berbagai sumber (rekomendasi, game, dan pengguna) dalam satu dataset yang dapat dianalisis secara menyeluruh.

### Menghapus Nilai Kosong pada Kolom Penting (Drop NA)

Proses dilakukan dengan menghapus baris-baris yang memiliki nilai kosong pada kolom yang dianggap penting, yaitu `title`, `date_release`, `price_final`, `positive_ratio`, `products`, dan `reviews`.

```python
important_columns = ['title', 'date_release', 'price_final', 'positive_ratio', 'products', 'reviews']
cleaned_df = merged_df.dropna(subset=important_columns)
```

Nilai kosong pada kolom ini bisa menyebabkan bias, error dalam model, atau hasil yang tidak valid ketika dilakukan analisis atau pemodelan.

### Menyimpan Dataset Bersih

Dataset yang sudah dibersihkan disimpan ke dalam file CSV agar bisa digunakan kembali pada proses modeling berikutnya tanpa perlu melakukan pembersihan ulan

```python
cleaned_df.to_csv('cleaned_recommendations.csv', index=False)
```

Dengan menyimpan versi final dalam bentuk pdf dapat meningkatkan efisiensi karena dataset siap digunakan untuk analisis lanjutan atau pemodelan seperti content-based filtering.

## 5. Modelling

### Content-Based Filtering
Pendekatan Content-Based Filtering digunakan untuk merekomendasikan game berdasarkan kemiripan fitur antar game, bukan dari preferensi pengguna lain. Pendekatan ini cocok untuk menghadapi masalah cold-start (misalnya, ketika pengguna baru belum memiliki cukup interaksi).

####  Fitur yang Digunakan
Rekomendasi dihitung berdasarkan kemiripan dari fitur-fitur game berikut:
- `positive_ratio` Rasio ulasan positif
- `rating` Rating konten (misalnya: "Everyone", "Mature", dll)

dan juga dukungan platform:
- `win`: Windows
- `mac`: macOS
- `linux`: Linux
- `steam_deck`: Steam Deck

####  Pra-pemrosesan Data
- Mengisi nilai kosong pada kolom `rating` dengan `"Unknown"`
- Mengkodekan kategori `rating` ke bentuk numerik dengan `category.codes`
- Normalisasi seluruh fitur numerik ke rentang 0-1 menggunakan `MinMaxScaler`

```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

content_features = games[['rating', 'positive_ratio', 'win', 'mac', 'linux', 'steam_deck']].copy()

content_features['rating'] = content_features['rating'].fillna("Unknown")
content_features['rating_encoded'] = content_features['rating'].astype('category').cat.codes

feature_matrix = content_features[['rating_encoded', 'positive_ratio', 'win', 'mac', 'linux', 'steam_deck']]
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(feature_matrix)
```
#### Perhitungan Kemiripan Game
Setelah fitur dinormalisasi, dihitung matriks kemiripan antar game menggunakan cosine similarity. Game yang mirip akan memiliki skor kemiripan mendekati 1

```python
similarity_matrix = cosine_similarity(scaled_features)
```

#### Fungsi Rekomendasi `recommend_similar_games(game_id)`

Fungsi ini mencari game yang paling mirip dengan game yang diberikan berdasarkan skor kemiripan tertinggi (dalam `similarity_matrix`), dan mengembalikan `top_n` game serupa

```python
def recommend_similar_games(game_id, top_n=5):
    try:
        idx = games[games['app_id'] == game_id].index[0]
        similarity_scores = list(enumerate(similarity_matrix[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        game_indices = [i[0] for i in similarity_scores]
        return games.iloc[game_indices][['app_id', 'title']]
    except:
        return "Game ID not found."
```

#### Contoh Output Rekomendasi dengan Content-Based Filtering

```python
print("Rekomendasi mirip dengan 'Super Blackjack Battle 2 Turbo Edition - The Card Warriors':")
print(recommend_similar_games(545200))
```

Output (Top-N):
```
Rekomendasi mirip dengan 'Super Blackjack Battle 2 Turbo Edition - The Card Warriors':
      app_id                                              title
18   1146320                      GRID Ultimate Edition Upgrade
85   2471820                                    СТРАШНО И ТОЧКА
360  1637251  Train Simulator: Southwestern Expressways: Rea...
409  2169810                                            Mermaid
428   657590                                        Grav Blazer
```

#### Evaluasi dengan Precision@5

Untuk mengevaluasi seberapa relevan hasil rekomendasi, digunakan metrik Precision@5, yang mengukur seberapa banyak rekomendasi yang benar-benar disukai pengguna dari 5 rekomendasi yang diberikan.

```python
def precision_at_k(user_game_df, game_features_df, k=5):
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

        hit_count = len(recommended_game_ids & liked_game_ids)
        hits += hit_count
        total += k

    precision = hits / total if total > 0 else 0
    return precision

precision = precision_at_k(cleaned_df, games, k=5)
print(f"\nPrecision@5 dari model Content-Based Filtering: {precision:.4f}")
```

Output Evaluasi:
```
Precision@5 dari model Content-Based Filtering: 0.1011
```

### Collaborative Filtering dengan SVD
Pendekatan Collaborative Filtering digunakan untuk memberikan rekomendasi game berdasarkan preferensi pengguna lain yang memiliki pola serupa. Dalam proyek ini, digunakan algoritma SVD (Singular Value Decomposition) dari pustaka `Surprise`.

#### Persiapan Data
Dataset yang digunakan memiliki format interaksi pengguna terhadap game, dengan tiga kolom penting:
- user_id: ID pengguna
- app_id: ID game
- is_recommended: Label biner (0 = tidak direkomendasikan, 1 = direkomendasikan)

```python
cf_data = recommendations[['user_id', 'app_id', 'is_recommended']].copy()
cf_data['is_recommended'] = cf_data['is_recommended'].astype(int)
```

#### Evaluasi Model dengan Cross-Validation
Model dilatih menggunakan evaluasi 3-fold cross-validation untuk mengukur performa dengan metrik:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)

```python
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(cf_data[['user_id', 'app_id', 'is_recommended']], reader)

model = SVD()
cross_validate(model, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
```

Output evaluasi:
```python
Evaluating RMSE, MAE of algorithm SVD on 3 split(s).

                  Fold 1  Fold 2  Fold 3  Mean    Std     
RMSE (testset)    0.3294  0.3295  0.3295  0.3295  0.0000  
MAE (testset)     0.2139  0.2137  0.2137  0.2137  0.0001  
Fit time          29.64   29.53   31.31   30.16   0.81    
Test time         5.35    4.91    4.34    4.87    0.41
```

#### Melatih Model Final
Setelah evaluasi, model dilatih ulang menggunakan seluruh data untuk menghasilkan model akhir:

```python
trainset = data.build_full_trainset()
model.fit(trainset)
```

#### Fungsi rekomendasi (`recommend_for_user(user_id)`)
Fungsi ini digunakan untuk memberikan rekomendasi game personalisasi kepada pengguna tertentu berdasarkan estimasi preferensi tertinggi.

```python
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
    recommended_games = games[games['app_id'].isin(top_app_ids)][['app_id', 'title']]
    return recommended_games
```

####  Contoh Output Rekomendasi dengan Collaborative Filtering
```python
print("Rekomendasi game untuk user_id = 253880:")
print(recommend_for_user(253880))
```

Output (Top-N):
```
Rekomendasi game untuk user_id = 253880:
      app_id                       title
35    498570       Extreme Forklifting 2
65    554310                   Rage Wars
100  1347030                THE CORRIDOR
114       20       Team Fortress Classic
120    97110  Kohan: Immortal Sovereigns
```

#### Perbandingan Pendekatan

| Pendekatan                  | Kelebihan                                                                 | Kekurangan                                                                 |
|----------------------------|---------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| **Content-Based Filtering** | - Cocok untuk *cold-start user* (pengguna baru)  <br> - Tidak tergantung pada data pengguna lain | - Rekomendasi bisa monoton (mirip terus) <br> - Sulit menangkap selera kompleks pengguna |
| **Collaborative Filtering (SVD)** | - Dapat menangkap pola kompleks dari interaksi pengguna <br> - Rekomendasi lebih personal dan beragam | - Tidak cocok untuk pengguna baru (cold-start) <br> - Perlu data interaksi yang cukup banyak |


## 6. Evaluasi

### Evaluasi Model dengan Content-Based Filtering

#### Output Rekomendasi

```python
print("\nRekomendasi mirip dengan 'Super Blackjack Battle 2 Turbo Edition - The Card Warriors':")
print(recommend_similar_games(545200))
```
output:
```
Rekomendasi mirip dengan 'Super Blackjack Battle 2 Turbo Edition - The Card Warriors':
      app_id                                              title
18   1146320                      GRID Ultimate Edition Upgrade
85   2471820                                    СТРАШНО И ТОЧКА
360  1637251  Train Simulator: Southwestern Expressways: Rea...
409  2169810                                            Mermaid
428   657590                                        Grav Blazer
```

#### Hasil Evaluasinya

```python
def precision_at_k(user_game_df, game_features_df, k=5):
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

        hit_count = len(recommended_game_ids & liked_game_ids)
        hits += hit_count
        total += k

    precision = hits / total if total > 0 else 0
    return precision

precision = precision_at_k(cleaned_df, games, k=5)
print(f"\nPrecision@5 dari model Content-Based Filtering: {precision:.4f}")
```

Output Evaluasi:
```
Precision@5 dari model Content-Based Filtering: 0.1011
```

### Evaluasi Model Collaborative Filtering

#### Hasil evaluasinya

```
Evaluating RMSE, MAE of algorithm SVD on 3 split(s).

                  Fold 1  Fold 2  Fold 3  Mean    Std     
RMSE (testset)    0.3294  0.3295  0.3295  0.3295  0.0000  
MAE (testset)     0.2139  0.2137  0.2137  0.2137  0.0001  
Fit time          29.64   29.53   31.31   30.16   0.81    
Test time         5.35    4.91    4.34    4.87    0.41   
```

#### Output Rekomendasi

```python
print("\nRekomendasi game untuk user_id = 253880:")
print(recommend_for_user(253880))
```
output:
```
Rekomendasi game untuk user_id = 253880:
      app_id                       title
35    498570       Extreme Forklifting 2
65    554310                   Rage Wars
100  1347030                THE CORRIDOR
114       20       Team Fortress Classic
120    97110  Kohan: Immortal Sovereigns
```

jika user tidak ditemukan maka akan muncul pesan berikut:

```python
User ID <id> tidak ditemukan dalam data pelatihan.
```

#### Analisis hasil Output

Content-Based Filtering
- Output dari fungsi `recommend_similar_games(545200)` menunjukkan lima game yang direkomendasikan karena memiliki kemiripan fitur dengan game 'Super Blackjack Battle 2 Turbo Edition - The Card Warriors':

```
Rekomendasi mirip dengan 'Super Blackjack Battle 2 Turbo Edition - The Card Warriors':
      app_id                                              title
18   1146320                      GRID Ultimate Edition Upgrade
85   2471820                                    СТРАШНО И ТОЧКА
360  1637251  Train Simulator: Southwestern Expressways: Rea...
409  2169810                                            Mermaid
428   657590                                        Grav Blazer
```

- Ini menunjukkan bahwa sistem berhasil mengidentifikasi game dengan atribut yang serupa, positive_ratio, rating, dan dukungan platform (win/mac/linux/steam_deck), dan mengurutkannya berdasarkan skor kemiripan (cosine similarity).

```
Precision@5 dari model Content-Based Filtering: 0.1011
```

- Artinya, dari setiap 5 game yang direkomendasikan, rata-rata hanya sekitar 0.5 game (10.11%) yang benar-benar relevan atau disukai pengguna, menunjukkan bahwa akurasi model masih cukup rendah

- akurasi yang cukup rendah ini disebabkan data menggunakan positive_ratio, rating, dan dukungan platform (win/mac/linux/steam_deck) yang kurang mewakili keunikan tiap game, terutama genre, gameplay, atau visual style yang biasa digunakan sebagai tolak ukur hubungan antara user dan game

- Meskipun content-based cocok untuk cold-start user, namun tidak memperhitungkan konteks preferensi pengguna secara personal (misalnya, genre favorit atau jam bermain)

- solusi termudah untuk meningkatkan akurasi ini yaitu menaambahkan Fitur tambahan seperti genre (diolah dengan TF-IDF atau one-hot encoding) serta tags atau categories

Collaborative Filtering
- Evaluasi Kinerja Model SVD, hasil evaluasi 3-fold cross-validation:

```
                  Fold 1  Fold 2  Fold 3  Mean    Std     
RMSE (testset)    0.3294  0.3295  0.3295  0.3295  0.0000  
MAE (testset)     0.2139  0.2137  0.2137  0.2137  0.0001  
Fit time          29.64   29.53   31.31   30.16   0.81    
Test time         5.35    4.91    4.34    4.87    0.41  
```

Nilai error yang rendah menunjukkan bahwa model SVD mampu memprediksi rating pengguna terhadap game dengan cukup akurat dan konsisten di setiap fold.

- Output Rekomendasi untuk `user_id = 253880`

```
Rekomendasi game untuk user_id = 253880:
      app_id                       title
35    498570       Extreme Forklifting 2
65    554310                   Rage Wars
100  1347030                THE CORRIDOR
114       20       Team Fortress Classic
120    97110  Kohan: Immortal Sovereigns
```

- Daftar ini menunjukkan game yang diprediksi disukai oleh pengguna berdasarkan pola historis rating dari pengguna lain yang mirip

Handling Cold-Start
- Jika `user_id` tidak ditemukan di data pelatihan, sistem menampilkan pesan, "User ID <id> tidak ditemukan dalam data pelatihan."
- menandakan bahwa sistem memiliki mekanisme penanganan yang jelas untuk pengguna baru atau tidak dikenal, memberikan transparansi pada hasil rekomendasi

Hybrid Recommender System
- Model Content-Based Filtering dan Collaborative Filtering dapat dikembangkan lebih lanjut dan digabungkan menjadi sebuah Hybrid Recommender System. Pendekatan ini memungkinkan peningkatan dari berbagai aspek, seperti sudut pandang rekomendasi yang lebih komprehensif, cakupan (coverage) yang lebih luas, serta fleksibilitas yang lebih tinggi dalam menangani berbagai kondisi

#### Kesimpulan Akhir

*Sistem berhasil merekomendasikan game yang mirip berdasarkan fitur game itu sendiri*
- Dengan menggunakan pendekatan Content-Based Filtering, sistem dapat merekomendasikan game berdasarkan kemiripan atribut seperti rating konten, rasio ulasan positif, dan dukungan platform. Proses normalisasi fitur serta penerapan cosine similarity terbukti efektif dalam mengukur tingkat kemiripan antar game. tapi untuk keakuratan sesuai preferensi pengguna masih perlu ditingkatkan dengan menambahkan fitur pembanding baru seperti genre atau categories. Walaupun begitu, Hal ini menjawab problem statement pertama, yaitu:

```
Bagaimana sistem dapat merekomendasikan game yang mirip dengan game yang dimainkan pengguna?
```

*Sistem berhasil menyarankan game berdasarkan histori interaksi pengguna lain*
- Pendekatan Collaborative Filtering berbasis algoritma SVD berhasil dibangun dan diuji menggunakan cross-validation. Hasil evaluasi menunjukkan performa yang stabil dan akurat, dengan nilai RMSE rendah (~0.33) dan MAE (~0.21). Fungsi rekomendasi untuk pengguna berhasil memberikan rekomendasi yang sesuai berdasarkan pola preferensi kolektif. Ini menjawab problem statement kedua:

```
Bagaimana sistem dapat menyarankan game yang mungkin disukai pengguna berdasarkan histori pengguna lain?
```

Menyediakan hasil rekomendasi yang dapat diinterpretasikan*
Sistem menyediakan hasil rekomendasi yang dapat diinterpretasikan secara langsung oleh pengguna. Pada pendekatan Content-Based Filtering, rekomendasi diberikan bersama dengan nama game yang mirip secara fitur, sehingga pengguna dapat memahami alasan di balik saran yang diberikan. Sementara pada Collaborative Filtering, hasil evaluasi seperti RMSE dan MAE yang rendah menunjukkan bahwa prediksi preferensi pengguna dapat dipercaya, dan daftar game yang direkomendasikan ditampilkan secara eksplisit. Selain itu, sistem menangani kasus pengguna baru dengan pesan yang informatif, sehingga mudah dipahami.

*Tujuan utama proyek telah tercapai, yaitu:*

- Membangun sistem rekomendasi game berbasis fitur (content-based)
- Membangun sistem rekomendasi berbasis histori pengguna (collaborative filtering)
- Menyediakan hasil rekomendasi yang dapat diinterpretasikan

*Peluang pengembangan lanjutan*
- Kedua pendekatan dapat dikombinasikan menjadi Hybrid Recommender System untuk meningkatkan cakupan rekomendasi, relevansi prediksi, dan kemampuan menangani berbagai skenario seperti pengguna baru (cold-start), game baru, atau data yang terbatas.

Dengan demikian, sistem rekomendasi game yang dibangun dalam proyek ini telah berhasil memenuhi tujuan dan menjawab permasalahan yang diangkat di awal studi, serta memberikan fondasi yang kuat untuk pengembangan lebih lanjut.







