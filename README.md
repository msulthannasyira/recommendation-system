# predictive-analysis-crypto-bitcoin
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
- Newzoo. (2023). Global Games Market Report 2023. Retrieved from https://newzoo.com/resources/trend-reports/newzoo-global-games-market-report-2023-free-version
- Ricci, F., Rokach, L., & Shapira, B. (2022). Recommender Systems Handbook (3rd ed.). Springer. DOI: 10.1007/978-1-0716-2197-4

## 2. Business Understanding
### Problem Statements
- Bagaimana sistem dapat merekomendasikan game yang mirip dengan game yang dimainkan pengguna?
- Bagaimana sistem dapat menyarankan game yang mungkin disukai pengguna berdasarkan histori pengguna lain?
  
### Goals
- Membangun sistem rekomendasi game berbasis fitur (content-based)
- Membangun sistem rekomendasi berbasis histori pengguna (collaborative)
- Memberikan visualisasi dan interpretasi atas rekomendasi yang diberikan

### Solution Statement

Untuk membangun sistem rekomendasi yang efektif dan adaptif terhadap preferensi pengguna, proyek ini mengadopsi dua pendekatan utama dalam pengembangan sistem rekomendasi, yaitu Content-Based Filtering dan Collaborative Filtering. Kedua pendekatan ini digunakan secara terpisah dan dapat dikombinasikan dalam sistem rekomendasi hybrid untuk meningkatkan akurasi dan relevansi hasil rekomendasi.

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

### Jumlah Data Total

| Dataset               | Jumlah Baris         |
|-----------------------|----------------------|
| `games.csv`           | 2.544 game           |
| `users.csv`           | 715.303 pengguna     |
| `recommendations.csv` | 2.057.740 review     |

### Fitur Penting

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



