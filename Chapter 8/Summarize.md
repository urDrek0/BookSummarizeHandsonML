# Bab 8: Reduksi Dimensi

Bab ini akan membahas:
* Masalah "kutukan dimensi" (*curse of dimensionality*) yang terjadi pada data berdimensi tinggi.
* Dua pendekatan utama untuk reduksi dimensi: Proyeksi dan Manifold Learning.
* Algoritma reduksi dimensi yang paling populer, **PCA (Principal Component Analysis)**, beserta variannya.
* Teknik non-linear seperti **Kernel PCA (kPCA)** dan **LLE (Locally Linear Embedding)**.
* Kapan dan mengapa menggunakan teknik-teknik ini, baik untuk visualisasi maupun untuk *preprocessing*.

---

## 1. Kutukan Dimensi (The Curse of Dimensionality)

Banyak hal bekerja sangat berbeda di ruang berdimensi tinggi. Masalah utama yang disebut "kutukan dimensi" adalah data di ruang berdimensi tinggi cenderung sangat **sparse** (jarang)[cite: 215].

* Instans latih kemungkinan besar akan berjauhan satu sama lain, membuat prediksi menjadi kurang andal karena didasarkan pada ekstrapolasi yang besar.
* Jumlah data latih yang dibutuhkan tumbuh secara eksponensial seiring dengan jumlah dimensi.
* Ini meningkatkan risiko **overfitting** secara signifikan[cite: 215].

Reduksi dimensi bertujuan untuk mengatasi kutukan ini. Manfaat utamanya adalah[cite: 213, 1118]:
1.  **Mempercepat** algoritma pelatihan (yang paling utama).
2.  **Memvisualisasikan data** (DataViz) dengan menguranginya menjadi 2 atau 3 dimensi.
3.  **Menghemat ruang** (kompresi).

---

## 2. Pendekatan Utama Reduksi Dimensi

### Proyeksi (Projection)
Pada kebanyakan masalah dunia nyata, data tidak tersebar merata. Data seringkali terletak pada (atau sangat dekat dengan) *subspace* berdimensi jauh lebih rendah[cite: 215, 1121]. Pendekatan ini bekerja dengan **memproyeksikan** setiap instans data ke *subspace* (hiperplane) berdimensi lebih rendah tersebut[cite: 1121].

* **Kelemahan**: Pendekatan ini tidak bekerja dengan baik untuk *manifold* (struktur data) yang "terpelintir", seperti dataset "Swiss roll"[cite: 1122, 1123]. Memproyeksikan Swiss roll hanya akan menindih lapisannya.

### Manifold Learning
Manifold Learning adalah pendekatan yang mengasumsikan bahwa data berdimensi tinggi Anda terletak pada *d-dimensional manifold* (bentuk berdimensi $d$ yang tertekuk di ruang berdimensi $n$ yang lebih tinggi)[cite: 218].

* **Manifold Assumption**: Hipotesis bahwa sebagian besar dataset dunia nyata terletak dekat dengan manifold berdimensi rendah[cite: 218].
* **Tujuan**: Algoritma ini mencoba untuk "membuka" atau "mengurai" *manifold* tersebut (misalnya, mengubah Swiss roll menjadi persegi panjang 2D)[cite: 1123].

---

## 3. PCA (Principal Component Analysis)

PCA adalah algoritma reduksi dimensi yang paling populer[cite: 219].

* **Cara Kerja**: PCA mengidentifikasi *hyperplane* yang paling dekat dengan data, dan kemudian memproyeksikan data ke *hyperplane* tersebut[cite: 219].
* **Mempertahankan Varians**: PCA memilih sumbu (disebut **principal components** / komponen utama) yang **mempertahankan jumlah varians maksimum** dalam data[cite: 219, 1125]. Sumbu pertama (PC1) adalah sumbu yang menyumbang varians paling besar, PC2 adalah sumbu kedua yang ortogonal terhadap PC1 dan menyumbang sisa varians terbesar, dan seterusnya.
* **Menemukan PC**: PC dapat ditemukan menggunakan teknik dekomposisi matriks standar yang disebut **Singular Value Decomposition (SVD)**[cite: 221, 1126].

### Explained Variance Ratio
Untuk setiap komponen utama, Anda dapat mengakses **explained variance ratio** (rasio varians yang dijelaskan), yang menunjukkan proporsi varians dataset yang terletak di sepanjang sumbu tersebut[cite: 222].

### Memilih Jumlah Dimensi yang Tepat
Daripada memilih jumlah dimensi ($d$) secara sewenang-wenang, Anda dapat:
1.  Memilih $d$ yang mempertahankan proporsi varians yang cukup besar (misalnya, 95%)[cite: 223].
2.  Memplot total varians yang dijelaskan sebagai fungsi dari jumlah dimensi dan mencari "siku" (elbow) pada kurva, di mana penambahan dimensi baru tidak lagi memberikan banyak varians tambahan[cite: 223].

### PCA untuk Kompresi
PCA sangat berguna untuk kompresi. Setelah mengurangi dimensi, Anda dapat mengembalikannya ke dimensi asli menggunakan `inverse_transform()`. Proses ini akan kehilangan beberapa informasi (karena varians yang dihilangkan), tetapi hasilnya akan sangat dekat dengan data asli. Perbedaan antara data asli dan data yang direkonstruksi disebut **reconstruction error**[cite: 224].

### Varian PCA
* **Incremental PCA (IPCA)**: Berguna untuk dataset besar yang tidak muat di memori. IPCA membagi dataset menjadi *mini-batch* dan "memberi makan" *batch* tersebut satu per satu ke algoritma [cite: 225-226].
* **Randomized PCA**: Algoritma stokastik yang menemukan perkiraan (aproksimasi) $d$ komponen utama pertama dengan jauh lebih cepat daripada SVD penuh. Sangat efisien ketika $d$ jauh lebih kecil dari jumlah fitur asli[cite: 225].

---

## 4. Kernel PCA (kPCA)

**Kernel trick** (dibahas di Bab 5) dapat diterapkan pada PCA. **Kernel PCA (kPCA)** memungkinkan dilakukannya proyeksi non-linear yang kompleks untuk reduksi dimensi[cite: 226].

* **Cara Kerja**: kPCA secara implisit memetakan data ke ruang fitur berdimensi sangat tinggi (feature space) dan kemudian menggunakan PCA linear di ruang tersebut[cite: 228].
* **Kegunaan**: Sangat baik untuk mempertahankan *cluster* instans setelah proyeksi atau untuk "membuka gulungan" *manifold* yang terpelintir[cite: 227].
* **Memilih Kernel**: Karena kPCA adalah *unsupervised*, tidak ada metrik performa yang jelas.
    1.  Gunakan *Grid Search* untuk memilih kernel dan *hyperparameter* yang memberikan performa terbaik pada tugas *supervised* (misalnya klasifikasi) **setelah** reduksi dimensi[cite: 227].
    2.  Pilih kernel yang menghasilkan *reconstruction pre-image error* terendah[cite: 228].

---

## 5. LLE (Locally Linear Embedding)

LLE adalah teknik **Manifold Learning** non-linear yang kuat dan tidak bergantung pada proyeksi[cite: 230].
* **Cara Kerja**:
    1.  Untuk setiap instans, LLE mengukur bagaimana ia berhubungan secara linear dengan tetangga terdekatnya (*closest neighbors* / c.n.)[cite: 230].
    2.  Kemudian, ia mencari representasi data berdimensi rendah di mana hubungan *lokal* ini paling dapat dipertahankan[cite: 230, 1136].
* **Kegunaan**: Sangat baik dalam membuka *manifold* yang sangat terpelintir (seperti Swiss roll) ketika tidak ada banyak *noise*[cite: 230].

---

# KESIMPULAN AKHIR

* Reduksi dimensi adalah alat penting untuk melawan **"kutukan dimensi"**, yang dapat mempercepat pelatihan, menghemat ruang, dan membantu visualisasi data.
* **PCA** adalah teknik reduksi dimensi linear yang paling umum, yang bekerja dengan menemukan sumbu yang memaksimalkan varians data.
* **Incremental PCA (IPCA)** dapat menangani dataset yang sangat besar yang tidak muat dalam memori.
* **Kernel PCA (kPCA)** menggunakan *kernel trick* untuk melakukan reduksi dimensi non-linear.
* **LLE** adalah teknik *manifold learning* yang bekerja dengan menemukan representasi berdimensi rendah yang mempertahankan hubungan *lokal* antar instans.
* Teknik lain seperti t-SNE dan LDA juga ada untuk tujuan visualisasi atau klasifikasi.