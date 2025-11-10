# Bab 9: Teknik Unsupervised Learning

Bab ini akan membahas:
* Tugas *unsupervised learning* (pembelajaran tak terawasi) utama: **Clustering** (Pengelompokan).
* Dua algoritma clustering yang populer: **K-Means** (berbasis centroid) dan **DBSCAN** (berbasis kepadatan).
* Aplikasi praktis dari clustering, termasuk *image segmentation* (segmentasi gambar), *preprocessing*, dan *semi-supervised learning* (pembelajaran semi-terawasi).
* **Gaussian Mixture Models (GMM)** sebagai model probabilistik yang kuat untuk *density estimation* (estimasi kepadatan), *clustering*, dan *anomaly detection* (deteksi anomali).

---

## 1. Clustering

Clustering adalah tugas untuk mengelompokkan instans yang mirip ke dalam *clusters* (kelompok). Tidak seperti klasifikasi, clustering adalah tugas *unsupervised* (tidak ada label). Algoritma yang berbeda akan menangkap jenis *cluster* yang berbeda.

Aplikasi utama clustering meliputi:
* **Segmentasi Pelanggan**: Mengelompokkan pelanggan untuk strategi pemasaran yang berbeda.
* **Analisis Data**: Menjalankan clustering terlebih dahulu untuk menganalisis setiap cluster secara terpisah.
* **Reduksi Dimensi**: Mengganti fitur instans dengan afinitasnya terhadap setiap cluster.
* **Deteksi Anomali**: Instans yang memiliki afinitas rendah ke semua cluster kemungkinan adalah anomali.
* **Semi-supervised Learning**: Jika Anda memiliki sedikit data berlabel, Anda dapat menyebarkan (propagate) label ke instans lain dalam cluster yang sama.
* **Segmentasi Gambar**: Mengelompokkan piksel berdasarkan warnanya untuk menyederhanakan gambar.

---

## 2. K-Means

K-Means adalah algoritma sederhana yang efisien untuk mengelompokkan data ke dalam $k$ *cluster*, di mana setiap *cluster* berbentuk bola (spherical).

* **Cara Kerja**:
    1.  Pilih $k$ **centroids** (pusat cluster) secara acak.
    2.  Tetapkan setiap instans ke *centroid* terdekat.
    3.  Perbarui *centroid* menjadi titik tengah (rata-rata) dari semua instans yang ditetapkan padanya.
    4.  Ulangi langkah 2 dan 3 hingga *centroid* berhenti bergerak.
* **Inertia**: Metrik yang digunakan model untuk dievaluasi, yaitu rata-rata kuadrat jarak antara setiap instans dan *centroid* terdekatnya.
* **Masalah Inisialisasi**: Algoritma ini sensitif terhadap inisialisasi *centroid* acak dan dapat konvergen ke solusi suboptimal. Solusinya adalah menjalankan algoritma beberapa kali (`n_init=10`) dengan inisialisasi berbeda dan memilih solusi dengan *inertia* terendah.
* **Menemukan $k$ (Jumlah Cluster)**:
    * **Metode Siku (Elbow Method)**: Plot *inertia* sebagai fungsi dari $k$. Kurva akan terlihat seperti siku. Pilih $k$ di titik "siku" tersebut, karena penambahan cluster setelah titik itu tidak memberikan penurunan *inertia* yang signifikan.
    * **Skor Silhouette (Silhouette Score)**: Metrik yang lebih baik yang mengukur seberapa baik sebuah instans "pas" di dalam clusternya dibandingkan dengan cluster tetangga. Skor berkisar dari -1 (ditempatkan di cluster yang salah) hingga +1 (sangat padat dan jauh dari cluster lain). Anda dapat memplot skor silhouette sebagai fungsi dari $k$.

### Batasan K-Means
K-Means cepat dan scalable, tetapi memiliki keterbatasan. Algoritma ini kesulitan jika:
* Cluster memiliki **ukuran yang sangat berbeda**.
* Cluster memiliki **kepadatan yang berbeda**.
* Cluster memiliki **bentuk non-spherical** (bukan bola), misalnya bentuk lonjong atau memanjang.

---

## 3. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

DBSCAN adalah algoritma clustering yang melihat *cluster* sebagai **wilayah padat (dense regions)** yang dipisahkan oleh wilayah dengan kepadatan rendah.

* **Cara Kerja**:
    1.  Untuk setiap instans, algoritma menghitung berapa banyak instans yang berada dalam jarak $\epsilon$ (*epsilon*) darinya.
    2.  Jika sebuah instans memiliki setidaknya `min_samples` tetangga dalam radius $\epsilon$, ia ditandai sebagai **core instance**.
    3.  Semua instans dalam radius $\epsilon$ dari *core instance* termasuk dalam cluster yang sama.
    4.  Setiap instans yang bukan *core instance* dan tidak memiliki *core instance* sebagai tetangga dianggap sebagai **anomali** (noise).
* **Kelebihan**:
    * Dapat menemukan *cluster* dengan **bentuk apa pun**.
    * Sangat **robust terhadap *outliers*** (anomali) dan hanya memiliki dua *hyperparameter* (`eps` dan `min_samples`).
* **Kekurangan**: Tidak dapat menangani dengan baik jika *cluster* memiliki kepadatan yang sangat bervariasi.

---

## 4. Gaussian Mixture Models (GMM)

GMM adalah model probabilistik yang mengasumsikan bahwa data dihasilkan dari campuran beberapa **distribusi Gaussian** (berbentuk elips/lonjong).

* **Cara Kerja**: GMM adalah generalisasi dari K-Means. Alih-alih menetapkan instans secara "keras" ke satu cluster, GMM menggunakan algoritma **Expectation–Maximization (EM)** untuk mengestimasi probabilitas ("lunak") instans tersebut milik setiap cluster.
* **Kelebihan**: GMM dapat menangani *cluster* dengan berbagai bentuk elips, ukuran, dan orientasi, tidak seperti K-Means yang terbatas pada bola.
* **Deteksi Anomali**: GMM juga merupakan *generative model*, yang berarti ia dapat mengestimasi *probability density function* (PDF) dari data. Anda dapat menggunakan ini untuk **deteksi anomali** dengan menetapkan ambang batas kepadatan (density threshold). Instans yang berada di wilayah berdensitas rendah dianggap sebagai anomali.
* **Memilih $k$ (Jumlah Cluster)**:
    * Karena GMM adalah model probabilistik, kita dapat menggunakan kriteria informasi teoretis untuk memilih $k$ (dan bentuk *covariance*) terbaik.
    * **BIC (Bayesian Information Criterion)** atau **AIC (Akaike Information Criterion)**: Model terbaik adalah yang memiliki skor BIC atau AIC terendah.
* **Bayesian GMM**: Varian GMM yang dapat secara **otomatis menemukan jumlah cluster yang optimal** dengan memberikan bobot nol (atau mendekati nol) pada cluster yang tidak diperlukan.

---

# KESIMPULAN AKHIR

* Bab ini mengeksplorasi *unsupervised learning*, dengan fokus pada *clustering*.
* **K-Means** adalah algoritma yang cepat dan sederhana, tetapi terbatas pada *cluster* berbentuk bola (spherical).
* **DBSCAN** adalah algoritma berbasis kepadatan yang dapat menemukan *cluster* dengan bentuk arbitrer dan sangat baik dalam mendeteksi *outlier* (anomali).
* **Gaussian Mixture Models (GMM)** adalah model probabilistik yang fleksibel, mengasumsikan *cluster* berbentuk elips (Gaussian). Model ini berguna tidak hanya untuk *clustering* tetapi juga untuk **estimasi kepadatan** dan **deteksi anomali**.
* Menemukan jumlah *cluster* ($k$) yang tepat adalah tantangan. Teknik seperti **Elbow Method**, **Silhouette Score** (untuk K-Means), dan **BIC/AIC** (untuk GMM) dapat digunakan untuk memilih $k$ secara objektif.