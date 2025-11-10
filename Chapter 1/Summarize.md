# Bab 1: Peta Jalan Machine Learning

Bab ini akan membahas:
* Definisi fundamental dari Machine Learning (ML) dan mengapa ML penting.
* Jenis-jenis utama sistem ML, dikategorikan berdasarkan cara belajarnya.
* Tantangan utama yang dihadapi dalam proyek ML, mulai dari data hingga algoritma.
* Alur kerja umum proyek ML dan metodologi penting untuk menguji dan memvalidasi model.

---

## Apa Itu Machine Learning?

Machine Learning (ML) adalah bidang studi yang memberi komputer kemampuan untuk belajar dari data tanpa diprogram secara eksplisit. Sebuah sistem ML belajar dari *experience* (E) terkait suatu *task* (T) dan diukur dengan *performance measure* (P). Sistem ini dikatakan belajar jika performanya (P) pada tugas (T) meningkat seiring dengan bertambahnya pengalaman (E).

Sebagai contoh, filter spam adalah sistem ML:
* **Tugas (T)**: Menandai email baru sebagai spam atau bukan.
* **Pengalaman (E)**: Data latih (*training set*) berisi email contoh beserta labelnya (spam/bukan spam).
* **Performa (P)**: Rasio email yang diklasifikasikan dengan benar (disebut **akurasi**).

### Mengapa Menggunakan Machine Learning?

Pendekatan ML berbeda dari pendekatan pemrograman tradisional.

* **Pendekatan Tradisional**: Anda mempelajari masalah, menulis aturan-aturan eksplisit, mengevaluasi, dan mengulanginya. Jika masalahnya kompleks (seperti filter spam), Anda akan berakhir dengan daftar aturan yang panjang dan rumit yang sulit dipelihara.
* **Pendekatan Machine Learning**: Sistem ML secara otomatis mempelajari pola dari data. Untuk filter spam, ia belajar kata-kata atau frasa mana yang menjadi prediktor kuat spam. Solusi ini lebih singkat, lebih mudah dipelihara, dan seringkali lebih akurat.

ML sangat berguna untuk:
1.  **Masalah kompleks** yang tidak memiliki solusi algoritmik tradisional.
2.  **Menggantikan daftar aturan** yang panjang dan rumit.
3.  **Lingkungan yang berfluktuasi**, karena sistem ML dapat beradaptasi dengan data baru.
4.  **Mendapatkan wawasan** tentang masalah kompleks melalui *data mining*.

---

## Jenis-Jenis Sistem Machine Learning

Sistem ML dapat dikategorikan berdasarkan beberapa kriteria.

### 1. Pembelajaran Terawasi (Supervised Learning)
Dalam supervised learning, data latih yang Anda berikan ke algoritma menyertakan solusi yang diinginkan, yang disebut **label**.

* **Klasifikasi (Classification)**: Tugas untuk memprediksi label kategori diskret. Contoh: Klasifikasi spam.
* **Regresi (Regression)**: Tugas untuk memprediksi nilai numerik kontinu. Contoh: Memprediksi harga mobil berdasarkan fiturnya.
* **Algoritma utama**: Linear Regression, Support Vector Machines (SVM), Decision Trees, dan Neural Networks .

### 2. Pembelajaran Tak Terawasi (Unsupervised Learning)
Dalam unsupervised learning, data latih yang Anda miliki tidak memiliki label. Sistem mencoba belajar tanpa guru.

* **Clustering**: Mengelompokkan data yang mirip. Contoh: K-Means atau DBSCAN untuk mengelompokkan pengunjung blog.
* **Reduksi Dimensi (Dimensionality Reduction)**: Menyederhanakan data dengan mengurangi jumlah fitur tanpa kehilangan banyak informasi. Contoh: Principal Component Analysis (PCA).
* **Deteksi Anomali (Anomaly Detection)**: Mendeteksi data yang tidak biasa (outlier). Contoh: Mendeteksi transaksi kartu kredit yang curang.
* **Association Rule Learning**: Menemukan hubungan menarik antar atribut dalam data. Contoh: "Orang yang membeli saus barbeku juga cenderung membeli steak".

### 3. Jenis Lainnya (Semisupervised dan Reinforcement)

* **Semisupervised Learning**: Menangani data yang sebagian besar tidak berlabel, dengan hanya sebagian kecil data yang memiliki label. Contoh: Google Photos mengelompokkan foto orang (unsupervised), lalu Anda hanya perlu memberi satu label per orang (supervised).
* **Reinforcement Learning (RL)**: Sistem ML yang sangat berbeda. **Agen** belajar dengan mengobservasi **lingkungan**, memilih dan melakukan **tindakan**, dan mendapatkan **rewards** (atau *penalties*) sebagai imbalannya. Tujuannya adalah untuk belajar strategi terbaik, yang disebut **policy**, untuk mendapatkan reward terbanyak seiring waktu. Contoh: Robot yang belajar berjalan atau AlphaGo.

### 4. Batch vs. Online Learning

* **Batch Learning (Offline Learning)**: Sistem dilatih menggunakan semua data yang tersedia sekaligus. Jika Anda ingin modelnya mempelajari data baru, Anda harus melatih versi baru dari awal menggunakan set data lengkap. Ini memakan waktu dan sumber daya.
* **Online Learning (Incremental Learning)**: Sistem dilatih secara bertahap dengan memberinya data secara sekuensial, baik satu per satu atau dalam kelompok kecil yang disebut *mini-batches*. Ini bagus untuk data yang datang terus-menerus (seperti harga saham) atau untuk sistem yang perlu beradaptasi cepat. **Learning rate** (laju belajar) adalah parameter penting: jika terlalu tinggi, ia cepat beradaptasi tapi juga cepat lupa data lama; jika terlalu rendah, ia belajar lebih lambat tapi kurang sensitif terhadap noise.

### 5. Instance-Based vs. Model-Based Learning

* **Instance-Based Learning**: Sistem belajar dengan menghafal contoh-contoh latih. Ketika diberi data baru, ia menggunakan *ukuran kesamaan* (similarity measure) untuk membandingkannya dengan data yang dihafal dan membuat prediksi. Contoh: k-Nearest Neighbors.
* **Model-Based Learning**: Sistem membangun sebuah model (representasi) dari data latih, kemudian menggunakan model tersebut untuk membuat prediksi. Ini adalah alur kerja yang umum:
    1.  **Pilih model** (misalnya, model linear).
    2.  **Latih model** dengan data latih. Ini berarti menemukan parameter model (misalnya, $\theta_0$ dan $\theta_1$) yang meminimalkan *cost function* (fungsi biaya).
    3.  **Lakukan inferensi**: Gunakan model yang telah dilatih untuk membuat prediksi pada data baru.

---

## Tantangan Utama Machine Learning

Dua hal utama yang bisa salah adalah "algoritma yang buruk" dan "data yang buruk".

### 1. Kualitas dan Kuantitas Data

* **Kekurangan Data Latih (Insufficient Quantity)**: Sebagian besar algoritma ML membutuhkan banyak data untuk bekerja dengan baik. Bahkan untuk masalah sederhana, ribuan contoh biasanya diperlukan.
* **Data Latih Tidak Representatif (Nonrepresentative)**: Data latih harus mewakili kasus-kasus baru yang ingin Anda generalisasi. Jika sampelnya terlalu kecil, Anda akan mengalami *sampling noise*. Jika metode pengambilan sampelnya cacat, Anda akan mengalami *sampling bias*.
* **Data Berkualitas Buruk (Poor-Quality)**: Jika data latih penuh dengan *errors*, *outliers*, dan *noise*, akan lebih sulit bagi sistem untuk mendeteksi pola yang mendasarinya.
* **Fitur Tidak Relevan (Irrelevant Features)**: Sistem hanya akan mampu belajar jika data latih berisi cukup fitur yang relevan dan tidak terlalu banyak fitur yang tidak relevan. Proses ini disebut **feature engineering**.

### 2. Algoritma (Overfitting dan Underfitting)

* **Overfitting (Overfitting the Training Data)**: Model bekerja sangat baik pada data latih, tetapi tidak dapat menggeneralisasi dengan baik untuk data baru. Ini terjadi ketika model terlalu kompleks relatif terhadap jumlah dan kebisingan data.
    * **Solusi**:
        1.  Menyederhanakan model (pilih parameter lebih sedikit, kurangi jumlah fitur).
        2.  Mengumpulkan lebih banyak data latih.
        3.  Mengurangi noise pada data (misalnya, memperbaiki data).
        4.  Menerapkan **Regularization** (regularisasi), yaitu membatasi model untuk membuatnya lebih sederhana dan mengurangi risiko overfitting.

* **Underfitting (Underfitting the Training Data)**: Model terlalu sederhana untuk mempelajari struktur data. Model ini akan memiliki performa buruk bahkan pada data latih.
    * **Solusi**:
        1.  Memilih model yang lebih kuat (dengan parameter lebih banyak).
        2.  Melakukan *feature engineering* yang lebih baik.
        3.  Mengurangi batasan pada model (misalnya, mengurangi *hyperparameter* regularisasi).

---

## Pengujian dan Validasi

Satu-satunya cara untuk mengetahui seberapa baik model akan menggeneralisasi ke data baru adalah dengan mencobanya pada data baru.

* **Test Set (Set Uji)**: Anda membagi data Anda menjadi *training set* dan *test set*. Anda melatih model menggunakan *training set*, dan mengujinya menggunakan *test set*. Tingkat kesalahan pada data baru disebut **generalization error** (error generalisasi).
* **Hyperparameter Tuning dan Validation Set**: Mengevaluasi model di *test set* dan memilih model terbaik berdasarkan itu adalah ide buruk, karena model tersebut akan dioptimalkan untuk *test set* spesifik tersebut (ini disebut *data snooping bias* ).
* **Solusi**: Gunakan **validation set** (set validasi) atau *development set (dev set)*. Anda melatih beberapa model dengan *hyperparameter* berbeda pada *training set* yang dikurangi (set latih penuh dikurangi set validasi), dan memilih model yang berkinerja terbaik pada *validation set*.
* **Data Mismatch**: Jika ada ketidakcocokan antara data latih dan data validasi/uji (misalnya, gambar berkualitas web vs. gambar berkualitas ponsel), Anda harus membuat **train-dev set**. Ini adalah sebagian dari *training set* yang disisihkan. Jika model bekerja baik di *training set* tetapi buruk di *train-dev set*, itu berarti *overfitting*. Jika bekerja baik di keduanya tetapi buruk di *validation set*, itu berarti ada *data mismatch*.

---

# KESIMPULAN AKHIR

* Machine Learning adalah tentang membuat mesin menjadi lebih baik dalam tugas tertentu dengan belajar dari data, alih-alih harus mengkodekan aturan secara eksplisit.
* Ada berbagai jenis sistem ML: supervised/unsupervised, batch/online, dan instance-based/model-based.
* Dalam proyek ML, Anda mengumpulkan data dalam *training set*, dan memberikannya ke algoritma pembelajaran. Algoritma ini menyesuaikan parameter model agar sesuai dengan data latih (jika *model-based*) atau hanya menghafal data (jika *instance-based*).
* Sistem tidak akan bekerja baik jika *training set* terlalu kecil, tidak representatif, ber-noise, atau memiliki fitur yang tidak relevan.
* Model Anda harus seimbang: tidak boleh terlalu sederhana (yang akan menyebabkan *underfitting*) atau terlalu kompleks (yang akan menyebabkan *overfitting*).
* Sebelum diluncurkan, model harus dievaluasi menggunakan *test set*. Untuk memilih model dan menyetel *hyperparameter*, gunakan *validation set*.