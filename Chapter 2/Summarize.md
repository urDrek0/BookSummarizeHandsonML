# Bab 2: Proyek Machine Learning End-to-End

Bab ini akan membahas:
* Alur kerja lengkap sebuah proyek ML, dari awal hingga akhir, menggunakan contoh nyata.
* Cara membingkai masalah (Framing the Problem) dan memilih metrik performa yang tepat.
* Teknik untuk mendapatkan, menjelajahi (exploring), dan memvisualisasikan data untuk mendapatkan wawasan.
* Metode penting untuk mempersiapkan data (Data Preparation), termasuk pembersihan, transformasi, dan feature engineering.
* Cara memilih, melatih, dan mengevaluasi berbagai model.
* Proses *fine-tuning* model terbaik untuk mendapatkan solusi optimal sebelum diluncurkan.

---

## Ringkasan Alur Kerja Proyek

Bab ini memandu kita melalui 8 langkah utama dalam sebuah proyek ML, menggunakan studi kasus prediksi harga rumah di California [cite: 463-467]:

1.  **Memahami Gambaran Besar (Look at the Big Picture)**: Memahami tujuan bisnis [cite: 491], bagaimana model akan digunakan, dan mendefinisikan masalah (misalnya, supervised, regression, batch learning)[cite: 502].
2.  **Mendapatkan Data (Get the Data)**: Mengunduh data dan melihat strukturnya[cite: 572, 584].
3.  **Menjelajahi dan Memvisualisasikan Data (Discover and Visualize)**: Menganalisis data, mencari korelasi, dan mengidentifikasi keunikan (quirks) untuk mendapatkan wawasan[cite: 464, 808].
4.  **Mempersiapkan Data (Prepare the Data)**: Menulis fungsi-fungsi untuk membersihkan dan mentransformasi data secara otomatis (misalnya, menangani data yang hilang, *feature scaling*, dll.)[cite: 465, 827].
5.  **Memilih dan Melatih Model (Select and Train a Model)**: Mencoba beberapa model (seperti Linear Regression, Decision Trees) dan mengevaluasinya[cite: 465, 928, 929].
6.  **Menyempurnakan Model (Fine-Tune Your Model)**: Mengoptimalkan *hyperparameter* dari model-model terbaik menggunakan teknik seperti Grid Search[cite: 466, 932].
7.  **Menyajikan Solusi (Present Your Solution)**: Mendokumentasikan dan mempresentasikan temuan Anda[cite: 467, 936].
8.  **Meluncurkan, Memantau, dan Memelihara (Launch, Monitor, and Maintain)**: Menerapkan model ke produksi dan memantau kinerjanya[cite: 467, 936].

---

## 1. Membingkai Masalah dan Memilih Metrik

Sebelum menulis kode, sangat penting untuk memahami tujuan bisnis. Bagaimana perusahaan akan mendapat manfaat dari model ini?[cite: 491]. Ini akan menentukan:
* **Framing**: Apakah ini tugas *supervised*, *unsupervised*, atau *Reinforcement Learning*? Apakah ini *classification* atau *regression*? Apakah ini *batch* atau *online learning*?[cite: 502]. (Dalam contoh ini: supervised regression, batch learning) [cite: 502].
* **Performance Measure**: Bagaimana kita mengukur kesuksesan? Untuk regresi, metrik yang umum adalah **Root Mean Square Error (RMSE)** [cite: 504] atau **Mean Absolute Error (MAE)**[cite: 519]. RMSE lebih sensitif terhadap *outlier* (nilai ekstrem)[cite: 523].

## 2. Mendapatkan Data dan Membuat Test Set

Setelah data didapat [cite: 572], langkah paling penting adalah membuat **Test Set** (Set Uji) dan **menyingkirkannya**[cite: 680].

Anda tidak boleh "mengintip" data uji. Otak Anda adalah sistem pendeteksi pola yang luar biasa, dan jika Anda melihat *test set*, Anda mungkin secara tidak sadar menemukan pola dan memilih model yang bias terhadap data tersebut. Ini disebut **data snooping bias**[cite: 681].

* **Pemisahan Acak**: Cara termudah adalah menggunakan `train_test_split` dari Scikit-Learn[cite: 698].
* **Stratified Sampling**: Jika dataset Anda tidak seimbang (misalnya, beberapa kategori sangat sedikit jumlahnya), Anda harus menggunakan *stratified sampling* (pengambilan sampel terstratifikasi) untuk memastikan *test set* Anda representatif terhadap keseluruhan populasi data[cite: 699].

## 3. Menjelajahi dan Memvisualisasikan Data

Fase ini adalah tentang memahami data secara mendalam:
* **Visualisasi**: Gunakan histogram (`hist()`) untuk melihat distribusi setiap atribut[cite: 638, 674]. Untuk data geografis, gunakan *scatterplot* untuk memvisualisasikan lokasi[cite: 732].
* **Mencari Korelasi**: Gunakan metode `corr()` untuk menghitung koefisien korelasi standar (Pearson's r) antar atribut[cite: 763]. Gunakan `scatter_matrix` untuk memvisualisasikan korelasi antar beberapa atribut sekaligus[cite: 795].
* **Eksperimen Kombinasi Atribut**: Lakukan *feature engineering* sederhana. Terkadang, mengkombinasikan atribut (misalnya, `jumlah_kamar / jumlah_rumah_tangga`) bisa menghasilkan fitur yang lebih berkorelasi dengan target[cite: 809].

## 4. Mempersiapkan Data (Data Preprocessing)

Kita harus menulis fungsi-fungsi transformasi data[cite: 827], karena ini memungkinkan kita untuk:
1.  Mereproduksi transformasi yang sama pada data baru (misalnya, *test set*).
2.  Membangun pustaka transformasi yang bisa digunakan kembali.
3.  Menggunakan transformasi ini dalam *live system* di produksi.

Langkah-langkah utamanya meliputi:

* **Data Cleaning**: Menangani nilai yang hilang (missing values).
    * Opsi 1: Hapus baris data tersebut (`dropna()`)[cite: 834].
    * Opsi 2: Hapus seluruh atribut/fitur (`drop()`)[cite: 835].
    * Opsi 3: Isi nilai yang hilang (dengan nol, rata-rata, atau median) menggunakan `fillna()` atau `SimpleImputer` Scikit-Learn[cite: 834, 836, 838].
* **Handling Teks dan Atribut Kategorikal**: Sebagian besar model ML lebih suka angka.
    * **OrdinalEncoder**: Mengubah kategori teks menjadi angka (misalnya, "jelek", "sedang", "bagus" menjadi 0, 1, 2)[cite: 873].
    * **OneHotEncoder**: Mengubah kategori menjadi vektor biner (misalnya, "MERAH" menjadi [1, 0, 0], "HIJAU" menjadi [0, 1, 0]). Ini mencegah model menganggap dua nilai yang berdekatan lebih mirip daripada yang berjauhan [cite: 883-884].
* **Feature Scaling**: Model ML seringkali berkinerja buruk jika fitur input memiliki skala yang sangat berbeda[cite: 915].
    * **Min-max scaling (Normalisasi)**: Mengubah skala nilai sehingga berakhir di rentang 0 hingga 1.
    * **Standardization (Standardisasi)**: Mengurangi nilai rata-rata dan membaginya dengan standar deviasi. Metode ini tidak membatasi nilai ke rentang tertentu, tetapi jauh lebih tidak terpengaruh oleh *outlier*[cite: 916].
* **Transformation Pipelines**: Scikit-Learn menyediakan kelas `Pipeline` untuk merangkai beberapa langkah transformasi secara berurutan[cite: 919]. Ini sangat membantu karena kita hanya perlu memanggil `fit()` dan `transform()` satu kali pada seluruh pipeline.
* **ColumnTransformer**: Seringkali, Anda perlu menerapkan transformasi berbeda pada kolom berbeda (misalnya, *OneHotEncoder* untuk data kategorikal dan *StandardScaler* untuk data numerik). `ColumnTransformer` dibuat khusus untuk ini[cite: 927].

## 5. Memilih dan Melatih Model

Setelah data disiapkan:
1.  **Latih Model Sederhana**: Mulailah dengan model sederhana seperti **Linear Regression**[cite: 928]. Latih pada *training set*.
2.  **Evaluasi**: Jangan hanya mengandalkan error di *training set*. Model yang lebih kompleks seperti **DecisionTreeRegressor** mungkin mendapatkan error nol pada data latih, tapi ini adalah tanda *overfitting* yang parah[cite: 929].
3.  **Cross-Validation**: Gunakan **K-fold cross-validation** untuk evaluasi yang lebih baik[cite: 929]. Scikit-Learn membagi data latih menjadi beberapa *folds* (lipatan), lalu melatih dan mengevaluasi model beberapa kali, mengambil satu *fold* berbeda sebagai set validasi setiap kalinya[cite: 929]. Ini memberi Anda estimasi performa yang lebih stabil.
4.  **Coba Model Lain**: Coba model yang lebih kuat seperti **RandomForestRegressor**, yang merupakan model *ensemble* dan seringkali berkinerja sangat baik[cite: 930].

## 6. Menyempurnakan Model (Fine-Tuning)

Setelah Anda memiliki daftar pendek model yang menjanjikan, saatnya untuk menyempurnakannya.
* **Grid Search**: Scikit-Learn menyediakan `GridSearchCV`. Anda memberi tahu kombinasi *hyperparameter* mana yang ingin Anda coba, dan itu akan mengevaluasi semuanya menggunakan *cross-validation* untuk menemukan kombinasi terbaik[cite: 932].
* **Randomized Search**: Jika ruang pencarian *hyperparameter* besar, `RandomizedSearchCV` seringkali lebih efisien. Ia mencoba sejumlah kombinasi acak dalam jumlah iterasi yang Anda tetapkan[cite: 934].
* **Analisis Model Terbaik**: Periksa model terbaik Anda. Misalnya, *Random Forest* dapat memberi tahu Anda *feature importance* (fitur mana yang paling penting)[cite: 934].
* **Evaluasi di Test Set**: Setelah Anda memiliki model final, evaluasi kinerjanya **satu kali saja** menggunakan *test set* untuk mendapatkan estimasi *generalization error*[cite: 935]. Jangan menyetel model berdasarkan *test set*!

---

# KESIMPULAN AKHIR

* Bab ini mempraktikkan alur kerja standar proyek ML, mulai dari membingkai masalah hingga memantau sistem di produksi.
* Selalu sisihkan *test set* Anda sejak awal dan jangan pernah melihatnya sampai akhir untuk menghindari *data snooping bias*.
* Lakukan eksplorasi data secara mendalam untuk mendapatkan wawasan sebelum melakukan *preprocessing*.
* Gunakan **Pipelines** dan **ColumnTransformer** dari Scikit-Learn untuk membuat alur kerja *preprocessing* yang bersih, dapat direproduksi, dan mudah diatur.
* Jangan hanya menggunakan satu model. Latih beberapa model (linear, tree, ensemble, dll.) dan gunakan *cross-validation* untuk mengevaluasinya secara objektif.
* Gunakan **Grid Search** atau **Randomized Search** untuk mengotomatiskan pencarian *hyperparameter* terbaik.
* Hanya evaluasi model final Anda di *test set* tepat sebelum peluncuran.