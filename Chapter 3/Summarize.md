Tentu, mari kita lanjutkan dengan Bab 3.

Berikut adalah ringkasan Bab 3 dari buku "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow", yang diformat sesuai dengan contoh Anda.

---

# Bab 3: Klasifikasi

Bab ini akan membahas:
* Tugas fundamental *supervised learning*: Klasifikasi.
* Dataset MNIST sebagai contoh klasik untuk melatih model klasifikasi.
* Metrik performa yang penting untuk mengevaluasi model klasifikasi, seperti Akurasi, Presisi, Recall, dan ROC.
* Cara kerja dan evaluasi *binary classifier* (pengklasifikasi biner).
* Perluasan ke *multiclass*, *multilabel*, dan *multioutput classification*.
* Pentingnya *error analysis* (analisis kesalahan) untuk menyempurnakan model.

---

## 1. Dataset MNIST

Bab ini menggunakan dataset MNIST sebagai studi kasus utamanya[cite: 85, 943]. MNIST adalah "hello world" dalam Machine Learning, berisi 70.000 gambar kecil angka tulisan tangan (0-9)[cite: 85, 943].

* **Dataset**: 60.000 gambar untuk *training set* (set latih) dan 10.000 gambar untuk *test set* (set uji)[cite: 947].
* **Fitur**: Setiap gambar berukuran 28x28 piksel, menghasilkan 784 fitur (piksel) per gambar[cite: 944].
* **Tugas**: Mengklasifikasikan gambar ke dalam 10 kelas (angka 0 hingga 9).

## 2. Melatih Binary Classifier

Untuk menyederhanakan masalah, bab ini dimulai dengan membuat *binary classifier* (pengklasifikasi biner), misalnya, "detektor angka 5" yang hanya bertugas membedakan antara "angka 5" dan "bukan angka 5"[cite: 948].

Model yang digunakan sebagai contoh awal adalah **Stochastic Gradient Descent (SGD) Classifier** (`SGDClassifier` dari Scikit-Learn)[cite: 948].

## 3. Metrik Performa (Performance Measures)

Mengevaluasi model klasifikasi jauh lebih rumit daripada model regresi.

### Akurasi dan Cross-Validation
Mengukur akurasi menggunakan *cross-validation* (`cross_val_score()`) adalah langkah awal yang baik[cite: 89, 949]. Namun, **akurasi seringkali bukan metrik yang baik**, terutama untuk *skewed datasets* (dataset condong)[cite: 950].

Contoh: Jika hanya 10% gambar adalah angka 5, model yang selalu menebak "bukan 5" akan memiliki akurasi 90%[cite: 950].

### Confusion Matrix (Matriks Kebingungan)
Cara yang jauh lebih baik untuk mengevaluasi performa adalah dengan melihat **Confusion Matrix**[cite: 90, 950]. Matriks ini menghitung berapa kali kelas A salah diklasifikasikan sebagai kelas B.

* **True Positives (TP)**: Prediksi benar positif (misal: menebak 5, dan itu benar 5).
* **False Positives (FP)**: Prediksi salah positif (misal: menebak 5, padahal bukan 5).
* **True Negatives (TN)**: Prediksi benar negatif (misal: menebak bukan 5, dan itu benar bukan 5).
* **False Negatives (FN)**: Prediksi salah negatif (misal: menebak bukan 5, padahal itu 5).

### Precision dan Recall
Dari *confusion matrix*, kita mendapatkan dua metrik yang lebih baik:

* **Precision (Presisi)**: Akurasi dari prediksi positif; seberapa sering tebakan "positif" Anda benar[cite: 91].
    * $Precision = \frac{TP}{TP + FP}$ [cite: 953]
* **Recall (Sensitivitas)**: Rasio instans positif yang berhasil dideteksi oleh pengklasifikasi; seberapa banyak "positif" yang berhasil Anda temukan[cite: 91].
    * $Recall = \frac{TP}{TP + FN}$ [cite: 953]

### Precision/Recall Trade-off
Meningkatkan *precision* seringkali mengurangi *recall*, dan sebaliknya[cite: 93, 953]. Ini disebut **precision/recall trade-off**. Model (`SGDClassifier`) membuat keputusan berdasarkan skor; dengan menaikkan atau menurunkan *decision threshold* (ambang batas keputusan), Anda bisa mendapatkan *precision* yang lebih tinggi dengan mengorbankan *recall*, atau sebaliknya[cite: 93, 954].

* **F1 Score**: Metrik yang menggabungkan *precision* dan *recall* menjadi satu angka (rata-rata harmonik)[cite: 92, 953]. Ini berguna jika Anda membutuhkan keseimbangan antara keduanya.

### Kurva ROC (Receiver Operating Characteristic)
Kurva ROC adalah alat umum lainnya untuk *binary classifiers*[cite: 97, 958]. Kurva ini memplot **True Positive Rate (Recall)** melawan **False Positive Rate (FPR)**[cite: 97, 958].

* **FPR**: Rasio instans negatif yang salah diklasifikasikan sebagai positif.
* Model yang baik akan menjauh dari garis diagonal (pengklasifikasi acak) dan menuju sudut kiri atas[cite: 958].
* **Area Under the Curve (AUC)**: Metrik tunggal yang mengukur performa model. Model sempurna memiliki AUC = 1, sedangkan model acak memiliki AUC = 0,5[cite: 98, 959].

> **Kapan menggunakan PR vs ROC?**
> Anda sebaiknya lebih memilih kurva Precision/Recall (PR) ketika kelas positif **jarang** (sedikit) atau ketika Anda lebih peduli pada *false positives* daripada *false negatives*. Jika tidak, kurva ROC adalah pilihan yang baik[cite: 959].

## 4. Multiclass, Multilabel, dan Multioutput

### Multiclass Classification
*Multiclass classifiers* (atau *multinomial classifiers*) dapat membedakan antara lebih dari dua kelas[cite: 100, 962].
* **One-versus-the-Rest (OvR) / One-versus-All (OvA)**: Melatih satu *binary classifier* untuk setiap kelas (misalnya, satu detektor 0, satu detektor 1, dst.). Saat memprediksi, pilih kelas yang pengklasifikasinya memberikan skor tertinggi[cite: 100, 962].
* **One-versus-One (OvO)**: Melatih *binary classifier* untuk setiap pasangan kelas (misal: satu untuk 0 vs 1, satu untuk 0 vs 2, satu untuk 1 vs 2, dst.). Ini membutuhkan pelatihan N x (N-1) / 2 pengklasifikasi[cite: 100, 962]. Scikit-Learn secara otomatis mendeteksi ketika Anda menggunakan *binary classifier* untuk tugas *multiclass* dan memilih OvR atau OvO (misalnya, `SVC` menggunakan OvO)[cite: 963].

### Error Analysis (Analisis Kesalahan)
Setelah melatih model *multiclass*, cara terbaik untuk meningkatkannya adalah dengan menganalisis kesalahannya. Dengan memplot *confusion matrix* (seringkali divisualisasikan sebagai gambar), Anda dapat melihat jenis kesalahan apa yang paling sering dibuat oleh model Anda [cite: 102-104, 965, 968].

### Multilabel dan Multioutput
* **Multilabel Classification**: Sistem klasifikasi yang dapat mengeluarkan beberapa label biner untuk setiap instans[cite: 106, 971]. Contoh: Mengenali beberapa orang dalam satu foto (Alice=Ya, Bob=Tidak, Charlie=Ya)[cite: 971].
* **Multioutput Classification**: Generalisasi dari *multilabel*, di mana setiap label bisa jadi *multiclass* (memiliki lebih dari dua nilai yang mungkin)[cite: 107, 973]. Contoh: Menghapus noise dari gambar, di mana setiap piksel adalah label (multioutput) dan setiap label dapat memiliki 256 nilai (multiclass)[cite: 973].

---

# KESIMPULAN AKHIR

* Bab ini berfokus pada **klasifikasi**, tugas *supervised learning* yang paling umum.
* Menggunakan dataset **MNIST** sebagai contoh utama.
* **Akurasi** bukanlah metrik yang baik untuk *skewed datasets*.
* Metrik performa yang lebih baik didapat dari **Confusion Matrix**, yaitu **Precision** dan **Recall**.
* Selalu ada **trade-off** antara *precision* dan *recall*, yang dapat dianalisis menggunakan kurva PR atau kurva **ROC**.
* Model *binary classifier* dapat diperluas untuk **multiclass classification** menggunakan strategi OvR atau OvO.
* **Error analysis** sangat penting untuk menemukan cara meningkatkan model Anda.
* Bab ini juga memperkenalkan **Multilabel Classification** (beberapa label biner per instans) dan **Multioutput Classification** (generalisasi dari multilabel).