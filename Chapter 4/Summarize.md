# Bab 4: Melatih Model

Bab ini akan membahas:
* Algoritma inti di balik model-model Machine Learning, dimulai dengan Regresi Linear.
* Dua cara untuk melatih model Regresi Linear: solusi "closed-form" langsung dan algoritma optimisasi iteratif (Gradient Descent).
* Polynomial Regression untuk memodelkan data non-linear.
* Konsep *learning curves* (kurva pembelajaran) untuk mendeteksi *overfitting* dan *underfitting*.
* Teknik regularisasi (Ridge, Lasso, Elastic Net) untuk mengurangi *overfitting*.
* Model Regresi Logistik dan Softmax untuk tugas klasifikasi.

---

## Regresi Linear

Regresi Linear adalah model fundamental yang membuat prediksi dengan menghitung *weighted sum* (jumlah berbobot) dari fitur input, ditambah nilai *bias* (juga disebut *intercept term*)[cite: 979].

$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n$

* $\hat{y}$ adalah nilai prediksi.
* $n$ adalah jumlah fitur.
* $x_i$ adalah nilai fitur ke-i.
* $\theta_j$ adalah parameter model (termasuk bias $\theta_0$ dan bobot fitur $\theta_1, \dots, \theta_n$).

Melatih model ini berarti menemukan nilai $\theta$ (theta) yang meminimalkan *cost function* (fungsi biaya), yang biasanya adalah **Mean SquaredError (MSE)**.

### 1. The Normal Equation (Persamaan Normal)
Ini adalah solusi "closed-form" (analitis) yang dapat langsung menemukan nilai $\theta$ yang meminimalkan MSE[cite: 982].

$\hat{\theta} = (X^T X)^{-1} X^T y$

* **Kelebihan**: Memberikan hasil optimal secara langsung tanpa iterasi.
* **Kekurangan**: Sangat lambat secara komputasi (kompleksitasnya bisa mencapai $O(n^3)$) ketika jumlah fitur ($n$) sangat besar. Tidak berfungsi jika matriks $X^T X$ tidak dapat di-invers (singular)[cite: 986, 986].

### 2. Gradient Descent (Penurunan Gradien)
Gradient Descent (GD) adalah algoritma optimisasi iteratif yang secara bertahap menyesuaikan parameter ($\theta$) untuk meminimalkan *cost function*[cite: 987].

* **Konsep**: Bayangkan Anda berada di gunung berkabut dan ingin turun ke lembah (minimum). Anda mengambil langkah-langkah kecil ke arah turunan yang paling curam.
* **Learning Rate (Laju Belajar)**: Ukuran dari langkah-langkah tersebut[cite: 988]. Jika terlalu kecil, butuh waktu lama untuk konvergen. Jika terlalu besar, Anda mungkin "melompati" lembah dan malah divergen (menjauh)[cite: 989].
* **Masalah Penskalaan**: GD sensitif terhadap skala fitur. Jika fitur memiliki skala yang sangat berbeda, *cost function* akan berbentuk seperti mangkuk lonjong, sehingga butuh waktu lebih lama untuk menemukan minimum. Inilah mengapa **feature scaling** (seperti `StandardScaler`) sangat penting[cite: 994].

Terdapat beberapa varian Gradient Descent:
* **Batch Gradient Descent (Batch GD)**: Menghitung gradien berdasarkan **seluruh** data latih pada setiap langkah. Ini akurat tetapi sangat lambat pada dataset besar [cite: 121-122].
* **Stochastic Gradient Descent (SGD)**: Hanya menggunakan **satu** instans data secara acak pada setiap langkah. Ini jauh lebih cepat tetapi kurang stabil (hasilnya "memantul-mantul"). Bagus untuk keluar dari minimum lokal [cite: 124-125].
* **Mini-batch Gradient Descent (Mini-batch GD)**: Kompromi yang baik. Menghitung gradien berdasarkan sekelompok kecil data (*mini-batch*). Ini mendapatkan performa lebih stabil daripada SGD sekaligus lebih cepat daripada Batch GD[cite: 127].

---

## Polynomial Regression

Bagaimana jika data Anda tidak lurus (non-linear)? Anda masih bisa menggunakan model linear. **Polynomial Regression** menambahkan pangkat (pangkat 2, 3, dst.) dari setiap fitur sebagai fitur baru, kemudian melatih model linear pada set fitur yang diperluas ini [cite: 128-129].

Contoh: `y = a*x^2 + b*x + c`

Ini adalah cara sederhana untuk menangani data non-linear, tetapi jika Anda menggunakan pangkat yang terlalu tinggi, model akan sangat rentan terhadap *overfitting*.

---

## Kurva Pembelajaran (Learning Curves)

*Learning curves* adalah plot performa model pada *training set* dan *validation set* sebagai fungsi dari ukuran *training set*. Ini adalah alat penting untuk mendiagnosis masalah model [cite: 130-131].

* **Underfitting**: Jika model *underfitting*, kedua kurva (latih dan validasi) akan mencapai *plateau* (datar) pada tingkat error yang tinggi dan saling berdekatan. Menambahkan lebih banyak data **tidak akan membantu**. Anda perlu model yang lebih kompleks atau fitur yang lebih baik[cite: 1012].
* **Overfitting**: Jika model *overfitting*, kurva error latih akan sangat rendah, tetapi kurva error validasi akan jauh lebih tinggi. Ada **celah (gap)** besar di antara keduanya. Solusinya adalah mendapatkan lebih banyak data latih atau melakukan regularisasi [cite: 1015-1016].

---

## Regularized Linear Models (Model Linear Teregulasi)

Regularisasi adalah cara untuk membatasi model agar tidak *overfitting*, biasanya dengan membatasi bobot (weights) model.

* **Ridge Regression (Regularisasi $\ell_2$)**: Menambahkan "penalti" ke *cost function* yang setara dengan kuadrat dari bobot. Ini memaksa bobot untuk tetap sekecil mungkin[cite: 1017].
* **Lasso Regression (Regularisasi $\ell_1$)**: Menambahkan penalti yang setara dengan nilai absolut dari bobot[cite: 1021]. Efek utamanya adalah Lasso cenderung **menghilangkan bobot fitur yang tidak penting** (menjadikannya nol). Ini berguna sebagai metode *feature selection* otomatis.
* **Elastic Net**: Gabungan dari Ridge dan Lasso. Anda dapat mengontrol rasio campuran antara $\ell_1$ dan $\ell_2$. Secara umum, Elastic Net lebih disukai daripada Lasso karena lebih stabil[cite: 1026].
* **Early Stopping**: Metode regularisasi yang berbeda. Model dihentikan pelatihannya segera setelah error pada *validation set* mencapai minimum (berhenti membaik) [cite: 1027-1028].

---

## Logistic dan Softmax Regression

### Regresi Logistik (Logistic Regression)
Meskipun namanya "regresi", ini adalah algoritma **klasifikasi biner** (binary classification). Model ini menghitung probabilitas bahwa sebuah instans termasuk dalam kelas tertentu[cite: 1030].

* Ia menghitung *weighted sum* dari fitur (seperti Regresi Linear).
* Kemudian memasukkan hasilnya ke **fungsi logistik (sigmoid)**, yang mengeluarkan angka antara 0 dan 1 (probabilitas) [cite: 1030-1031].
* *Cost function* yang digunakan adalah **Log Loss** (atau *cross-entropy*)[cite: 1032].

### Regresi Softmax (Softmax Regression)
Ini adalah generalisasi dari Regresi Logistik untuk mendukung banyak kelas secara langsung (klasifikasi multikelas), bukan hanya dua[cite: 1040].

* Model ini menghitung skor untuk setiap kelas.
* Kemudian skor tersebut dilewatkan ke **fungsi softmax**, yang mengubah skor menjadi probabilitas (memastikan semua probabilitas jika dijumlahkan adalah 1)[cite: 1040].
* Model ini memprediksi kelas dengan probabilitas tertinggi.

---

# KESIMPULAN AKHIR

* Bab ini menyelami cara kerja dan pelatihan model-model linear fundamental.
* **Regresi Linear** dapat dilatih secara langsung (Normal Equation) atau secara iteratif (Gradient Descent).
* **Gradient Descent** adalah algoritma optimisasi inti yang memiliki beberapa varian (Batch, Stochastic, Mini-batch) untuk menyeimbangkan kecepatan dan stabilitas.
* **Feature scaling** sangat penting untuk algoritma berbasis Gradien.
* **Polynomial Regression** memungkinkan model linear untuk memfit data non-linear.
* **Kurva pembelajaran** adalah alat penting untuk mendiagnosis *overfitting* dan *underfitting*.
* **Regularisasi** (Ridge, Lasso, Elastic Net) adalah teknik krusial untuk mengontrol *overfitting* dengan membatasi parameter model.
* **Regresi Logistik** (untuk klasifikasi biner) dan **Regresi Softmax** (untuk klasifikasi multikelas) menggunakan prinsip serupa untuk memprediksi probabilitas kelas.