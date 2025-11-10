# Bab 11: Melatih Jaringan Saraf Tiruan yang Dalam

Bab ini akan membahas:
* Masalah fundamental yang muncul saat melatih Deep Neural Networks (DNNs), yaitu *vanishing/exploding gradients*.
* Solusi untuk masalah ini, termasuk inisialisasi bobot (Weight Initialization), *nonsaturating activation functions* (fungsi aktivasi non-saturasi), Batch Normalization, dan Gradient Clipping.
* Menggunakan kembali *layer* yang sudah dilatih (Transfer Learning) untuk mempercepat pelatihan dan mengatasi kekurangan data.
* Algoritma optimisasi (optimizer) yang lebih cepat selain SGD standar, seperti Adam, Nadam, dan RMSProp.
* Teknik regularisasi khusus untuk deep learning, seperti Dropout dan Max-Norm, untuk mencegah *overfitting*.

---

## 1. Masalah Vanishing/Exploding Gradients

Saat melatih DNN menggunakan *backpropagation*, gradien (turunan) seringkali menjadi semakin kecil (*vanishing*) atau semakin besar (*exploding*) saat merambat turun melalui *hidden layers*.

* **Vanishing Gradients**: Gradien menjadi sangat kecil, menyebabkan *layer* bagian bawah (yang dekat dengan input) belajar sangat lambat atau berhenti belajar sama sekali.
* **Exploding Gradients**: Gradien menjadi sangat besar, menyebabkan pembaruan bobot (weights) menjadi tidak stabil dan model gagal konvergen (divergen).

Masalah ini sebagian besar disebabkan oleh kombinasi fungsi aktivasi logistik/sigmoid dan metode inisialisasi bobot yang tidak tepat.

---

## 2. Solusi untuk Gradien yang Tidak Stabil

Bab ini menawarkan beberapa solusi teknis untuk melawan masalah gradien yang tidak stabil.

### a. Inisialisasi Glorot dan He
Cara Anda menginisialisasi bobot sangatlah penting. Inisialisasi yang salah dapat membuat varians sinyal meledak atau menghilang.
* **Inisialisasi Glorot (Xavier)**: Inisialisasi default di Keras. Bekerja baik dengan fungsi aktivasi sigmoid, tanh, dan softmax. Tujuannya adalah menjaga varians sinyal tetap sama saat *forward pass* dan *backward pass* [cite: 333-334].
* **Inisialisasi He**: Didesain khusus untuk fungsi aktivasi **ReLU** (dan variannya). Sangat penting digunakan jika Anda menggunakan ReLU[cite: 334].

### b. Nonsaturating Activation Functions
Fungsi aktivasi Sigmoid dan Tanh akan "jenuh" (menjadi datar) pada nilai input yang besar, yang berarti gradiennya menjadi nol. Ini menghentikan aliran gradien.
* **ReLU (Rectified Linear Unit)**: Pilihan default yang paling populer. Cepat dihitung dan tidak jenuh untuk nilai positif. Namun, ia memiliki masalah "dying ReLUs", di mana neuron bisa "mati" (selalu mengeluarkan 0)[cite: 335].
* **Leaky ReLU**: Varian dari ReLU yang memperbolehkan sedikit gradien negatif (misalnya, `LeakyReLU(alpha=0.2)`). Ini memperbaiki masalah "dying ReLUs" [cite: 335-336].
* **ELU (Exponential Linear Unit)**: Mengungguli ReLU. Dapat mengambil nilai negatif, memungkinkan output rata-rata *layer* lebih dekat ke 0, yang membantu meringankan masalah *vanishing gradient*.
* **SELU (Scaled ELU)**: Varian dari ELU. Jika Anda menggunakan inisialisasi `lecun_normal`, standarisasi input, dan arsitektur sekuensial, jaringan akan **self-normalize** (menormalkan dirinya sendiri). Ini seringkali memberikan performa yang sangat baik [cite: 337-338].

### c. Batch Normalization (BN)
Teknik ini adalah salah satu terobosan terpenting. BN menambahkan sebuah operasi di setiap *layer* (sebelum atau sesudah fungsi aktivasi) yang menormalkan output dari *layer* tersebut (zero-center dan normalize) sebelum diteruskan ke *layer* berikutnya [cite: 338-339].
* **Manfaat**:
    1.  Secara signifikan mengurangi masalah *vanishing/exploding gradients*.
    2.  Memungkinkan penggunaan *learning rate* yang jauh lebih tinggi, sehingga pelatihan jauh lebih cepat.
    3.  Berfungsi sebagai **regularizer**, mengurangi kebutuhan akan teknik lain seperti Dropout.
* **Implementasi**: Cukup tambahkan `keras.layers.BatchNormalization()` di antara *layer-layer* Anda.

### d. Gradient Clipping
Teknik lain untuk mengatasi *exploding gradients* adalah dengan "memotong" (clipping) gradien selama *backpropagation* agar tidak pernah melebihi *threshold* (ambang batas) tertentu. Ini paling sering digunakan dalam Recurrent Neural Networks (RNN)[cite: 345].

---

## 3. Menggunakan Kembali Layer yang Sudah Dilatih (Transfer Learning)

Daripada melatih DNN besar dari awal, hampir selalu lebih baik menggunakan **Transfer Learning**.
* **Konsep**: Temukan jaringan saraf yang sudah ada (model *pretrained*) yang menyelesaikan tugas serupa dengan tugas Anda. Gunakan kembali *layer-layer* bawah dari jaringan tersebut, karena *layer* bawah cenderung mempelajari fitur-fitur generik (misalnya, deteksi tepi, tekstur) yang berguna untuk banyak tugas [cite: 345-346].
* **Cara Melakukan**:
    1.  Muat model *pretrained* (misalnya, dari `keras.applications`), sisakan bagian *output layer*-nya (`include_top=False`).
    2.  Tambahkan *output layer* baru Anda sendiri di atasnya yang sesuai dengan tugas Anda.
    3.  **Bekukan (freeze)** bobot *layer* yang digunakan kembali (atur `layer.trainable = False`) agar tidak hancur selama pelatihan awal.
    4.  Latih model Anda hanya pada *layer* baru selama beberapa *epoch*.
    5.  **Buka (unfreeze)** beberapa *layer* teratas yang dibekukan, gunakan *learning rate* yang jauh lebih kecil, dan lanjutkan pelatihan untuk menyempurnakan (*fine-tune*) bobot untuk tugas spesifik Anda [cite: 347-348].

**Unsupervised Pretraining** (Pra-pelatihan Tak Terawasi) adalah teknik terkait di mana Anda terlebih dahulu melatih model Anda pada data tak berlabel (misalnya, menggunakan Autoencoder), kemudian menyempurnakannya pada data berlabel untuk tugas Anda [cite: 349-350].

---

## 4. Optimizer yang Lebih Cepat

Melatih DNN yang besar bisa sangat lambat. Menggunakan optimizer yang lebih baik dari `SGD` standar sangat penting.

* **Momentum Optimization**: Membantu SGD berakselerasi di arah yang benar dan melewati minimum lokal yang dangkal dengan menambahkan "momentum" (rata-rata tertimbang dari gradien sebelumnya) [cite: 351-352].
* **Nesterov Accelerated Gradient (NAG)**: Varian *momentum* yang sedikit lebih cerdas; ia mengukur gradien "sedikit di depan" ke arah momentum, menghasilkan konvergensi yang sedikit lebih cepat[cite: 353].
* **AdaGrad**: Optimizer dengan *adaptive learning rate*. Ia menurunkan *learning rate* lebih banyak untuk parameter yang sering diperbarui (fitur yang curam) dan lebih sedikit untuk parameter yang jarang diperbarui. Bekerja baik untuk masalah sederhana, tetapi seringkali berhenti terlalu dini [cite: 354-355].
* **RMSProp**: Memperbaiki AdaGrad dengan hanya mengakumulasi gradien dari iterasi terbaru (menggunakan *exponential decay*). Ini adalah optimizer yang sangat populer [cite: 355-356].
* **Adam (Adaptive Moment Estimation)**: Optimizer yang paling banyak digunakan. Menggabungkan ide dari *momentum optimization* dan *RMSProp*. Ini adalah optimizer default yang direkomendasikan [cite: 356-357].
* **Nadam**: Adam ditambah dengan *Nesterov trick*.

---

## 5. Regularisasi untuk Menghindari Overfitting

DNN sangat fleksibel dan memiliki jutaan parameter, membuatnya sangat rentan terhadap *overfitting*.

* **$\ell_1$ dan $\ell_2$ Regularization**: Sama seperti pada model linear, Anda dapat menambahkan penalti $\ell_1$ atau $\ell_2$ pada bobot koneksi di *layer* Keras untuk menjaga bobot tetap kecil.
* **Dropout**: Teknik regularisasi yang paling populer untuk DNN. Pada setiap langkah pelatihan, setiap neuron (kecuali di *output layer*) memiliki probabilitas $p$ untuk "dijatuhkan" (diabaikan) sementara. Ini memaksa neuron untuk belajar dengan rekan-rekan yang berbeda dan tidak terlalu bergantung pada neuron spesifik lainnya. Hasilnya adalah model yang lebih robust [cite: 365-367].
* **Monte Carlo (MC) Dropout**: Teknik menggunakan *dropout* pada saat *inference* (test time) untuk mendapatkan beberapa prediksi. Rata-rata dari prediksi ini seringkali lebih baik daripada prediksi tunggal, dan standar deviasinya memberi Anda ukuran ketidakpastian (*uncertainty*) model [cite: 368-369].
* **Max-Norm Regularization**: Membatasi bobot $w$ dari setiap neuron sehingga $||w||_2 \le r$ (di mana $r$ adalah *hyperparameter*). Ini membantu membatasi bobot agar tidak meledak.

---

# KESIMPULAN AKHIR

* Melatih DNN yang dalam menghadirkan tantangan spesifik, terutama **vanishing/exploding gradients** dan **overfitting**.
* Masalah gradien dapat diatasi dengan kombinasi: (1) **Inisialisasi He** yang tepat, (2) **Fungsi aktivasi non-saturasi** (seperti SELU atau ELU), dan (3) **Batch Normalization**.
* **Transfer Learning** (menggunakan kembali *layer* dari model *pretrained*) adalah teknik yang sangat efektif untuk mendapatkan performa tinggi dengan cepat dan dengan data yang lebih sedikit.
* Menggunakan **optimizer** modern seperti **Adam** atau **Nadam** jauh lebih efisien daripada SGD standar.
* **Dropout** adalah teknik regularisasi pilihan untuk DNN, yang secara efektif melatih ansambel jaringan yang lebih kecil untuk mencegah *overfitting*.