# GettingStartedwithNLP

![Cover Buku](Cover%20Buku.jpg)

# Praktik Pembelajaran Mesin dengan Scikit-Learn, Keras & TensorFlow

Repositori ini berisi reproduksi kode dan catatan teoretis dari buku *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (Edisi ke-2)* oleh Aurélien Géron.

Tujuan utama repositori ini adalah untuk mempraktikkan konsep, alat, dan teknik untuk membangun sistem cerdas. Konten dibagi menjadi dua bagian utama:

1.  **Bagian I: Dasar-Dasar Pembelajaran Mesin**
    Fokus pada konsep inti ML dan algoritma klasik, sebagian besar menggunakan **Scikit-Learn**. Ini mencakup regresi, klasifikasi, SVM, Decision Tree, Ensemble Learning, dan teknik unsupervised learning seperti pengurangan dimensi dan clustering.
2.  **Bagian II: Jaringan Saraf Tiruan dan Deep Learning**
    Fokus pada implementasi jaringan saraf tiruan (ANN) menggunakan **TensorFlow** dan **Keras**. Ini mencakup MLP, CNN (untuk computer vision), RNN (untuk sekuens), Autoencoder, GAN (untuk generative learning), Reinforcement Learning, dan cara mendeploy model dalam skala besar.

---

## Daftar Istilah

* **Scikit-Learn (sklearn)**: Pustaka Python populer untuk tugas machine learning klasik (non-deep learning) seperti regresi, klasifikasi, clustering, dan reduksi dimensi.
* **TensorFlow (TF)**: Framework komputasi numerik skala besar dari Google yang menjadi fondasi utama untuk melatih dan mendeploy model deep learning.
* **Keras**: API deep learning tingkat tinggi yang menjadi antarmuka resmi dan mudah digunakan untuk TensorFlow. Memudahkan pembuatan prototipe model yang kompleks.
* **Klasifikasi (Classification)**: Tugas supervised learning untuk memprediksi label kategori diskret (misalnya, 'spam' atau 'bukan spam').
* **Regresi (Regression)**: Tugas supervised learning untuk memprediksi nilai numerik kontinu (misalnya, harga rumah).
* **Support Vector Machine (SVM)**: Algoritma supervised learning yang andal, baik untuk klasifikasi linear maupun non-linear, dengan menemukan "jalan" (margin) terluas antar kelas.
* **Decision Tree**: Model serbaguna yang dapat melakukan tugas klasifikasi dan regresi dengan membuat serangkaian keputusan sederhana (seperti diagram alir).
* **Random Forest**: Metode *ensemble* yang menggabungkan banyak Decision Tree untuk mendapatkan prediksi yang lebih akurat dan stabil.
* **Reduksi Dimensi (Dimensionality Reduction)**: Proses mengurangi jumlah fitur (dimensi) dalam dataset untuk mengatasi *curse of dimensionality*, mempercepat training, dan mempermudah visualisasi. Contoh: PCA.
* **Clustering**: Tugas unsupervised learning untuk mengelompokkan data yang mirip ke dalam grup (cluster). Contoh: K-Means, DBSCAN.
* **Jaringan Saraf Tiruan (ANN)**: Model komputasi yang terinspirasi dari struktur otak manusia, terdiri dari lapisan-lapisan neuron buatan.
* **CNN (Convolutional Neural Network)**: Jenis ANN yang sangat efektif untuk data visual (gambar), menggunakan operasi konvolusi untuk mendeteksi pola hierarkis.
* **RNN (Recurrent Neural Network)**: Jenis ANN yang dirancang untuk data sekuensial (seperti teks atau time series) dengan memiliki "memori" dari langkah waktu sebelumnya. Varian canggihnya termasuk LSTM dan GRU.
* **Autoencoder**: Jenis ANN yang dilatih secara unsupervised untuk mengompresi data ke representasi berdimensi rendah (encoding) dan kemudian merekonstruksinya kembali (decoding).
* **GAN (Generative Adversarial Network)**: Arsitektur yang terdiri dari dua jaringan (Generator dan Discriminator) yang "bersaing" untuk menghasilkan data baru yang realistis (misalnya, gambar wajah palsu).
* **Reinforcement Learning (RL)**: Cabang ML di mana "agen" belajar mengambil tindakan dalam suatu "lingkungan" untuk memaksimalkan "reward" kumulatif.

---

## Bagian 1: Dasar-Dasar Pembelajaran Mesin

### Bab 1: Peta Jalan Machine Learning
Pengantar konsep fundamental. Bab ini mendefinisikan apa itu Machine Learning dan mengapa itu berguna. Ini mencakup berbagai jenis sistem ML: supervised/unsupervised, batch/online, dan instance-based/model-based. Bab ini juga membahas tantangan utama (misalnya, overfitting, underfitting) dan alur kerja proyek ML.

### Bab 2: Proyek Machine Learning End-to-End
Membangun proyek regresi lengkap menggunakan data harga rumah California. Bab ini memandu melalui langkah-langkah praktis: mendapatkan data, melakukan visualisasi untuk mendapatkan wawasan, menyiapkan data (termasuk *preprocessing pipelines*), memilih dan melatih model (Linear Regression, Decision Tree), serta menyempurnakan model (fine-tuning) menggunakan cross-validation.

### Bab 3: Klasifikasi
Fokus pada tugas klasifikasi. Menggunakan dataset MNIST sebagai contoh, bab ini membahas metrik performa utama (akurasi, presisi, recall, F1-score, kurva ROC), cara menangani klasifikasi multiclass, dan analisis kesalahan (error analysis) untuk meningkatkan model.

### Bab 4: Melatih Model
Menyelam lebih dalam ke algoritma pelatihan untuk model linear. Bab ini mencakup cara kerja Linear Regression (menggunakan Normal Equation dan Gradient Descent), Polynomial Regression, kurva pembelajaran (learning curves) untuk mendeteksi overfitting, dan teknik regularisasi (Ridge, Lasso, Elastic Net) untuk mengatasinya.

### Bab 5: Support Vector Machines (SVM)
Menjelajahi Support Vector Machines, salah satu model ML klasik yang paling kuat. Bab ini mencakup konsep *margin* (hard dan soft), *kernel trick* (Polynomial, Gaussian RBF) untuk menangani data non-linear secara efisien, dan cara menggunakan SVM untuk tugas regresi.

### Bab 6: Decision Trees
Membahas cara kerja Decision Tree, model yang intuitif dan mudah diinterpretasi. Bab ini mencakup algoritma pelatihan (CART), cara visualisasi pohon, konsep Gini impurity dan entropy, parameter regularisasi untuk menghindari overfitting, dan penggunaan Decision Tree untuk tugas regresi.

### Bab 7: Ensemble Learning dan Random Forests
Membahas teknik menggabungkan beberapa model (disebut *ensemble*) untuk mendapatkan performa yang lebih baik. Bab ini mencakup *voting classifiers*, *bagging* (Bootstrap Aggregating), *pasting*, *Random Forests* (ensemble dari Decision Tree), *boosting* (AdaBoost, Gradient Boosting), dan *stacking*.

### Bab 8: Reduksi Dimensi
Membahas "kutukan dimensi" (curse of dimensionality) dan cara mengatasinya. Bab ini mencakup dua pendekatan utama: proyeksi (PCA) dan manifold learning (LLE). Teknik ini sangat penting untuk visualisasi data, kompresi, dan mempercepat pelatihan.

### Bab 9: Teknik Unsupervised Learning
Menjelajahi algoritma yang belajar tanpa data berlabel. Fokus utamanya adalah *clustering* menggunakan algoritma K-Means dan DBSCAN. Bab ini juga mencakup *Gaussian Mixture Models* (GMM) untuk clustering probabilistik dan penggunaannya untuk *anomaly detection* (deteksi anomali).

---

## Bagian 2: Jaringan Saraf Tiruan dan Deep Learning

### Bab 10: Pengenalan Jaringan Saraf Tiruan dengan Keras
Transisi dari ML klasik ke deep learning. Bab ini memperkenalkan konsep Perceptron dan Multi-Layer Perceptron (MLP) serta algoritma *backpropagation*. Ini adalah panduan praktis pertama untuk Keras: cara membangun, melatih, dan mengevaluasi model regresi dan klasifikasi menggunakan Sequential API dan Functional API.

### Bab 11: Melatih Jaringan Saraf Tiruan yang Dalam
Membahas tantangan dalam melatih DNN yang sangat dalam. Bab ini mengidentifikasi masalah seperti *vanishing/exploding gradients* dan menawarkan solusi praktis: inisialisasi bobot (He, Glorot), fungsi aktivasi non-saturasi (ReLU, ELU, SELU), *Batch Normalization*, dan *transfer learning* (menggunakan kembali layer dari model yang sudah dilatih).

### Bab 12: Model Kustom dan Pelatihan dengan TensorFlow
Menyelam ke API low-level TensorFlow untuk fleksibilitas maksimum. Bab ini mengajarkan cara memanipulasi Tensor dan operasi, lalu menunjukkan cara membuat *custom loss functions*, *layers*, *models*, dan *training loops*. Ini juga memperkenalkan TF Functions dan AutoGraph untuk optimalisasi performa.

### Bab 13: Memuat dan Memproses Data dengan TensorFlow
Membahas cara membangun pipeline data yang sangat efisien dan scalable menggunakan `tf.data` API. Bab ini mencakup cara memuat file teks (CSV), file biner (TFRecord), dan cara membuat *preprocessing layers* (untuk normalisasi, one-hot encoding, dan embeddings) langsung di dalam model.

### Bab 14: Deep Computer Vision Menggunakan Convolutional Neural Networks (CNN)
Fokus pada arsitektur deep learning terbaik untuk gambar. Bab ini menjelaskan blok bangunan CNN (lapisan konvolusi dan pooling) dan arsitektur modern yang canggih seperti AlexNet, GoogLeNet, ResNet, dan Xception. Bab ini juga mencakup teknik lanjutan seperti *Object Detection* (YOLO) dan *Semantic Segmentation* (FCN).

### Bab 15: Memproses Sekuens Menggunakan RNN dan CNN
Membahas arsitektur untuk data sekuensial (time series, teks). Bab ini memperkenalkan *Recurrent Neural Networks* (RNN) dan masalah *short-term memory*. Solusinya, LSTM dan GRU, dibahas secara rinci. Bab ini juga menunjukkan cara menggunakan 1D ConvNets untuk memproses sekuens dengan sangat cepat (misalnya, arsitektur WaveNet).

### Bab 16: Pemrosesan Bahasa Alami (NLP) dengan RNN dan Attention
Mengaplikasikan RNN untuk NLP. Bab ini mencakup pembuatan Char-RNN untuk generasi teks, analisis sentimen, dan *word embeddings*. Ini berpuncak pada arsitektur Encoder-Decoder, *Attention Mechanism* (yang merevolusi NMT), dan arsitektur *Transformer* (dasar dari model canggih seperti BERT dan GPT).

### Bab 17: Representation Learning dan Generative Learning Menggunakan Autoencoder dan GAN
Menjelajahi model *unsupervised learning* yang canggih. *Autoencoder* (Stacked, Denoising, Sparse, Variational) digunakan untuk kompresi, ekstraksi fitur, dan generasi gambar. *Generative Adversarial Networks* (GANs), termasuk DCGAN dan StyleGAN, dibahas untuk menghasilkan gambar yang sangat realistis.

### Bab 18: Reinforcement Learning
Pengantar Reinforcement Learning (RL), di mana agen belajar melalui *rewards* dan *punishments*. Bab ini mencakup konsep inti (MDP, Bellman equation) dan algoritma fundamental seperti Policy Gradients (PG) dan Deep Q-Networks (DQN). Implementasi praktis dilakukan menggunakan library TF-Agents untuk melatih agen bermain game.

### Bab 19: Melatih dan Mendeploy Model TensorFlow dalam Skala Besar
Membawa model dari eksperimen ke produksi. Bab ini mencakup cara mengekspor model ke format SavedModel dan mendeploy-nya menggunakan **TF Serving** atau **Google Cloud AI Platform**. Ini juga mencakup TFLite (untuk mobile/embedded), dan cara mendistribusikan pelatihan pada banyak GPU dan server (misalnya, `MirroredStrategy`).