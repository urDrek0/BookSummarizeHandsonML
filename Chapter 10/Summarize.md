# Bab 10: Pengenalan Jaringan Saraf Tiruan dengan Keras

Bab ini akan membahas:

  * Inspirasi biologis dan konsep dasar di balik Jaringan Saraf Tiruan (ANN).
  * Evolusi dari Perceptron, unit komputasi sederhana, hingga Multi-Layer Perceptrons (MLP).
  * Algoritma fundamental *Backpropagation* untuk melatih Jaringan Saraf Tiruan yang dalam (DNN).
  * Implementasi praktis model MLP untuk tugas regresi dan klasifikasi menggunakan API tingkat tinggi **Keras**.
  * Tiga cara membangun model di Keras: Sequential API, Functional API, dan Subclassing API.
  * Teknik penting seperti menyimpan model, *callbacks* (seperti *early stopping*), dan visualisasi dengan **TensorBoard**.
  * Cara melakukan *fine-tuning* (penyetelan) *hyperparameter* Jaringan Saraf Tiruan.

-----

## 1 Dari Neuron Biologis ke Neuron Buatan

ANN terinspirasi oleh jaringan neuron biologis di otak kita[cite: 1183, 1186].

  * **Neuron Biologis**: Menerima sinyal dari neuron lain melalui *dendrit*, memprosesnya, dan mengeluarkan sinyalnya sendiri melalui *akson* [cite: 1185-1186].
  * **Neuron Buatan (TLU)**: Model komputasi sederhana yang terinspirasi dari ini pertama kali diusulkan pada tahun 1943. Ia menghitung *weighted sum* (jumlah berbobot) dari input-inputnya, lalu menerapkan *step function* (fungsi langkah) pada hasil jumlah tersebut untuk menghasilkan output [cite: 1184, 1188-1189].

### Perceptron

Perceptron adalah arsitektur ANN sederhana, ditemukan oleh Frank Rosenblatt pada tahun 1957, yang didasarkan pada neuron buatan (TLU)[cite: 1188].

  * Ia terdiri dari satu lapisan TLU, dengan setiap TLU terhubung ke semua input[cite: 1189].
  * Ketika setiap neuron terhubung ke setiap neuron di lapisan sebelumnya, ini disebut **fully connected layer** (lapisan terhubung penuh) atau **dense layer**[cite: 1189].
  * **Pelatihan**: Perceptron dilatih menggunakan aturan yang memperkuat koneksi yang membantu mengurangi kesalahan prediksi. Scikit-Learn menyediakan kelas `Perceptron` yang mengimplementasikan ini.
  * **Keterbatasan**: Perceptron (sebagai *linear classifier*) tidak mampu menyelesaikan beberapa masalah sepele, seperti masalah klasifikasi **XOR**[cite: 1192].

-----

## 2 Multi-Layer Perceptron (MLP) dan Backpropagation

Keterbatasan Perceptron dapat diatasi dengan menumpuk beberapa lapisan TLU. Arsitektur yang dihasilkan disebut **Multi-Layer Perceptron (MLP)**.

  * **Arsitektur**: MLP terdiri dari satu *input layer* (lapisan input), satu atau lebih *hidden layers* (lapisan tersembunyi), dan satu *output layer* (lapisan output)[cite: 1193]. Ketika MLP memiliki banyak *hidden layers*, ia disebut **Deep Neural Network (DNN)**[cite: 1193].
  * **Fungsi Aktivasi**: Untuk melatih MLP, *step function* diganti dengan fungsi yang dapat didiferensiasi (memiliki turunan) seperti:
      * **Fungsi Logistik (Sigmoid)**: Menghasilkan output antara 0 dan 1.
      * **Fungsi Hyperbolic Tangent (tanh)**: Menghasilkan output antara -1 dan 1. Seringkali konvergen lebih cepat.
      * **Fungsi ReLU (Rectified Linear Unit)**: `ReLU(z) = max(0, z)`. Ini adalah fungsi aktivasi default yang paling populer karena cepat dihitung dan tidak mengalami saturasi untuk nilai positif[cite: 1196, 1205].

### Backpropagation

*Backpropagation* adalah algoritma pelatihan fundamental untuk MLP. Ini pada dasarnya adalah **Gradient Descent** yang efisien untuk jaringan saraf[cite: 1194].

1.  **Forward Pass**: Data *mini-batch* dilewatkan melalui jaringan dari lapisan input ke output. Hasil perantara dari setiap lapisan disimpan.
2.  **Measure Error**: *Loss function* (fungsi kerugian) menghitung seberapa salah prediksi jaringan.
3.  **Backward Pass**: Algoritma menghitung seberapa besar kontribusi setiap koneksi (bobot) terhadap kesalahan, bekerja secara mundur dari lapisan output ke input. Ini dilakukan dengan menggunakan *chain rule* (aturan rantai) kalkulus.
4.  **Gradient Descent Step**: Bobot jaringan diperbarui (disesuaikan) untuk mengurangi kesalahan, menggunakan gradien yang dihitung pada langkah 3.

-----

## 3 Implementasi MLP dengan Keras

Keras adalah API deep learning tingkat tinggi yang kuat dan mudah digunakan, yang terintegrasi penuh ke dalam TensorFlow sebagai `tf.keras` [cite: 1199-1200].

### Membangun Image Classifier (Klasifikasi Gambar)

Bab ini menggunakan dataset **Fashion MNIST** (pengganti MNIST yang lebih menantang)[cite: 1201].

**1. Sequential API (API Sekuensial)**:
Cara paling sederhana untuk membangun model adalah dengan `Sequential`, yang merupakan tumpukan lapisan linear[cite: 1203].

```python
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]), # Meratakan gambar 28x28 jadi vektor 784
    keras.layers.Dense(300, activation="relu"),  # Hidden layer pertama (300 neuron)
    keras.layers.Dense(100, activation="relu"),  # Hidden layer kedua (100 neuron)
    keras.layers.Dense(10, activation="softmax") # Output layer (10 kelas, softmax untuk probabilitas)
])
```

**2. Compiling (Kompilasi Model)**:
Sebelum pelatihan, model harus di-`compile`, yang menentukan *loss function*, *optimizer*, dan *metrics*[cite: 1206].

  * **Loss**: Menggunakan `"sparse_categorical_crossentropy"` karena kita memiliki label *sparse* (hanya indeks kelas, misal: 9) dan kelas-kelasnya *exclusive*.
  * **Optimizer**: `"sgd"` memberi tahu Keras untuk menggunakan Stochastic Gradient Descent.
  * **Metrics**: `["accuracy"]` digunakan untuk memantau akurasi selama pelatihan.

**3. Training dan Evaluating (Pelatihan dan Evaluasi)**:
Model dilatih menggunakan metode `fit()`. Metode `evaluate()` digunakan pada *test set* untuk mengukur *generalization error*.

### Membangun Regression MLP (Regresi MLP)

MLP juga dapat digunakan untuk regresi. Perbedaan utamanya adalah [cite: 1211-1212]:

  * **Output Layer**: Hanya memiliki satu neuron (jika memprediksi satu nilai) dan **tidak menggunakan fungsi aktivasi**.
  * **Loss Function**: Menggunakan `"mean_squared_error"` (MSE)[cite: 1212].

-----

## 4 API Keras Tingkat Lanjut

### Functional API

Untuk arsitektur yang lebih kompleks (non-sekuensial), seperti **Wide & Deep network** (jaringan Lebar & Dalam), kita menggunakan Functional API. API ini memungkinkan Anda menghubungkan lapisan-lapisan secara eksplisit.

  * **Multiple Inputs**: Berguna saat Anda ingin memproses subset fitur yang berbeda secara terpisah (misalnya, beberapa fitur melewati *deep path* dan beberapa melewati *wide path*) [cite: 1214-1215].
  * **Multiple Outputs**: Berguna untuk *multitask learning* atau untuk regularisasi (misalnya, menambahkan *auxiliary output* di lapisan yang lebih rendah) [cite: 1215-1216].

### Subclassing API

Untuk model yang sangat dinamis (misalnya, yang melibatkan *loop* atau percabangan kondisional), Anda dapat membuat *subclass* dari `keras.Model`.

  * Lapisan-lapisan didefinisikan dalam konstruktor (`__init__`).
  * Logika *forward pass* didefinisikan dalam metode `call()` [cite: 1217-1218].
  * Ini memberikan fleksibilitas terbesar tetapi bisa lebih sulit untuk di-debug dan dianalisis.

-----

## 5 Menyimpan, Memuat, dan Callbacks

  * **Saving/Loading**: Menyimpan model (arsitektur, bobot, dan optimizer) sangat mudah: `model.save("my_model.h5")` dan `model = keras.models.load_model("my_model.h5")` [cite: 1218-1219].
  * **Callbacks**: *Callbacks* adalah objek yang dapat diteruskan ke metode `fit()` agar Keras memanggilnya pada titik-titik tertentu selama pelatihan (misalnya, di akhir setiap epoch).
      * **`ModelCheckpoint`**: Menyimpan *checkpoints* model Anda secara berkala, seringkali hanya menyimpan model terbaik (`save_best_only=True`)[cite: 1219].
      * **`EarlyStopping`**: Menghentikan pelatihan secara otomatis ketika performa pada *validation set* tidak membaik selama beberapa epoch (ditentukan oleh `patience`), untuk mencegah *overfitting* [cite: 1219-1220].

## 6 TensorBoard dan Fine-Tuning

  * **TensorBoard**: Alat visualisasi yang hebat. Anda dapat menggunakannya sebagai *callback* (`keras.callbacks.TensorBoard()`) untuk memvisualisasikan *learning curves* (kurva pembelajaran), arsitektur model, dan statistik pelatihan lainnya secara *real-time* [cite: 1221-1222].
  * **Fine-Tuning Hyperparameters**: Untuk menemukan kombinasi *hyperparameter* terbaik (misalnya, jumlah *layer*, jumlah neuron, *learning rate*), Anda dapat membungkus model Keras Anda menggunakan `KerasRegressor` atau `KerasClassifier` (dari `keras.wrappers.scikit_learn`) dan kemudian menggunakan *tools* Scikit-Learn seperti `RandomizedSearchCV` [cite: 1224-1225].

-----

# KESIMPULAN AKHIR

  * Bab ini memperkenalkan **Jaringan Saraf Tiruan (ANN)**, mulai dari inspirasi biologisnya hingga implementasi praktis model yang dalam (Deep Neural Networks).
  * **Perceptron** adalah unit dasar yang dibatasi pada masalah linear, sedangkan **MLP** menggunakan beberapa lapisan dan fungsi aktivasi non-linear (seperti **ReLU**) untuk memecahkan masalah kompleks.
  * **Backpropagation** adalah algoritma optimisasi inti (berbasis Gradient Descent) untuk melatih MLP.
  * **Keras** menyediakan tiga API untuk membangun model:
      * **Sequential API**: Paling sederhana, untuk tumpukan lapisan.
      * **Functional API**: Untuk arsitektur non-sekuensial seperti *multiple inputs/outputs*.
      * **Subclassing API**: Untuk model dinamis dengan logika imperatif.
  * Teknik-teknik penting seperti **Callbacks** (`EarlyStopping`, `ModelCheckpoint`) dan visualisasi **TensorBoard** sangat krusial untuk alur kerja pelatihan yang efisien.
  * *Hyperparameter* ANN (jumlah *layer*, neuron, *learning rate*) dapat dioptimalkan menggunakan teknik seperti **Randomized Search**.