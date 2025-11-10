# Bab 3: Keras dan Pengambilan Data di TensorFlow 2

Bab ini dibagi menjadi dua fokus utama: cara membuat arsitektur model menggunakan berbagai API Keras, dan cara menyiapkan data pipeline yang efisien menggunakan fungsionalitas TF2.

---

# 1. API Pembuatan Model Keras (Keras Model-Building APIs) 

Keras menyediakan tiga cara utama untuk mendefinisikan arsitektur jaringan saraf, masing-masing dengan tingkat fleksibilitas yang berbeda:

**A. The Sequential API**

- Deskripsi: Ini adalah API yang paling sederhana dan paling mudah dipelajari.

- Cara Kerja: Model Sequential didefinisikan sebagai urutan linear lapisan (layer). 

- Input mengalir dari satu lapisan ke lapisan berikutnya tanpa percabangan.

- Cocok Untuk: Model sederhana yang memiliki satu input dan satu output, serta tidak memerlukan arsitektur yang kompleks (misalnya, Jaringan Fully Connected dasar).


**B. The Functional API**

- Deskripsi: Memberikan fleksibilitas yang jauh lebih besar daripada Sequential API.

- Cara Kerja: Memungkinkan Anda mendefinisikan model sebagai grafik lapisan yang tidak harus linear. Anda dapat menentukan 

- input dan output model secara eksplisit, memungkinkan adanya percabangan, koneksi multi-input, koneksi multi-output, dan berbagi lapisan (seperti arsitektur Residual Networks atau Inception).

- Cocok Untuk: Sebagian besar model deep learning modern yang membutuhkan arsitektur kompleks dan kustomisasi alur data.

**C. The Sub-Classing API**

- Deskripsi: Ini adalah API paling fleksibel dan digunakan untuk kustomisasi tingkat lanjut.

- Cara Kerja: Anda mendefinisikan model dengan membuat sub-class dari kelas tf.keras.Model dan mendefinisikan lapisan di dalam metode __init__, lalu menentukan alur forward pass (komputasi) di dalam metode call(self, inputs).

- Cocok Untuk: Peneliti atau pengembang yang perlu melakukan logika komputasi khusus di dalam forward pass yang tidak dapat diekspresikan dengan Sequential atau Functional API (misalnya, membuat lapisan kustom atau model dinamis).

---

# 2. Pengambilan Data untuk Model TensorFlow/Keras (Retrieving Data) 

Bagian kedua bab ini sangat penting untuk memastikan data dimasukkan ke model secara efisien dan dalam format yang benar. Tiga mekanisme utama dibahas:

**A. `tf.data` API**

- Deskripsi: Merupakan cara paling efisien dan berkinerja tinggi untuk membangun data pipeline di TensorFlow.

- Fungsi: Memungkinkan Anda untuk memuat, mentransformasi, dan meng-augmentasi data dalam memori atau dari disk. Fungsi utamanya adalah membuat 

- Pipeline yang dapat secara otomatis melakukan pre-fetching dan paralelisme, sehingga GPU atau CPU tidak perlu menunggu data.

- Keunggulan: Penting untuk melatih model pada dataset besar di mana data tidak muat di dalam memori.


**B. Keras `DataGenerators` (Khususnya `ImageDataGenerator`)**

- Deskripsi: Sebuah API warisan Keras, tetapi masih sangat berguna, terutama untuk tugas-tugas Visi Komputer.

- Fungsi: Digunakan untuk memuat data dari disk dan, yang paling penting, melakukan augmentasi data saat itu juga (on-the-fly) (misalnya, rotasi, zoom, shift gambar) untuk meningkatkan generalisasi model dan mengurangi overfitting.

**C. `tensorflow-datasets` Package**

- Deskripsi: Sebuah utilitas yang menawarkan koleksi besar dataset yang siap digunakan untuk deep learning.

- Fungsi: Menyediakan cara mudah untuk mengunduh, menyiapkan (pre-process), dan memuat dataset terkenal (seperti MNIST, CIFAR, atau ImageNet) langsung sebagai objek `tf.data.Dataset` yang sudah dikonfigurasi.

Singkatnya, Bab 3 adalah tentang practicality. Ini mengajarkan pembaca untuk beralih dari pemahaman konseptual (Bab 2) ke implementasi model yang sesungguhnya, menggunakan Keras untuk struktur dan `tf.data` untuk data yang efisien.
