# Bab 2: TensorFlow 2

Panduan mengenai arsitektur internal TensorFlow 2, berfokus pada struktur data dasar, operasi komputasi, dan cara kerjanya di bawah kap mesin (under the hood).

---

# 1. Evolusi dari TensorFlow 1 ke TensorFlow 2 (TF2)

Perbedaan mendasar yang harus dipahami adalah filosofi eksekusi:

- **TF1 (TensorFlow 1)**: Menggunakan mode eksekusi Grafik Statis (Static Graph). Anda harus mendefinisikan seluruh grafik komputasi terlebih dahulu (fase tf.Graph), kemudian menjalankan grafik tersebut menggunakan sesi (tf.Session). Ini cenderung kurang fleksibel dan sulit di-debug.

- **TF2 (TensorFlow 2)**: Mengadopsi Eksekusi Eager (Eager Execution) secara default. Ini memungkinkan operasi dieksekusi segera dan hasilnya dikembalikan, seperti halnya kode Python biasa. Hal ini membuat debugging dan pengembangan model jauh lebih mudah dan intuitif.

    Meskipun Eksekusi Eager adalah default, TF2 tetap efisien untuk produksi dengan menggunakan fitur @tf.function. Dekorator ini dapat mengubah fungsi Python biasa menjadi grafik komputasi TensorFlow yang dioptimalkan, mendapatkan kembali manfaat kecepatan dan efisiensi TF1 tanpa mengorbankan kemudahan debugging dari eager execution.

---

# 2. Blok Bangunan Inti TensorFlow
Semua komputasi di TensorFlow, mulai dari yang sederhana hingga jaringan saraf terdalam, dibangun dari tiga komponen dasar: Tensor, Variable, dan Operation.

**A. `tf.Tensor` (Tensor)**
Tensor adalah struktur data fundamental di TensorFlow.  Secara harfiah, Tensor adalah array multidimensi (n-dimensi).

- Peran: Mewakili semua bentuk data, termasuk masukan (input), keluaran (output), dan hasil komputasi antara. Semua data (seperti skalar, vektor, matriks, atau data gambar 4D) direpresentasikan sebagai Tensor.

- Properti Utama: Tensor memiliki properti kunci seperti:

- `dtype`: Tipe data elemen (misalnya, `tf.float32`, `tf.int64`).

- `shape`: Jumlah elemen di setiap dimensi (misalnya, shape `(3, 224, 224, 3)` untuk batch 3 gambar RGB berukuran 224x224).

- Immutabilitas: Tensor di TF2 bersifat immutable (tidak dapat diubah) setelah dibuat. Setiap operasi yang mengubah Tensor akan menghasilkan Tensor baru.

**B. `tf.Variable` (Variabel)**
Variabel adalah tensor khusus yang dirancang untuk dapat diubah (mutable).

- Peran: Digunakan untuk menyimpan status model yang perlu diperbarui selama pelatihan—terutama bobot (weights) dan bias dari lapisan jaringan saraf.

- Mutabilitas: Variabel dapat di-update (misalnya, melalui operasi assignment atau saat optimizer melakukan langkah backpropagation), yang sangat penting untuk proses training model. Variabel mempertahankan nilainya di antara eksekusi.

**C. `tf.Operation` (Operasi)**
Operasi adalah simpul (node) dalam grafik komputasi yang melakukan fungsi matematika pada satu atau lebih Tensor (sebagai masukan) dan menghasilkan Tensor baru (sebagai keluaran).

- Contoh: Operasi sederhana seperti penambahan (`tf.add`), perkalian matriks (`tf.matmul`), hingga operasi kompleks seperti konvolusi (`tf.nn.conv2d`) atau fungsi aktivasi (`tf.nn.relu`).

- Fungsi: Operasi mendefinisikan alur data dan komputasi yang membentuk model deep learning.

---

#  3. Komputasi Dasar Jaringan Saraf
Bab ini menutup dengan menunjukkan bagaimana building blocks di atas digunakan untuk operasi krusial dalam jaringan saraf:

- Perkalian Matriks (`tf.matmul`): Operasi yang mendefinisikan lapisan fully connected (dense) dalam model, di mana input dikalikan dengan bobot (weights) lapisan.

- Operasi Konvolusi (`tf.nn.conv2d`): Blok bangunan utama Convolutional Neural Networks (CNNs), digunakan untuk mengekstrak fitur spasial dari data (terutama gambar).

- Operasi Pooling: Digunakan untuk mengurangi dimensi (downsample) spasial dari data, mengurangi jumlah parameter, dan membuat model lebih kuat terhadap sedikit perubahan posisi fitur.

Pemahaman mendalam tentang Tensor, Variable, dan Operation di bawah mode Eager Execution adalah kunci yang diberikan Bab 2, yang mempersiapkan pembaca untuk membangun model deep learning yang kompleks dan efisien menggunakan API tingkat tinggi (seperti Keras) di bab-bab selanjutnya.

