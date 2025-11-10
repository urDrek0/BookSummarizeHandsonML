# Bab 4 : Dipping toes in deep learning

Bab ini akan membahas,
- Implementasi dan keseluruhan pelatihan FCN (__Fully Connected Networks__) menggunakan Keras
- Implementasi dan pelatihan CNN (__Convolutional Nerual Networks__) untuk klasifikasi gambar
- Implementasi dan pelatihan RNN (__Recurrent Neural Networks__) untuk menyelesaikan permasalahan deret waktu (__time-series__)

---

## FCN (Fully Connected Networks)

Dalam buku ini diberikan narasi yang akan menjelaskan mengenai **Hyperparameter optimization**, yaitu sebuah proses yang cukup memakan sumber daya dan juga untuk evaluasinya perlu banyak model untuk dapat dipilih set parameter yang terbaik. hal ini membuatnya sulit diaplikasikan untuk metode deep learning, karena metode ini biasanya memiliki banyak data dan model dengan tingkat kompleksitas yang tinggi.

### 1. Memahami Data
Pada contoh proses dalam buku, digunakan MNIST digit data set, yaitu sebuah hand-drawn numbers dari 0 - 9. Proses Load data MNIST di TensorFlow dapat menggunakan

`from tensorflow.keras.datasets.minst import load_data
(x_train, y_train),  (x_text,y_test) = load_data()`

Code `load.data()` memungkinkan return dari data latihan dan data yang diujikan. Dalam kasus ini dilakukan __unsupervised task__, sehingga tidak ada label untuk data latihan, sehingga tidak perlu label untuk menyelesaikan tugas ini. MNIST kini dianggap terlalu mudah untuk di identifikasi oleh __computer vision__, sehingga untuk implementasi data set lain sangat dianjurkan. Selanjutnya, untuk memahami proses pelatihan dapat digunakan `print` untuk melihat prosesnya secara langsung, seperti contoh.

`print(x_train)
print('x_train has shape {}'.format(x_train.shape))`

begitu juga berlaku dengan `y_train`,

`print(y_train)
print('y_train has shape {}'.format(y_train.shape))`

selanjutnya bisa dilakukan normalisasi sepert contoh

`norm_x_train = ((x_train - 128.0)/128.0).reshape([-1,784])`

pada contoh kasus shape dari x dan y train adalah sebanyak 60000,28,28 sehingga dengan normalisasi mengaturnya ke satu dimensional vektor sepanjang 784 satuan. Apabila kita menampilkan data maka gambar akan terlihat bisa dikenali secara sekilas, lalu bagaimana cara kita untuk dapat membuat model yang dapat mengenali bahkan ketika gambar terlihat terdistorsi? dengan code berikut akan membuat gambar terdistorsi.

`import numpy as np
def generate_masked_inputs(x, p, seed=None):
    if seed:
        np.random.seed(seed)
    mask = np.random.binomial(n=1, p=p, size=x.shape).astype('float32')
    return x * mask
masked_x_train = generate_masked_inputs(norm_x_train, 0.5)`

pada `masked_x_train = generate_masked_inputs(norm_x_train, 0.5)` memberikan kita data yang hanya dapat dikenali 50% dari data aslinya.

### 2. Model Auto Encoder

FCN dan MLP(__Multilayer Perceptron__) beroprasi secara similar, namun MLP di desain untuk menyelesaikan supervised task, dimana autoencoder di desain untuk menyelesaikan unsupervised task, maka dari itu sebenarnya MLP dan FCN adalah FCN juga.

Perbedaan antara unsupervised dan supervised utamanya terletak di karakteristik dataset, dimana unsupervised tidak berlabel dalam datanya, sedangkan supervised berlabel. Beberapa sueprvised task adalah klasifikasi gambar, deteksi objek, pengenalan suara, dan analisis sentimen. Sedangkan contoh unsupervised adalah rekonstruksi gambar, image generation dengan generative adversarial networks, klustering text, dan language modeling.

Selain itu perlu juga untuk di lakukan "denosising autoencoders", jika kita kaitkan pada contoh maka tujuannya adalah,

- Produce gambar dengan kualitas yang lebih baik
- Menghilangkan randomness variations seperti pencahayaan atau warna
- Artifacts yang dikarenakan kompresi JPEG

---

## CNN (Convolutional Neural Networks)

CNN yang digunakan untuk mengklasifikasikan gambar diimplementasikan ke sebuah model dilengkapi dengan dataset yang mencukupi serta variansi data set yang tinggi. CNN berfokus pada akurasi pengklasifikasian gambar.

### 1. Memahami Data

Pada contoh dalam buku di berikan beberapa supervised data dengan label bermacam - macam dan gambar yang bermacam - macam. Hal ini dimaksudkan untuk nantinya di klasifikasikan berdasarkan label nya. pada buku diberikan code berupa,

`import tensorflow_datasets as tfds
data = tfds.load('cifar10')`
print(data)`

Setelah data di load dan di tampilkan maka akan menampilkan hasil seperti dibawah ini,

`{'train': <_PrefetchDataset element_spec={'id': TensorSpec(shape=(), dtype=tf.string, name=None), 'image': TensorSpec(shape=(32, 32, 3), dtype=tf.uint8, name=None), 'label': TensorSpec(shape=(), dtype=tf.int64, name=None)}>, 'test': <_PrefetchDataset element_spec={'id': TensorSpec(shape=(), dtype=tf.string, name=None), 'image': TensorSpec(shape=(32, 32, 3), dtype=tf.uint8, name=None), 'label': TensorSpec(shape=(), dtype=tf.int64, name=None)}>}`

Setelahnya akan dilakukan konversi data images ke `float32` untuk membuat data yang konsisten dan label ke vektor _one-hot encoded_, dengan code di bawah ini.

`import tensorflow as tf
def format_data(x, depth):`
  return (tf.cast(x["image"], 'float32'), tf.one_hot(x["label"], depth=depth))`

lalu dengan membuat batch data set dengan fungsi,

`tr_data = data["train"].map(lambda x: format_data(x, depth=10)).batch(32)`

lalu kita dapat melihat isi datanya dengan,

`for d in tr_data.take(1):
  print(d)`

Setelahnya kita bisa memastikan bahwa data sudah siap untuk di ujikan ke model

### 2. Implementasi

CNN terkenal dalam menyelesaikan tugas computer vision dan pilihan populer untuk tugas yang melibatkan gambar di dalamnya. Hal ini ada 2 alasan yang mendasari, yaitu.

- CNN memproses gambar sambil mempertahankan informasi spasial (misalnya seperti mempertahankan width dan height gambar), sedangkan fully connected layer harus konversi menjadi satu jenis data di dimensi tunggal, hal ini dapat menyebabkan ada nya kehilangan data yang penting
- Tidak seperti fully connected layer dimana setiap input terkoneksi dengan setiap output, operasi konvolusi menggeser kernel yang lebih kecil daripada keseluruhan gambar, sehingga hanya membutuhkan parameter yang lebih sedikit, membuat CNN menjadi lebih efisien dari aspek parameter yang harus di penuhi.

CNN terdiri dari beberapa operasi konvolusi dan pooling layers, disertai beberapa fully connected layers, sehingga memiliki 3 layer utama, yaitu.
- Convolution Layers
- Pooling Layers
- Fully Conenected Layers

Sebagai Contoh, kita akan membuat CNN dengan Sequenstial API dari Keras dengan code berikut,

`from tensorflow.keras import layers, models
import tensorflow.keras.backend as K
K.clear_session()
cnn = models.Sequential(
  [layers.Conv2D(
    filters=16, kernel_size= (9,9), strides=(2,2), activation='relu',` padding='valid', input_shape=(32,32,3)
  ),
  layers.Conv2D(
    `filters=32, kernel_size= (7,7), activation='relu', padding='valid'
  ),
  layers.Conv2D(
    `filters=64, kernel_size= (7,7), activation='relu', padding='valid'
  ),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')]
)`

Hal ini akan memnuculkan pertanyaan baru, Bagaimana Hyperparameters di CNN? Pada jaringan CNN, seperti pada contoh, terdapat beberapa parameter, yaitu `filters`, `kernel_size`, `strides`, `activation`, `padding`, dan `input_shape`, selanjutnya parameter tersebut berada di layer `Conv2D`. Idealnya hyperparameters ini harus di seleksi oleh algoritma yang mensortir optimisasi hyperparameter, yang mana akan running ratusan bahkan ribuan model dengan hyperparameters yang berbeda - beda untuk menentukan hyperparamters mana yang lebih optimal untuk akurasi hasil pelatihannya.

Maka jika kita rangkum secara singkat,

|                      | Dimensionality                            | Contoh                                                                             |
|----------------------|:-----------------------------------------:|:-----------------------------------------------------------------------------------:|
| Input                |  batch size, height, width, in channels   | [32, 64, 64, 3] (i.e., a batch of 32, 64 × 64 RGB images)                           |
| Filter Konvolusi     |  height, width out channels               | [5, 5, 3, 16] (i.e., 16 convolution filters ofsize 5 × 5 with 3 incoming channels)  |
| Output               |  batch size, height, width, out channels  | [32, 64, 64, 16] (i.e., a batch of 32, 64 ×64 × 16 tensors)                         |

Permasalahannya saat membuat deep models, khususnya adalah ini membatai jumlah layers yang kita punya, pada suatu waktu input nya akan menjadi 1x1 pixel karena automatisasi pengurangan dimensi, sehingga akan membuat kita kehilangan beberapa informasi (akibat bottleneck) saat data di transfer ke fully connected layers. Salah satu solusi yang dapat diterapkan adalah padding, untuk contoh solusi padding jelasnya bisa di lihat di link ini: `https://www.tensorflow.org/api_docs/python/tf/pad`, dimana padding meliputi,
- Nilai konstan
- Refleksi input
- Nilai terdekat

---

## RNN (Recurrent Neural Networks)

RNN memiliki kemampuan khusus yang CNN dan FCN tidak miliki, yaitu dapat membaca dan mempelajari data dengan deret waktu (__time-series__). Sebenarnya FCN dan CNN dapat membaca data deret waktu, namun perlu perlakuan dan adaptasi yang khusus. RNN tidak hanya digunakan untuk membuat prediksi, tapi juga menggunakan ingatan dari jaringan lama untuk menentukan step pada waktu tertentu.

### 1. Memahami Data

Prosesnya kurang lebih sama dengan diatas berupa load data, lalu menampilkan data, dengan code berikut.

`import requests
import os
def download_data():
  """ This function downloads the CO2 data from
  https:/ /datahub.io/core/co2-ppm/r/co2-mm-gl.csv
  if the file doesn't already exist
  """
  save_dir = "data"
  save_path = os.path.join(save_dir, 'co2-mm-gl.csv')
  # Create directories if they are not there
  if not os.path.exists(save_dir):
    `os.makedirs(save_dir)
  # Download the data and save
  if not os.path.exists(save_path):
    url = "https:/ /datahub.io/core/co2-ppm/r/co2-mm-gl.csv"
    r = requests.get(url)
    with open(save_path, 'wb') as f:
      f.write(r.content)`
  else:
    print("co2-mm-gl.csv already exists. Not downloading.")
  return save_path
# Downloading the data
save_path = download_data()`


Atau dapat juga di download melalui `https://datahub.io/core/co2-ppm/r/co2-mm-gl.csv`,  lalu menampilkan data head dengan pandas

`import pandas as pd
data = pd.read_csv(save_path)
data.head()`

setelahnya dapat kita sort data berdasarkan `date` dan menampikannya dalam bentuk grafik untuk memperoleh visualisasi data yang memudahkan kita untuk menganalisis permasalahannya.

`data = data.set_index('Date')
data[["Average"]].plot(figsize=(12,6))`

setelahnya kita dapat menampilkan sejauh apa data dengan rata - rata nya.

`data["Average Diff"]=data["Average"] - data["Average"].shift(1).fillna(method='bfill')`

setelahnya kita dapat lakukan sebuah proses agar data berada di single positions, sehingga data dapat dicacah dalam jumlah yang lebih kecil untuk di berikan kepada model untuk belajar.

`import numpy as np
def generate_data(co2_arr,n_seq):
  x, y = [],[]
  for i in range(co2_arr.shape[0]-n_seq):
    x.append(co2_arr[i:i+n_seq-1])
    y.append(co2_arr[i+n_seq-1:i+n_seq])
  x = np.array(x)
  y = np.array(y)
  return x,y`

### 2. Implementasi Model

Ada beberapa hal yang akan di impelementasikan, yaitu.
- Layer RNN dengan 64 hidden units
- Layer Dense dengan 64 hidden units dan aktivasi ReLU
- Layer Dense dengan single output dan aktivasi linear
berikut adalah code yang digunakan,

`from tensorflow.keras import layers, models
rnn = models.Sequential([
  layers.SimpleRNN(64),
  layers.Dense(64, activation='relu'),
  layers.Dense(1)
])`

dengan begitu kita dapat meninjau ulang algoritma yang digunakan untuk RNN sederhana, yang biasa dikenal dengan _Elman-networks_. Algoritma ini ditemukan pada tahun 1990 oleh J.L Elman melalui jurnalnya yang berjudul "Finding Structure in Time", untuk mengetahui lebih lanjut dapat di tinjau melalui link : `http://mng.bz/xnJg` dan `http://mng.bz/Ay2g`

Pada kasus ini yang akan kita prediksi adalah kandungan CO2 di masa depan, dengan beberapa langkah implementasi dan prediksi maka kita bisa menggunakan code,

`rnn.compile(loss='mse', optimizer='adam')
x, y = generate_data(data[“Average Diff”], n_seq=13)
rnn.fit(x, y, shuffle=True, batch_size=64, epochs=25)`

Dengan optimizer menggunakan adam, Kita akan memperoleh output:

`ValueError:
Input 0 of layer sequential_1 is incompatible with the layer:
expected ndim=3, found ndim=2. Full shape received: [None, 12]`

Permasalahan pada RNN sederhana adalah hanya dapat menerima format tertentu, dimana data harus memiliki 3 dimensi utama, yaitu,
- Batch
- Waktu
- Fitur

Maka kita akan mencoba code untuk mencacah data dalam jumlah kecil,

`import numpy as np
def generate_data(co2_arr,n_seq):
  x, y = [],[]
  for i in range(co2_arr.shape[0]-n_seq):
    x.append(co2_arr[i:i+n_seq-1])
    y.append(co2_arr[i+n_seq-1:i+n_seq])
  x = np.array(x).reshape(-1,n_seq-1,1)
  y = np.array(y)
  return x,y`

lalu melatih model kembali

`x, y = generate_data(data[“Average Diff”], n_seq=13)
rnn.fit(x, y, shuffle=True, batch_size=64, epochs=25)`

Selanjutnya kita akan memprediksi CO2 dengan model yang sudah dilatih, menggunakan code

`history = data["Average Diff"].values[-12:].reshape(1,-1,1)
true_vals = []
prev_true = data["Average"].values[-1]
for i in range(60):
  p_diff = rnn.predict(history).reshape(1,-1,1)
  history = np.concatenate((history[:,1:,:],p_diff),axis=1)
  true_vals.append(prev_true+p_diff[0,0,0])`
  prev_true = true_vals[-1]`

pertama kita ekstrak data untuk kurun waktu 12 tahun terakhir

`history = data["Average Diff"].values[-12:].reshape(1,-1,1)`

kita akan mengambil data paling terakhir di dalam kolom `average`

`prev_true = data["Average"].values[-1]`

Sekarang kita akan mencoba untuk 5 tahun ke depan

`history = np.concatenate((history[:,1:,:],p_diff),axis=1)`

lalu kita memprediksi

`true_vals.append(prev_true+p_diff[0,0,0])`

Setelahnya kita mengupdate `prev_true` dengan data prediksi terbaru

`prev_true = true_vals[-1]`
 
---

# KESIMPULAN AKHIR

- Jaringan yang terhubung penuh (FCN) adalah salah satu jaringan saraf tiruan yang paling mudah.
- FCN dapat diimplementasikan menggunakan lapisan Keras Dense.
- Jaringan saraf tiruan konvolusional (CNN) adalah pilihan populer untuk tugas-tugas visi komputer.
- TensorFlow menawarkan berbagai lapisan, seperti Conv2D, MaxPool2D, dan Flatten, yang membantu kita mengimplementasikan CNN dengan cepat.
- CNN memiliki parameter seperti ukuran kernel, langkah, dan padding yang harus diatur dengan hati-hati. Jika tidak, hal ini dapat menyebabkan tensor yang bentuknya tidak tepat dan kesalahan runtime.
- Jaringan saraf tiruan rekuren (RNN) terutama digunakan untuk belajar dari data deret waktu.
- RNN pada umumnya mengharapkan data untuk diorganisasikan ke dalam tensor tiga dimensi dengan dimensi batch, waktu, dan fitur. 
- Jumlah langkah waktu yang diamati RNN merupakan hiperparameter penting yang harus dipilih berdasarkan data.