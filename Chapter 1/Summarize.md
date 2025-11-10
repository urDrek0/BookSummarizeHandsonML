# Bagian 1: Dasar-Dasar TensorFlow 2 dan Deep Learning 
---
**Bab 1: Dunia TensorFlow yang Menakjubkan**

Bab pengantar ini memberikan gambaran umum tingkat tinggi tentang apa itu TensorFlow dan posisinya dalam lanskap machine learning.

# Teori
- **Definisi TensorFlow**: TensorFlow adalah sebuah kerangka kerja machine learning end-to-end yang dirancang untuk berjalan cepat pada perangkat keras yang dioptimalkan seperti GPU dan TPU. TensorFlow juga mendukung seluruh siklus hidup model, mulai dari pengembangan, deployment, hingga monitoring.
- **Perbandingan CPU vs GPU vs TPU**: 
    - CPU digambarkan sebagai komponen yang mampu menyelesaikan instruksi kompleks dengna waktu singkat.
    - GPU digambarkan sebagai sebuah komponen yang mampu untuk menjalankan sebuah instruksi sederhana dengan skala yang lebih besar.
    - TPU digambarkan sebagai komponen yang mampu untuk menyelesaikan sebuah instruksi yang spesifik, namun tidak fleksibel atau mudah untuk di aplikasikan jika dibandingkan dengan CPU dan GPU
- **Kapan Menggunakan TensorFlow?** Penggunaan TensorFlow optimal untuk,
    - Prototyping model deep learning
    
    Karena TensorFlow mendukung untuk Layer khusus untuk koneksi ke internet atau jaringan, konvolusi layer untuk neural network konvolusional, dan layer RNN (recurrent neural network)/LSTM (long short-term memory)/GRU (gated recurrent unit) untuk model sekuensial.

    - Implementasi model yang butuh akselerasi perangkat keras
    
    TensorFlow memiliki kernels yang dioptimasi untuk akselerasi di GPU dan TPU, jika model yang di develope bisa memanfaatkan kelebihan tersbuet (seperti regresi linier) maka  run model untuk data dalam jumlah besar secara berulang-kali, TensorFlow akan membantu untuk model berjalan lebih cepat.

    - Productionizing model di cloud
    
    Menggunakan TensorFlow memungkinkan untuk developement model-serving API via TensorFlow, lebih dari itu, jika kita mengintegrasikan CPU atau TPU, TensoFlow akan menggunakannya dalam membuat prediksi.

    - Monitoring model saat training model
    
    Melatih deep learning membutuhkan kesabaran ekstra, bahkan dengan akses ke GPU, dikarenakan kebutuhan komputasional yang tinggi. Hal tersebut akan membuat monitoring pelatihan model menjadi sulit, maka TensorFlow menyediakan log dan persistensi metriks performa yang akan berguna untuk referensi.

    - Membuat data pipeline yang padat
    
    TensorFlow menyediakan API untuk streaming data, salah satunya untuk keperluan deep learning. Yang perlu dilakukan adalah memahami fungsi dari syntax yang sudah di sediakan dan memanfaatkannya sebaik mungkin. Beberapa contoh skenario adalah seperti,
        - Pipeline yang mengonsumsi banyak gambar dan melakukan preprocessing
        - Pipeline yang mengonsumsi banyak data terstruktur di format standar
        - Pipeline yang mengonsumsi banyak data teks dan preprocessing sederhana

- **Kapan Tidak Menggunakan TensorFlow?** Sebaliknya hindari TensorFlow untuk, 
    - Model machine learning tradisional
    
    Machine Learning memiliki portofolio model yang besar (seperti regresi linier/logistik, mendukung vector machines, decision trees, k-means) yang termasuk kedalam banyak kategori dan memiliki motivasi yang berbeda, pendekatan, kelebihan, dan kelemahan.

    - Manipulasi data terstruktur skala kecil
    
    Ketika kita menggunakan data dengan skala yang kecil, maka akan sangant mudah untuk masuk ke memori, karena hal tersebut pandas dan NumPy cenderung lebih cocok untuk eksplorasi dan analisis data, untuk menghindari overfitting.

    - Pipeline NLP yang kompleks
    
    NLP biasanya jarang untuk memindahkan data ke model tanpa preprocessing sederhana (text lowering, menghilangkan tanda baca), namun tahapan asli yang akan menentukan hasil akhir dari preprocessing adalah use case dan model nya. Tahapan seperti lemmatization, stemming, spelling correction, dan preprocessing akan memenuhi pipeline preprocessing, TensorFlow akan menghambat progress ini, untuk kasus ini lebih cocok menggunakan spaCy karena menyediakan antarmuka yang intuitif dan model yang lebih baik untuk tugas pemrosesan standar NLP.

- **"Jika tidak menggunakan TensorFlow, lalu apa solusinya?"** Disarankan menggunakan pustaka lain seperti Scikit-learn, Pandas/NumPy, atau spaCy.

- **Rangkuman waktu menggunakan TensorFlow**

| Kondisi                                                     | Ya  | Tidak |
|--------------------------------------------------------------|:---:|:-----:|
| Prototyping model deep learning                              |  ✓  |       |
| Implementasi model yang butuh akselerasi perangkat keras     |  ✓  |       |
| Productionizing model di cloud                               |  ✓  |       |
| Monitoring model saat training model                         |  ✓  |       |
| Membuat data pipeline yang padat                             |  ✓  |       |
| Model machine learning tradisional                           |      |  ✓    |
| Manipulasi data terstruktur skala kecil                       |      |  ✓    |
| Pipeline NLP yang kompleks                                   |      |  ✓    |

Note : PRX Something is GOAT, ya hear me? He is the GOAT


