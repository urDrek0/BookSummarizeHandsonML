# Bab 4 : State of the art in deep learning : Transformers

Bab ini akan membahas,
- Representasi teks dalam format numerik untuk model machine learning
- Membuat model transformasi menggunakan sub-classing API dari Keras

Transformasi adalah generasi terakhir dari deep networks. Jurnal Vaswani er al., yang berjudul "Attention is All You Need" (dapat di akses di : `https://arxiv.org/pdf/1706.03762.pdf`), mempopulerkan ide tersebut. Transformer ini di implementasikan ke company global seperi Google, OpenAI, dan Facebook, dimana model mereka terbilang besar dan melebihi kemampuan model lain di domain NLP (Natural Language Processing). Fokus dari bab ini akan mengacu pada bagaimana penggunaan Transformer di domain NLP, secara khusus di model yang bertugas utama mentranslasikan bahasa. 

---

## Representasi Teks sebagai nomor

Untuk merepresentasikan teks menjadi penomoran maka perlu untuk memahami struktur penomorannya, dan salah satu caranya adalah dengan NLP (Natural Language Processing).

Sebagai contoh, kita diberikan beberapa set kalimat, 
- I went to the beach.
- It was cold.
- I came back to the house.

Langkah pertama adalah me-nomori setiap kata yang similar dengan ID, dimulai dari 1 dengan 0 digunakan untuk karakter khusus, contohnya kita me-nomori dengan ID seperti berikut,
- I → 1
- went → 2
- to → 3
- the → 4
- beach → 5
- It → 6
- was → 7
- cold → 8
- came → 9
- back → 10
- house → 11

Sehingga diperoleh ID dan kalimat yang berkoresponden seperti berikut,
- [1, 2, 3, 4, 5]
- [6, 7, 8]
- [1, 9, 10, 3, 4, 11]

---

## Memahami Model Transformer

### Encoder-decoder pada Transformer

### Pengertian Lebih Dalam

### Layer Self-Attention

### Memeahami Self-Attention dengan skalar

### Self-Attention adalah kompetisi

### Masked Self-Attention layers

### Multi-Head Attention

### Fully Connected Layer

### Implementasi keseluruhan

---

# KESIMPULAN AKHIR

- Jaringan transformator telah mengungguli model lain di hampir semua tugas NLP.
- Transformator adalah jaringan saraf tipe encoder-decoder yang terutama digunakan untuk mempelajari tugas-tugas NLP.
- Dengan Transformer, encoder dan dekoder terdiri dari dua sublapisan komputasional: lapisan self-attention dan lapisan terhubung penuh.
- Lapisan self-attention menghasilkan jumlah masukan terbobot untuk langkah waktu tertentu, berdasarkan seberapa penting memperhatikan posisi lain dalam urutan tersebut saat memproses posisi saat ini.
- Lapisan terhubung penuh menciptakan representasi nonlinier dari keluaran yang diperhatikan yang dihasilkan oleh lapisan self-attention.
- Dekoder menggunakan masking pada lapisan self-attention-nya untuk memastikan bahwa dekoder tidak melihat prediksi di masa mendatang saat menghasilkan prediksi saat ini.
