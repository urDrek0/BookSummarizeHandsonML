# Bab 7: Ensemble Learning dan Random Forests

Bab ini akan membahas:
* Konsep "wisdom of the crowd" (kebijaksanaan orang banyak) dan cara menggabungkan beberapa model untuk mendapatkan prediksi yang lebih baik.
* Teknik *Ensemble Learning* (Pembelajaran Ansambel) yang populer: Voting, Bagging, dan Pasting.
* Pengenalan Random Forests, salah satu algoritma ML yang paling kuat.
* Metode Boosting (AdaBoost dan Gradient Boosting) yang melatih model secara sekuensial.
* Teknik Stacking, yang melatih model untuk mengagregasi prediksi model lain.

---

## 1. Voting Classifiers (Pengklasifikasi Voting)

Ide inti dari *Ensemble Learning* adalah **wisdom of the crowd**: Mengagregasi prediksi dari sekelompok model (disebut *ensemble*) seringkali menghasilkan prediksi yang lebih baik daripada prediksi model individu terbaik sekalipun[cite: 1094].

Sebuah ansambel dapat berhasil bahkan jika setiap model adalah *weak learner* (model yang performanya sedikit lebih baik dari tebakan acak), asalkan ada cukup banyak *weak learner* dan mereka cukup **beragam** (membuat tipe kesalahan yang berbeda)[cite: 1096].

* **Hard Voting**: Mengambil prediksi dari setiap pengklasifikasi dan memilih kelas yang mendapatkan suara terbanyak sebagai prediksi akhir[cite: 1095].
* **Soft Voting**: Jika semua pengklasifikasi dapat mengestimasi probabilitas kelas, Anda dapat menghitung rata-rata probabilitas untuk setiap kelas dan memprediksi kelas dengan probabilitas rata-rata tertinggi. Ini seringkali berkinerja lebih baik karena memberi bobot lebih pada prediksi yang sangat yakin[cite: 1097].

## 2. Bagging dan Pasting

Kedua metode ini menggunakan algoritma pelatihan yang sama untuk setiap *predictor* (model), tetapi melatihnya pada subset acak yang berbeda dari *training set*[cite: 1097].

* **Bagging (Bootstrap Aggregating)**: Setiap *predictor* dilatih pada subset data yang diambil **dengan** pengembalian (*with replacement*). Ini berarti beberapa instans mungkin diambil beberapa kali untuk satu *predictor*, sementara yang lain mungkin tidak diambil sama sekali[cite: 1098].
* **Pasting**: Setiap *predictor* dilatih pada subset data yang diambil **tanpa** pengembalian (*without replacement*)[cite: 1098].

Setelah semua *predictor* dilatih, ansambel membuat prediksi dengan mengagregasi prediksi mereka (biasanya menggunakan *statistical mode* atau "suara terbanyak" untuk klasifikasi, dan rata-rata untuk regresi)[cite: 1098]. Metode ini sangat **scalable** karena *predictor* dapat dilatih dan membuat prediksi secara paralel.

### Out-of-Bag (OOB) Evaluation
Saat menggunakan *bagging*, rata-rata hanya sekitar 63% instans latih yang diambil sampelnya untuk setiap *predictor*. Sisa 37% instans yang tidak diambil sampelnya disebut **out-of-bag (OOB) instances**[cite: 1100].

Karena *predictor* tidak pernah "melihat" instans OOB selama pelatihan, instans ini dapat digunakan sebagai *validation set* (set validasi) tanpa perlu menyisihkan data validasi secara manual. Ini berarti Anda dapat menggunakan lebih banyak data untuk pelatihan dan tetap mendapatkan evaluasi yang *unbiased*[cite: 1101].

## 3. Random Forests

Random Forest adalah ansambel dari Decision Trees[cite: 1102], yang biasanya dilatih menggunakan metode *bagging*.

Selain mengambil sampel instans secara acak (seperti *bagging*), Random Forest juga memperkenalkan keacakan ekstra saat menumbuhkan pohon: alih-alih mencari fitur terbaik saat membelah *node*, algoritma ini mencari fitur terbaik di antara **subset fitur yang dipilih secara acak**[cite: 1102].

Ini menghasilkan pohon yang lebih beragam dan sedikit meningkatkan *bias* namun dapat **menurunkan *variance*** secara signifikan, yang pada akhirnya menghasilkan model keseluruhan yang lebih baik[cite: 1102].

* **Extra-Trees (Extremely Randomized Trees)**: Varian yang lebih acak lagi. Alih-alih mencari *threshold* (ambang batas) terbaik untuk setiap fitur, ia menggunakan *threshold* acak. Ini membuat pelatihan lebih cepat[cite: 1103].
* **Feature Importance (Pentingnya Fitur)**: Random Forest menyediakan cara mudah untuk mengukur pentingnya relatif dari setiap fitur. Scikit-Learn melakukannya dengan melihat seberapa besar *node* pohon yang menggunakan fitur tersebut berhasil mengurangi *impurity* (ketidakmurnian)[cite: 1103].

## 4. Boosting

Boosting adalah metode ansambel yang melatih *predictor* secara **sekuensial**, di mana setiap *predictor* baru mencoba **memperbaiki kesalahan** *predictor* sebelumnya[cite: 1104]. Dua metode yang paling populer adalah:

### AdaBoost (Adaptive Boosting)
Cara kerja AdaBoost adalah dengan **meningkatkan bobot (weight)** dari instans latih yang salah diklasifikasikan oleh *predictor* sebelumnya[cite: 1105]. *Predictor* berikutnya kemudian dilatih dan dipaksa untuk lebih fokus pada kasus-kasus sulit tersebut.

Untuk membuat prediksi, AdaBoost menggunakan *weighted vote*, di mana *predictor* yang lebih akurat (memiliki *error rate* lebih rendah) diberi bobot yang lebih tinggi[cite: 1107].

### Gradient Boosting
Metode ini juga bekerja secara sekuensial, tetapi alih-alih mengubah bobot instans, ia melatih *predictor* baru pada **residual errors** (kesalahan residual) yang dibuat oleh *predictor* sebelumnya[cite: 1108].

* **Shrinkage**: Teknik regularisasi penting dalam Gradient Boosting adalah *hyperparameter* `learning_rate`. Nilai yang rendah (misalnya 0.1) "menyusutkan" kontribusi dari setiap pohon baru. Ini berarti Anda memerlukan lebih banyak pohon dalam ansambel, tetapi modelnya biasanya menggeneralisasi lebih baik[cite: 1110].
* **Stochastic Gradient Boosting**: Menggunakan sebagian kecil (fraksi) dari data latih untuk melatih setiap pohon, yang mempercepat pelatihan dan menambah keacakan untuk mengurangi *overfitting*[cite: 1112].

## 5. Stacking (Stacked Generalization)

Stacking adalah pendekatan ansambel yang didasarkan pada ide sederhana: alih-alih menggunakan fungsi sepele (seperti voting) untuk mengagregasi prediksi, mengapa kita tidak **melatih sebuah model untuk melakukan agregasi** tersebut?[cite: 1113].

1.  Dataset latih dibagi menjadi beberapa bagian.
2.  Sekelompok model (disebut *layer 1*) dilatih pada bagian pertama.
3.  Model-model *layer 1* ini kemudian digunakan untuk membuat prediksi pada bagian data yang lain (disebut *hold-out set*)[cite: 1114].
4.  Prediksi-prediksi ini (yang "bersih" karena dibuat pada data yang tidak dilihat model) digunakan sebagai fitur untuk melatih model baru (disebut **blender** atau **meta learner**)[cite: 1114].
5.  *Blender* inilah yang membuat prediksi akhir.

---

# KESIMPULAN AKHIR

* *Ensemble Learning* menggabungkan beberapa model ML (disebut *predictor*) untuk menghasilkan performa yang lebih baik daripada model individu.
* **Voting** adalah cara termudah, dengan mengambil suara mayoritas (hard) atau rata-rata probabilitas (soft).
* **Bagging** dan **Pasting** melatih *predictor* yang sama pada subset data yang berbeda secara paralel. **Random Forests** adalah implementasi *bagging* yang sangat sukses menggunakan Decision Trees dengan tambahan keacakan fitur.
* **Boosting** (seperti AdaBoost dan Gradient Boosting) melatih *predictor* secara sekuensial, di mana setiap model baru berfokus pada perbaikan kesalahan model sebelumnya.
* **Stacking** menggunakan pendekatan berlapis, di mana sebuah *meta-model* (blender) dilatih untuk mengagregasi prediksi dari model-model di lapisan bawahnya.