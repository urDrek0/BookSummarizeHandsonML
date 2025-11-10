# Bab 6: Decision Trees

Bab ini akan membahas:

  * Cara melatih, memvisualisasikan, dan membuat prediksi dengan Decision Trees (Pohon Keputusan).
  * Algoritma pelatihan CART (Classification and Regression Tree) yang digunakan oleh Scikit-Learn.
  * Cara mengatur *hyperparameter* regularisasi untuk menghindari *overfitting*.
  * Penggunaan Decision Trees untuk tugas Regresi.
  * Keterbatasan dan kelemahan dari Decision Trees.

-----

## 1 Melatih dan Memvisualisasikan Decision Tree

Decision Trees adalah model serbaguna yang dapat melakukan tugas klasifikasi, regresi, dan bahkan *multioutput*[cite: 1077]. Mereka adalah algoritma yang kuat dan menjadi komponen dasar dari **Random Forests** (dibahas di Bab 7)[cite: 1077].

Melatih model ini di Scikit-Learn sangat mudah:

```python
from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)
```

Anda dapat memvisualisasikan pohon yang telah dilatih menggunakan fungsi `export_graphviz()` untuk menghasilkan file `.dot`, yang kemudian dapat dikonversi menjadi gambar[cite: 1078].

## 2 Membuat Prediksi

Membuat prediksi dengan Decision Tree sangat intuitif. Anda mulai dari simpul akar (*root node*) di paling atas dan menjawab serangkaian pertanyaan (misalnya, "Apakah panjang kelopak \<= 2.45 cm?") [cite: 1079-1080]. Berdasarkan jawaban Anda, Anda bergerak ke simpul anak (kiri atau kanan) dan mengulangi prosesnya.

Ketika Anda mencapai **leaf node** (simpul daun, yang tidak memiliki anak), Anda mengambil prediksi kelas dari simpul tersebut [cite: 1079-1080].

### Atribut Simpul

  * **samples**: Menghitung berapa banyak instans data latih yang "berlaku" pada simpul tersebut[cite: 1080].
  * **value**: Menunjukkan berapa banyak instans latih dari setiap kelas yang berlaku pada simpul tersebut[cite: 1080].
  * **gini**: Mengukur **impurity** (ketidakmurnian) dari simpul. Simpul "murni" (`gini=0`) berarti semua instans di dalamnya milik satu kelas yang sama[cite: 1080].

Salah satu keunggulan besar Decision Trees adalah mereka hampir tidak memerlukan persiapan data; khususnya, mereka **tidak memerlukan *feature scaling* atau *centering***[cite: 1080].

### Model Interpretation: White Box vs Black Box

Decision Trees sangat mudah dipahami dan diinterpretasi. Anda dapat melihat dengan jelas aturan-aturan yang digunakan untuk membuat keputusan. Model seperti ini disebut **white box models**. Ini berlawanan dengan model seperti Jaringan Saraf Tiruan atau SVM, yang dianggap sebagai **black box models** karena sulit untuk menjelaskan mengapa mereka membuat prediksi tertentu[cite: 1082].

## 3 Estimasi Probabilitas Kelas

Decision Tree juga dapat mengestimasi probabilitas bahwa sebuah instans termasuk dalam kelas *k*. Caranya adalah dengan menelusuri pohon hingga ke *leaf node* untuk instans tersebut, lalu mengembalikan rasio instans latih kelas *k* yang ada di simpul tersebut[cite: 1082].

## 4 Algoritma Pelatihan CART

Scikit-Learn menggunakan algoritma **Classification and Regression Tree (CART)** untuk melatih Decision Trees.

  * **Cara Kerja**: Algoritma ini membagi *training set* menjadi dua subset menggunakan satu fitur ($k$) dan satu *threshold* ($t_k$) (misalnya, "petal length \<= 2.45 cm")[cite: 1083].
  * **Pemilihan Split**: Ia mencari pasangan ($k$, $t_k$) yang menghasilkan subset paling "murni" (dengan Gini impurity terendah), yang dihitung menggunakan *cost function* CART[cite: 1083].
  * **Rekursif**: Setelah membelah data, ia mengulangi proses yang sama pada subset yang dihasilkan, lalu pada sub-subset, dan seterusnya, secara rekursif[cite: 1083].
  * **Kapan Berhenti**: Pembelahan berhenti ketika mencapai `max_depth` (kedalaman maksimum) atau jika tidak dapat menemukan pembelahan yang akan mengurangi *impurity*[cite: 1083].
  * **Algoritma Greedy**: CART adalah **greedy algorithm** (algoritma serakah). Ia mencari *split* optimal pada level saat ini, tanpa memeriksa apakah *split* ini akan menghasilkan *impurity* terendah di beberapa level di bawahnya. Ini sering menghasilkan solusi yang cukup baik, tetapi tidak dijamin optimal[cite: 1085].

## 5 Regularisasi Hyperparameters

Decision Trees membuat sangat sedikit asumsi tentang data. Jika tidak dibatasi, ia akan terus beradaptasi dengan data latih hingga sangat pas, yang menyebabkan **overfitting**[cite: 1086].

Untuk menghindari *overfitting*, kita perlu melakukan **regularisasi** dengan membatasi "kebebasan" Decision Tree. Ini dikontrol oleh beberapa *hyperparameter* [cite: 1086-1087]:

  * **`max_depth`**: Kedalaman maksimum pohon. Menguranginya akan meregularisasi model.
  * **`min_samples_split`**: Jumlah minimum sampel yang harus dimiliki sebuah simpul sebelum dapat dibelah.
  * **`min_samples_leaf`**: Jumlah minimum sampel yang harus dimiliki oleh sebuah *leaf node*.
  * **`max_leaf_nodes`**: Jumlah maksimum *leaf node*.
  * **`max_features`**: Jumlah maksimum fitur yang dievaluasi untuk setiap pembelahan.

Menaikkan `min_*` atau mengurangi `max_*` akan meregularisasi model.

## 6 Regresi

Decision Trees juga mampu melakukan tugas regresi (`DecisionTreeRegressor`)[cite: 1089].

  * Cara kerjanya mirip, tetapi alih-alih memprediksi *kelas* di *leaf node*, ia memprediksi sebuah *nilai*[cite: 1089].
  * Nilai yang diprediksi adalah **nilai rata-rata target** dari semua instans latih yang ada di *leaf node* tersebut[cite: 1089].
  * Algoritma CART-nya bekerja dengan cara meminimalkan **MSE (Mean Squared Error)**, bukan Gini impurity, saat mencari *split* terbaik[cite: 1090].
  * Sama seperti klasifikasi, model regresi ini juga rentan terhadap *overfitting* jika tidak diregularisasi[cite: 1090].

## 7 Instabilitas (Keterbatasan)

Decision Trees memiliki beberapa keterbatasan:

1.  **Sensitif terhadap Rotasi**: Mereka menyukai *decision boundaries* yang ortogonal (tegak lurus sumbu). Jika data Anda diputar (dirotasi), model akan kesulitan dan menghasilkan *boundary* yang terlalu rumit[cite: 1091].
2.  **Sensitif terhadap Variasi Data**: Mereka sangat sensitif terhadap variasi kecil dalam data latih. Jika Anda menghapus satu instans saja, Anda bisa mendapatkan pohon yang sangat berbeda[cite: 1091].

-----

# KESIMPULAN AKHIR

  * Decision Trees adalah model yang kuat, serbaguna, dan mudah diinterpretasi ("white box").
  * Mereka dapat melakukan klasifikasi, regresi, dan tidak memerlukan *feature scaling*.
  * Scikit-Learn menggunakan algoritma **CART**, yang secara rekursif membelah data untuk menciptakan subset "termurni".
  * Model ini **nonparametric** dan sangat rentan terhadap **overfitting** jika tidak dibatasi.
  * Regularisasi dilakukan dengan mengatur *hyperparameter* seperti `max_depth` dan `min_samples_leaf`.
  * Kelemahan utamanya adalah sensitivitasnya terhadap rotasi data dan variasi kecil pada data latih.