# Bab 5: Support Vector Machines (SVM)

Bab ini akan membahas:
* Ide fundamental di balik Support Vector Machines (SVM): *large margin classification*.
* Perbedaan antara *hard margin* dan *soft margin classification*.
* Cara SVM menangani data non-linear menggunakan *polynomial features* dan *kernel trick*.
* Implementasi SVM untuk tugas klasifikasi dan regresi menggunakan Scikit-Learn.
* Konsep matematis di balik cara kerja SVM (termasuk *dual problem* dan *kernels*).

---

## 1. Linear SVM Classification

Ide fundamental di balik SVM adalah **large margin classification**[cite: 1046]. Alih-alih hanya menempatkan garis yang memisahkan dua kelas, SVM mencoba menemukan "jalan" (margin) terluas yang bisa memisahkan kedua kelas tersebut.

* **Support Vectors**: Instans latih yang terletak di tepi "jalan" (margin) disebut sebagai *support vectors*. Mereka adalah satu-satunya data yang "menopang" atau menentukan decision boundary. Instans yang berada "di luar jalan" sama sekali tidak memengaruhi model[cite: 1047, 1079].
* **Sensitivitas Skala**: SVM sangat sensitif terhadap skala fitur. Penting untuk melakukan *feature scaling* (misalnya, menggunakan `StandardScaler`) sebelum melatih SVM agar model tidak mengabaikan fitur-fitur berskala kecil[cite: 1048, 1018].

### Hard Margin vs. Soft Margin

* **Hard Margin Classification**:
    * Pendekatan ini secara ketat memaksakan bahwa semua instans harus berada di luar "jalan" dan di sisi yang benar.
    * Ini hanya berfungsi jika data dapat dipisahkan secara linear (linearly separable).
    * Sangat sensitif terhadap *outliers* (pencilan)[cite: 1049].
* **Soft Margin Classification**:
    * Ini adalah pendekatan yang lebih fleksibel. Tujuannya adalah menemukan keseimbangan antara menjaga "jalan" selebar mungkin dan membatasi *margin violations* (pelanggaran margin)—yaitu, instans yang berakhir di tengah jalan atau bahkan di sisi yang salah[cite: 1049].
    * Di Scikit-Learn, Anda dapat mengontrol keseimbangan ini dengan *hyperparameter* `C`:
        * `C` yang **rendah** menghasilkan margin yang lebih lebar tetapi lebih banyak *violations*.
        * `C` yang **tinggi** menghasilkan lebih sedikit *violations* tetapi margin yang lebih sempit (berisiko *overfitting*)[cite: 1050].

Anda dapat mengimplementasikan SVM linear di Scikit-Learn menggunakan `LinearSVC(C=1, loss="hinge")`, `SVC(kernel="linear")`, atau `SGDClassifier(loss="hinge")`[cite: 1051].

---

## 2. Nonlinear SVM Classification

Banyak dataset tidak dapat dipisahkan secara linear. SVM dapat menangani data non-linear menggunakan dua teknik utama:

### 1. Polynomial Features
Seperti yang dibahas di Bab 4, Anda dapat menambahkan fitur-fitur polinomial (misalnya, $x_1^2$, $x_2^2$, $x_1 x_2$) untuk membuat data yang tadinya tidak linear menjadi dapat dipisahkan secara linear. Setelah itu, Anda bisa melatih model `LinearSVC` pada data yang telah ditransformasi ini[cite: 1052, 1053].

### 2. The Kernel Trick (Trik Kernel)
Menambahkan *polynomial features* bisa sangat lambat secara komputasi jika derajat polinomnya tinggi (menyebabkan "ledakan kombinatorial" fitur).

Untungnya, SVM memiliki trik matematis ajaib yang disebut **kernel trick**. Trik ini memungkinkan Anda mendapatkan hasil yang sama *seolah-olah* Anda telah menambahkan banyak fitur polinomial (bahkan berderajat sangat tinggi) tanpa harus benar-benar menambahkannya. Ini membuat pelatihan menjadi jauh lebih efisien[cite: 1054].

Implementasi di Scikit-Learn:
* **Polynomial Kernel**: Gunakan `SVC(kernel="poly", degree=3, coef0=1)`. `coef0` mengontrol seberapa besar pengaruh polinomial berderajat tinggi vs. rendah[cite: 1054].
* **Gaussian RBF Kernel**: Metode kernel populer lainnya yang bekerja dengan menambahkan fitur berdasarkan "fungsi kesamaan" (similarity function). Ini dapat menangani data non-linear yang sangat kompleks[cite: 1056].
    * `SVC(kernel="rbf", gamma=5, C=0.001)`
    * **gamma ($\gamma$)**: Berperan sebagai *hyperparameter* regularisasi. Meningkatkan `gamma` membuat *decision boundary* lebih tidak beraturan (berisiko *overfitting*); menurunkannya membuat *boundary* lebih mulus (berisiko *underfitting*)[cite: 1057, 1058].

> **Aturan Praktis**: Selalu coba `LinearSVC` terlebih dahulu (jauh lebih cepat). Jika dataset tidak terlalu besar, coba `SVC(kernel="rbf")`, karena ini seringkali bekerja dengan baik[cite: 1058].

---

## 3. SVM Regression

SVM juga dapat digunakan untuk tugas regresi. Alih-alih mencoba memfit "jalan" terluas *di antara* dua kelas, SVM Regression mencoba memfit sebanyak mungkin instans *di dalam* "jalan".

* Tujuannya adalah membatasi *margin violations* (instans yang berada *di luar* jalan)[cite: 1059].
* Lebar "jalan" dikontrol oleh *hyperparameter* **epsilon ($\epsilon$)**.
* Menambahkan data di dalam "jalan" (di dalam margin $\epsilon$) tidak memengaruhi prediksi model. Model ini disebut **$\epsilon$-insensitive**[cite: 1060].
* Untuk data non-linear, Anda dapat menggunakan kernel, sama seperti pada klasifikasi.
* Implementasi di Scikit-Learn: `LinearSVR` (untuk regresi linear) dan `SVR` (untuk regresi non-linear yang di-kernel)[cite: 1061, 1062].

---

## 4. Di Balik Layar (Under the Hood)

* **Fungsi Keputusan**: SVM Linear memprediksi kelas berdasarkan hasil fungsi $w^T \cdot x + b$. Jika hasilnya positif, ia memprediksi kelas positif (1), jika negatif, ia memprediksi kelas negatif (0)[cite: 1063].
* **Tujuan Pelatihan**: Tujuan dari pelatihan adalah menemukan bobot $w$ dan bias $b$ yang membuat margin (lebar "jalan") sebesar mungkin. Mendapatkan margin yang besar sama dengan meminimalkan $||w||$[cite: 1066].
* **Quadratic Programming (QP)**: Masalah optimisasi (baik *hard margin* maupun *soft margin*) yang coba dipecahkan oleh SVM disebut *Quadratic Programming*[cite: 1068].
* **The Dual Problem**: Untuk masalah QP, ada ekspresi matematis alternatif yang disebut *dual problem*. SVM sering dilatih menggunakan *dual problem* karena ini **lebih cepat** ketika jumlah instans latih (m) lebih kecil daripada jumlah fitur (n), dan yang terpenting, ini **memungkinkan *kernel trick*** bekerja[cite: 1069, 1070].

---

# KESIMPULAN AKHIR

* SVM adalah model serbaguna yang sangat kuat untuk klasifikasi (linear atau non-linear), regresi, dan bahkan deteksi anomali.
* SVM bekerja dengan menemukan *margin* (batas) terbesar antar kelas.
* Model ini hanya bergantung pada **support vectors** (instans di tepi margin), membuatnya efisien untuk membuat prediksi.
* SVM sangat sensitif terhadap **feature scaling**.
* Untuk data non-linear, SVM menggunakan **kernel trick** (seperti `poly` atau `rbf`) untuk menangani data tanpa harus benar-benar menambahkan fitur baru.
* `LinearSVC` cepat dan bagus untuk dataset linear, sedangkan `SVC(kernel="rbf")` adalah pilihan default yang baik untuk dataset non-linear yang tidak terlalu besar.
* `SVR` dan `LinearSVR` adalah padanan SVM untuk tugas regresi.