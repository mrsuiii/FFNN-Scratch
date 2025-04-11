# Tugas Besar 1 IF3270 Pembelajaran Mesin - Feed Forward Neural Network (FFNN)

## Kelompok 48
- Kharris Khisunica (13522051)
- Emery Fathan Zwageri (13522079)
- Abdul Rafi Radityo Hutomo (13522089)

## Daftar Isi

- [Deskripsi Singkat Repository dan Program](#deskripsi-singkat-repository-dan-program)
- [Set up dan run](#set-up-dan-run)
- [Pembagian Tugas](#pembagian-tugas)


## Deskripsi Singkat Repository dan Program

### Struktur Repository
```
FFNN-Scratch   
│
├───doc
│       Tubes1_G65.pdf
│
├───src
│       activation.py
│       draw_FFNN.py
│       FFNN.py
│       init.py
│       Layer.py
│       loss.py
│       test.ipynb
│       Value.py
│       visualize.py
│
├───tests
│      tes.py
│      test_backprop.py
│      test_cce_loss.py
│      test_softmax.py
│      test_softmax_training.py
│      test_training.py
│
│    readme.md
```


Repository ini berisi kode implementasi _Feed Forward Neural Network_ (FFNN) yang dibangun dari _scratch_ dan diimplementasikan menggunakan **_automic differentiation_**. 

Kode ini menerima beberapa fungsi aktivasi, fungsi loss, inisialisasi bobot yang diimplementasi secara manual. Kode juga menerima masukan apakah pengguna mau melakukan regularisasi dan normalisasi dengan RMSNorm (Bonus). 

Model FFNN yang dibangun juga dapat melakukan _forward_ dan _backward_ _propagation_ dengan nilai parameter yang ditentukan oleh pengguna dan menghasilkan histori proses pelatihan.  

Fungsi aktivasi yang diimplementasikan adalah
1. _Linear_
2. _ReLU_
3. _Sigmoid_
4. _Hyperbolic Tangent_ (tanh)
5. _Softmax_
6. _Leaky ReLU_ (Bonus)
7. _Swish_ (Bonus)

Fungsi loss yang diimplementasikan adalah
1. _Minimum Square Error_ (MSE)
2. _Binary Cross-Entropy_ (BSE)
3. _Categorical Cross-Entropy_ (CCE)

Fungsi inisialisasi bobot yang diimplementasikan adalah
1. _Zero Initialization_
2. _Uniform Initialization_
3. _Normal Initialization_
4. _He Initialization_ (Bonus)
5. _Xavier Initialization_ (Bonus)

Fungsi regularisasi yang diimplementasikan adalah
1. Regularisasi L1
2. Regularisasi L2




## Set up dan Run
1. Pastikan Python sudah terinstal pada komputer yang menjalani program. Instalasi Python dapat dilakukan [di sini](https://www.python.org/downloads/).

2. Clone repo ini dengan cara berikut.
```
git clone https://github.com/mrsuiii/FFNN-Scratch
cd FFNN-Scratch
cd src
```
3. Gunakan code editor/IDE yang dapat support Jupyter notebook (.ipynb) seperti `VSCode`.

4. Navigasi workspace ke file `test.ipynb` dan jalankan menggunakan fitur `Run All` yang tersedia.



## Pembagian Tugas

| NIM      | Kontribusi                                                                                                                                                                                                                            |
------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 13522051 | Laporan, Readme, Value, Autograd, Test                                                          |
| 13522079 | Value, Autograd, Layer, loss, init, test, RMSNorm, activation, laporan                                                          |
| 13522089 |FFNN, Value, Autograd, init, test, visualization, regularization, activation, laporan                                                          |