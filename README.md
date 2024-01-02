# Laporan Proyek Machine Learning

### Nama : Matius Yudika Sitorus
### Nim  : 211351079
### Kelas : Malam B

## Domain Proyek
Web app ini berfungsi untuk menentukan tujuan spesifik proyek dan data yang akan digunakan, apakah itu data klinis pasien, data gizi, atau data genetika. Selain itu, perlu memperhatikan aspek etika dan keamanan dalam pengelolaan data kesehatan.

## Business Understanding
Web app ini memungkinkan untuk meningkatkan diagnosis dini diabetes, memberikan rekomendasi perawatan yang lebih baik, atau mengoptimalkan manajemen penyakit. Dapat memastikan bahwa implementasi KNN diabetes tidak hanya teknis, tetapi juga memberikan nilai strategis.

## Problem Statements
Diabetes merinci masalah atau tantangan yang hendak diatasi menggunakan model tersebut.

## Solution Statements
Dapat meningkatkan diagnosis dini tetapi juga memberikan dasar untuk perawatan yang dipersonalisasi, mengoptimalkan sumber daya kesehatan, dan meningkatkan kualitas hidup pasien.

## Goals
Mengembangkan web app yang memberikan klarifikasi diabetes yang akurat berdasarkan informasi yang di inputkan.

## Data Understanding
Dataset ini berisi informasi tentang langkah kritis dalam membangun model KNN yang efektif dan dapat diandalkan untuk tugas klasifikasi diabetes. Pemahaman ini membantu mengidentifikasi aspek-aspek yang perlu diperhatikan selama pra-pemrosesan data dan pelatihan model.
[Pima Indians Diabetes Dataset] (https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

### Variabel- variabel pada Pima Indians Diabetes Dataset adalah sebagai berikut:
- Pregnancies : Mempresentasikan berapa kali wanita tersebut hamil selama hidupnya.
- Glucose: Mempresentasikan konsentrasi glukosa plasma pada 2 jam dalam tes toleransi glukosa.
- Blood Pressure: Mempresentasikan tekanan darah diastolik dalam (mm / Hg) ketika jantung rileks setelah kontraksi.
- Skin Thickness: Nilai yang digunakan untuk memperkirakan lemak tubuh (mm) yang diukur pada lengan kanan setengah antara proses olecranon dari siku dan proses akromial skapula.
- Insulin: Tingkat insulin 2 jam insulin serum dalam satuan mu U/ml.
- BMI: Berat dalam kg / (tinggi dalam meter kuadrat), dan merupakan indikator kesehatan seseorang.
- Diabetes Pedigree Function: Indikator riwayat diabetes dalam keluarga.
- Age: Mempresentasikan dari umur.

## Data Preparation
Sebelumnya saya sudah mendownload datasetnya dari kaggle
Import Dataset

```py
path_dataset=("/content/diabetes.csv")
# disini saya menggunakan copypath dalam google colab
```

## Import Library
Import library yang akan digunakan

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.plotting import plot_decision_regions
from sklearn import datasets, neighbors
from sklearn import metrics

import pickle
sns.set()
import warnings
warnings.filterwarnings('ignore')
```

## Data discovery
Membuat variabel dataframe.

```py
df=pd.read_csv(path_dataset)
```

Lihat 5 data dalam data frame.

```py
df.head()
```

Melihat informasi dalam dataframe tersebut.

```py
df.info()
```

Melihat rangkuman data numerik

```py
df.describe()
```

Melihat apakah ada data null

```py
df.isnull().sum().sort_values(ascending=False)
```

Menghitung jumlah kemunculan setiap nilai di kolom 'Insulin'

```py
df['Insulin'].value_counts()
```

## EDA

Kita lihat distribusi dari histogram ini dengan melihat presentasenya

```py
p = df.hist(figsize = (10,10))
```

![](image/histogram.png)

Selanjutnya melihat heatmap korelasi kolom dengan label tetapi kita harus mengubah object menjadi numerik terlebih dahulu

```py
plt.figure(figsize=(15,12))
p=sns.heatmap(df.corr(), annot=True,cmap ='RdYlGn')
```

![](image/corr.png)

Selanjutnya kita distribusi berdasarkan glukosa

```py
color_wheel = {1: "#0392cf",
               2: "#7bc043"}
colors = df["Glucose"].map(lambda x: color_wheel.get(x + 1))
print(df.Glucose.value_counts())
p=df.Glucose.value_counts().plot(kind="bar")
```

![](image/glucose.png)

## Preprocessing

Disini kita melihat data data preprocessing

```py
df.dropna(inplace=True)
```

Disini kita melihat data yang di drop

```py
df.drop_duplicates(inplace=True)
```

Disini kita mengdrop data 'Outcome'

```py
# X = df.drop("Outcome", axis = 1)
y = df.Outcome
y.head(5)
```

## Modeling

Membuat variabel X yaitu feature dari semua kolom yang ada kecuali Outcome dan membuat y yaitu label yang berisi data Kolom Outcome

```
X = df.drop('Outcome',axis=1).values
Y = df[["Outcome"]]
```

Selanjutnya split data dan mencari nilai test_scores dan train_score terbaik

```py
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3, random_state=20)
```

```py
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

```py
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
```

Selanjutnya kita  membuat model dengan menggunakan k= 3

```py
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

print("Akurasi model KNN =", knn_model.score(X_train,y_train))
```

dan hasil scorenya 85.84%


## Visualisasi hasil algoritma

Pertama kita buat prediksi dari model yang telah kita buat pada sebelumnya

```py
prediction = knn_model.predict(X_test)

input_data = np.array([[6,148,72,35,0,33.6,0.627,50]]) # 1

prediction = knn_model.predict(input_data)
if(prediction[0] == 1) :
  print("Diabetes")
else:
  print("Tidak diabetes")
```

Outputnya adalah Diabetes yang dimana sesuai dengan data aktualnya


Selanjutnya melihat decision region knn dari glukosa dan kehamilannnya

```py
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

ks = [3, 10]

for i, ax in enumerate(axs.flatten()):
    knn_comparison(df, ks[i], ax)

plt.tight_layout()
plt.show()
```

![](image/decision_reg.png)

## Save Model

```py
filename = 'diabetes.pkl'
with open(filename, 'wb') as file:
    pickle.dump(knn, file)
```


```py
filename = 'diabetes.sav'
pickle.dump(knn_model, open(filename, 'wb'))
```

## Evaluation

Pada tahap ini saya menggunakan confusion matrix dan classification sebagai matrix evaluasinya

```py
y_pred = knn.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
```

```py
print(classification_report(y_test,y_pred))
```

![](image/report.png)

## Deployment

[Klasifikasi Diabetes](https://knndiabet.streamlit.app/)<br>
