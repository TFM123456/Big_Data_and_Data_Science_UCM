# --- Modelado

## Cargamos librerias


```python
!pip install tensorflow
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import requests
import seaborn as sns
import sklearn.multiclass
import tensorflow as tf
from numpy import array 
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
from tensorflow import keras
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from matplotlib import pyplot
from sklearn.metrics import multilabel_confusion_matrix
```

    Requirement already satisfied: tensorflow in c:\users\rtx9652\anaconda3\lib\site-packages (2.6.0)
    Requirement already satisfied: typing-extensions~=3.7.4 in c:\users\rtx9652\anaconda3\lib\site-packages (from tensorflow) (3.7.4.3)
    Requirement already satisfied: opt-einsum~=3.3.0 in c:\users\rtx9652\anaconda3\lib\site-packages (from tensorflow) (3.3.0)
    Requirement already satisfied: termcolor~=1.1.0 in c:\users\rtx9652\anaconda3\lib\site-packages (from tensorflow) (1.1.0)
    Requirement already satisfied: astunparse~=1.6.3 in c:\users\rtx9652\anaconda3\lib\site-packages (from tensorflow) (1.6.3)
    Requirement already satisfied: numpy~=1.19.2 in c:\users\rtx9652\anaconda3\lib\site-packages (from tensorflow) (1.19.5)
    Requirement already satisfied: tensorflow-estimator~=2.6 in c:\users\rtx9652\anaconda3\lib\site-packages (from tensorflow) (2.6.0)
    Requirement already satisfied: gast==0.4.0 in c:\users\rtx9652\anaconda3\lib\site-packages (from tensorflow) (0.4.0)
    Requirement already satisfied: h5py~=3.1.0 in c:\users\rtx9652\anaconda3\lib\site-packages (from tensorflow) (3.1.0)
    Requirement already satisfied: absl-py~=0.10 in c:\users\rtx9652\anaconda3\lib\site-packages (from tensorflow) (0.13.0)
    Requirement already satisfied: keras-preprocessing~=1.1.2 in c:\users\rtx9652\anaconda3\lib\site-packages (from tensorflow) (1.1.2)
    Requirement already satisfied: google-pasta~=0.2 in c:\users\rtx9652\anaconda3\lib\site-packages (from tensorflow) (0.2.0)
    Requirement already satisfied: grpcio<2.0,>=1.37.0 in c:\users\rtx9652\anaconda3\lib\site-packages (from tensorflow) (1.39.0)
    Requirement already satisfied: wheel~=0.35 in c:\users\rtx9652\anaconda3\lib\site-packages (from tensorflow) (0.36.2)
    Requirement already satisfied: keras~=2.6 in c:\users\rtx9652\anaconda3\lib\site-packages (from tensorflow) (2.6.0)
    Requirement already satisfied: protobuf>=3.9.2 in c:\users\rtx9652\anaconda3\lib\site-packages (from tensorflow) (3.17.3)
    Requirement already satisfied: six~=1.15.0 in c:\users\rtx9652\anaconda3\lib\site-packages (from tensorflow) (1.15.0)
    Requirement already satisfied: wrapt~=1.12.1 in c:\users\rtx9652\anaconda3\lib\site-packages (from tensorflow) (1.12.1)
    Requirement already satisfied: tensorboard~=2.6 in c:\users\rtx9652\anaconda3\lib\site-packages (from tensorflow) (2.6.0)
    Requirement already satisfied: clang~=5.0 in c:\users\rtx9652\anaconda3\lib\site-packages (from tensorflow) (5.0)
    Requirement already satisfied: flatbuffers~=1.12.0 in c:\users\rtx9652\anaconda3\lib\site-packages (from tensorflow) (1.12)
    Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in c:\users\rtx9652\anaconda3\lib\site-packages (from tensorboard~=2.6->tensorflow) (1.8.0)
    Requirement already satisfied: google-auth<2,>=1.6.3 in c:\users\rtx9652\anaconda3\lib\site-packages (from tensorboard~=2.6->tensorflow) (1.35.0)
    Requirement already satisfied: markdown>=2.6.8 in c:\users\rtx9652\anaconda3\lib\site-packages (from tensorboard~=2.6->tensorflow) (3.3.4)
    Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in c:\users\rtx9652\anaconda3\lib\site-packages (from tensorboard~=2.6->tensorflow) (0.4.5)
    Requirement already satisfied: setuptools>=41.0.0 in c:\users\rtx9652\anaconda3\lib\site-packages (from tensorboard~=2.6->tensorflow) (52.0.0.post20210125)
    Requirement already satisfied: werkzeug>=0.11.15 in c:\users\rtx9652\anaconda3\lib\site-packages (from tensorboard~=2.6->tensorflow) (1.0.1)
    Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in c:\users\rtx9652\anaconda3\lib\site-packages (from tensorboard~=2.6->tensorflow) (0.6.1)
    Requirement already satisfied: requests<3,>=2.21.0 in c:\users\rtx9652\anaconda3\lib\site-packages (from tensorboard~=2.6->tensorflow) (2.25.1)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in c:\users\rtx9652\anaconda3\lib\site-packages (from google-auth<2,>=1.6.3->tensorboard~=2.6->tensorflow) (0.2.8)
    Requirement already satisfied: rsa<5,>=3.1.4 in c:\users\rtx9652\anaconda3\lib\site-packages (from google-auth<2,>=1.6.3->tensorboard~=2.6->tensorflow) (4.7.2)
    Requirement already satisfied: cachetools<5.0,>=2.0.0 in c:\users\rtx9652\anaconda3\lib\site-packages (from google-auth<2,>=1.6.3->tensorboard~=2.6->tensorflow) (4.2.2)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in c:\users\rtx9652\anaconda3\lib\site-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard~=2.6->tensorflow) (1.3.0)
    Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in c:\users\rtx9652\anaconda3\lib\site-packages (from pyasn1-modules>=0.2.1->google-auth<2,>=1.6.3->tensorboard~=2.6->tensorflow) (0.4.8)
    Requirement already satisfied: chardet<5,>=3.0.2 in c:\users\rtx9652\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard~=2.6->tensorflow) (4.0.0)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\users\rtx9652\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard~=2.6->tensorflow) (1.26.4)
    Requirement already satisfied: idna<3,>=2.5 in c:\users\rtx9652\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard~=2.6->tensorflow) (2.10)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\rtx9652\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard~=2.6->tensorflow) (2020.12.5)
    Requirement already satisfied: oauthlib>=3.0.0 in c:\users\rtx9652\anaconda3\lib\site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard~=2.6->tensorflow) (3.1.1)
    

## Vamos a leer nuestra base de datos


```python
url="https://raw.githubusercontent.com/TFM123456/Big_Data_and_Data_Science_UCM/main/datos_galicia_limpio.csv"
s=requests.get(url).content
datos_galicia=pd.read_csv(io.StringIO(s.decode('ISO-8859-1')))
```

# EDA - Análisis exploratorio de datos


```python
datos_galicia.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>id</th>
      <th>superficie</th>
      <th>fecha</th>
      <th>lat</th>
      <th>lng</th>
      <th>idprovincia</th>
      <th>idmunicipio</th>
      <th>causa</th>
      <th>muertos</th>
      <th>...</th>
      <th>TMIN</th>
      <th>TMAX</th>
      <th>VELMEDIA</th>
      <th>RACHA</th>
      <th>SOL</th>
      <th>Trimestre</th>
      <th>Mes</th>
      <th>Año</th>
      <th>DIR_VIENTO</th>
      <th>PRES_RANGE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2001150021</td>
      <td>5.0</td>
      <td>2001-02-20</td>
      <td>43.703581</td>
      <td>-8.038777</td>
      <td>A Coruña</td>
      <td>CEDEIRA</td>
      <td>negligencia</td>
      <td>0</td>
      <td>...</td>
      <td>7.0</td>
      <td>15.6</td>
      <td>2-4 m/s</td>
      <td>16.9</td>
      <td>10.2</td>
      <td>Q1</td>
      <td>febrero</td>
      <td>2001</td>
      <td>E</td>
      <td>4.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>2001150094</td>
      <td>1.5</td>
      <td>2001-02-25</td>
      <td>43.186836</td>
      <td>-8.685470</td>
      <td>A Coruña</td>
      <td>CARBALLO</td>
      <td>intencionado</td>
      <td>0</td>
      <td>...</td>
      <td>6.5</td>
      <td>11.6</td>
      <td>4-6 m/s</td>
      <td>11.1</td>
      <td>10.2</td>
      <td>Q1</td>
      <td>febrero</td>
      <td>2001</td>
      <td>NE</td>
      <td>4.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>2001150145</td>
      <td>1.5</td>
      <td>2001-04-13</td>
      <td>43.699889</td>
      <td>-7.984566</td>
      <td>A Coruña</td>
      <td>CEDEIRA</td>
      <td>negligencia</td>
      <td>0</td>
      <td>...</td>
      <td>10.4</td>
      <td>17.4</td>
      <td>4-6 m/s</td>
      <td>13.9</td>
      <td>12.1</td>
      <td>Q2</td>
      <td>abril</td>
      <td>2001</td>
      <td>NE</td>
      <td>3.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>2001150151</td>
      <td>1.1</td>
      <td>2001-04-13</td>
      <td>42.758649</td>
      <td>-8.917814</td>
      <td>A Coruña</td>
      <td>LOUSAME</td>
      <td>causa desconocida</td>
      <td>0</td>
      <td>...</td>
      <td>10.4</td>
      <td>17.4</td>
      <td>4-6 m/s</td>
      <td>13.9</td>
      <td>12.1</td>
      <td>Q2</td>
      <td>abril</td>
      <td>2001</td>
      <td>NE</td>
      <td>3.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>2001150153</td>
      <td>1.5</td>
      <td>2001-04-13</td>
      <td>43.063218</td>
      <td>-9.235604</td>
      <td>A Coruña</td>
      <td>MUXÃÂA</td>
      <td>intencionado</td>
      <td>0</td>
      <td>...</td>
      <td>10.4</td>
      <td>17.4</td>
      <td>4-6 m/s</td>
      <td>13.9</td>
      <td>12.1</td>
      <td>Q2</td>
      <td>abril</td>
      <td>2001</td>
      <td>NE</td>
      <td>3.2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 29 columns</p>
</div>




```python
print('El dataset tiene un total de {} lineas y {} columnas'.format(datos_galicia.shape[0],datos_galicia.shape[1]))
```

    El dataset tiene un total de 12976 lineas y 29 columnas
    

Podemos ver también cómo se distribuyen nuestras variables:

<!-- Podemos también ver cómo se distribuyen nuestras variables: -->


```python
datos_galicia.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>id</th>
      <th>superficie</th>
      <th>lat</th>
      <th>lng</th>
      <th>muertos</th>
      <th>heridos</th>
      <th>time_ctrl</th>
      <th>time_ext</th>
      <th>personal</th>
      <th>medios</th>
      <th>TMEDIA</th>
      <th>PRECIPITACION</th>
      <th>TMIN</th>
      <th>TMAX</th>
      <th>RACHA</th>
      <th>SOL</th>
      <th>Año</th>
      <th>PRES_RANGE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>12976.000000</td>
      <td>1.297600e+04</td>
      <td>12976.000000</td>
      <td>12976.000000</td>
      <td>12976.000000</td>
      <td>12976.000000</td>
      <td>12976.000000</td>
      <td>12976.000000</td>
      <td>12976.000000</td>
      <td>12976.000000</td>
      <td>12976.000000</td>
      <td>12976.000000</td>
      <td>12976.000000</td>
      <td>12976.000000</td>
      <td>12976.000000</td>
      <td>12976.000000</td>
      <td>12976.000000</td>
      <td>12976.000000</td>
      <td>12976.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>9781.431181</td>
      <td>2.005867e+09</td>
      <td>2.586703</td>
      <td>42.555421</td>
      <td>-7.994614</td>
      <td>0.000077</td>
      <td>0.000462</td>
      <td>1.781149</td>
      <td>1.781149</td>
      <td>12.281520</td>
      <td>1.964396</td>
      <td>16.584664</td>
      <td>0.132167</td>
      <td>9.717810</td>
      <td>23.452366</td>
      <td>8.537230</td>
      <td>8.547102</td>
      <td>2005.592555</td>
      <td>3.836675</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5848.203194</td>
      <td>3.830998e+06</td>
      <td>1.863610</td>
      <td>0.451212</td>
      <td>0.562395</td>
      <td>0.008779</td>
      <td>0.021499</td>
      <td>0.894420</td>
      <td>0.894420</td>
      <td>7.252796</td>
      <td>1.674277</td>
      <td>5.954963</td>
      <td>0.338685</td>
      <td>6.252533</td>
      <td>7.001220</td>
      <td>3.298872</td>
      <td>3.549735</td>
      <td>3.828872</td>
      <td>1.639618</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>2.001150e+09</td>
      <td>1.000000</td>
      <td>41.833819</td>
      <td>-9.293500</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-1.300000</td>
      <td>0.000000</td>
      <td>-8.600000</td>
      <td>5.600000</td>
      <td>1.700000</td>
      <td>0.000000</td>
      <td>2001.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4839.750000</td>
      <td>2.003150e+09</td>
      <td>1.250000</td>
      <td>42.189372</td>
      <td>-8.422458</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.170000</td>
      <td>1.170000</td>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>12.400000</td>
      <td>0.000000</td>
      <td>5.200000</td>
      <td>18.600000</td>
      <td>6.100000</td>
      <td>6.500000</td>
      <td>2003.000000</td>
      <td>2.600000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9402.500000</td>
      <td>2.005151e+09</td>
      <td>2.000000</td>
      <td>42.470163</td>
      <td>-7.987553</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.500000</td>
      <td>1.500000</td>
      <td>11.000000</td>
      <td>2.000000</td>
      <td>17.400000</td>
      <td>0.000000</td>
      <td>10.600000</td>
      <td>23.400000</td>
      <td>8.100000</td>
      <td>9.400000</td>
      <td>2005.000000</td>
      <td>3.600000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>14958.500000</td>
      <td>2.009150e+09</td>
      <td>3.000000</td>
      <td>42.932203</td>
      <td>-7.556843</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.310000</td>
      <td>2.310000</td>
      <td>16.000000</td>
      <td>3.000000</td>
      <td>21.100000</td>
      <td>0.000000</td>
      <td>14.900000</td>
      <td>28.400000</td>
      <td>10.600000</td>
      <td>11.200000</td>
      <td>2009.000000</td>
      <td>4.900000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>20544.000000</td>
      <td>2.015361e+09</td>
      <td>10.200000</td>
      <td>43.730713</td>
      <td>-6.771548</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>4.190000</td>
      <td>4.190000</td>
      <td>33.000000</td>
      <td>8.000000</td>
      <td>31.200000</td>
      <td>1.000000</td>
      <td>23.400000</td>
      <td>42.000000</td>
      <td>28.300000</td>
      <td>14.800000</td>
      <td>2015.000000</td>
      <td>8.600000</td>
    </tr>
  </tbody>
</table>
</div>




```python
datos_galicia.hist(figsize = (12, 12));

```


    
![png](output_11_0.png)
    


Vamos a ver si los tipos de datos han migrado bien desde R


```python
print('______________________________________________________________\n'
      '\n Tipo de datos a nivel de variable:\n'
      '______________________________________________________________\n\n\n{}'.format(datos_galicia.dtypes))
```

    ______________________________________________________________
    
     Tipo de datos a nivel de variable:
    ______________________________________________________________
    
    
    Unnamed: 0         int64
    id                 int64
    superficie       float64
    fecha             object
    lat              float64
    lng              float64
    idprovincia       object
    idmunicipio       object
    causa             object
    muertos            int64
    heridos            int64
    time_ctrl        float64
    time_ext         float64
    personal           int64
    medios             int64
    gastos            object
    ALTITUD           object
    TMEDIA           float64
    PRECIPITACION      int64
    TMIN             float64
    TMAX             float64
    VELMEDIA          object
    RACHA            float64
    SOL              float64
    Trimestre         object
    Mes               object
    Año                int64
    DIR_VIENTO        object
    PRES_RANGE       float64
    dtype: object
    

En primer lugar, generamos una copia de nuestro dataset para poder hacer modificaciones sin alterar el dataset original


```python
datos_galicia1 = datos_galicia.copy()
```


```python
print('Finalmente, nuestro dataset tiene un total de {} lineas y {} columnas'.format(datos_galicia1.shape[0],datos_galicia1.shape[1]))
```

    Finalmente, nuestro dataset tiene un total de 12976 lineas y 29 columnas
    

### Valores faltantes - NaN


```python
print('______________________________________________________________\n'
      '\n Numero de observaciones con datos faltantes:\n'
      '______________________________________________________________\n\n\n{}'.format(datos_galicia1.isnull().sum()))
```

    ______________________________________________________________
    
     Numero de observaciones con datos faltantes:
    ______________________________________________________________
    
    
    Unnamed: 0       0
    id               0
    superficie       0
    fecha            0
    lat              0
    lng              0
    idprovincia      0
    idmunicipio      0
    causa            0
    muertos          0
    heridos          0
    time_ctrl        0
    time_ext         0
    personal         0
    medios           0
    gastos           0
    ALTITUD          0
    TMEDIA           0
    PRECIPITACION    0
    TMIN             0
    TMAX             0
    VELMEDIA         0
    RACHA            0
    SOL              0
    Trimestre        0
    Mes              0
    Año              0
    DIR_VIENTO       0
    PRES_RANGE       0
    dtype: int64
    

Ahora, vamos a separar las variables categóricas de las numéricas.
Comprobamos los valores únicos por variable:


```python
print('______________________________________________________________\n'
      '\n\t Número de observaciones únicos \n'
      '______________________________________________________________\n')
      
print('Dataset Galicia: ', datos_galicia1.shape)

df = pd.DataFrame(columns = ['Variable','Valores_unicos'])

for i in datos_galicia1.columns.values:
    df = df.append({'Variable':i, 'Valores_unicos':(len(datos_galicia1[i].unique()))}, ignore_index = True)

df
```

    ______________________________________________________________
    
    	 Número de observaciones únicos 
    ______________________________________________________________
    
    Dataset Galicia:  (12976, 29)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Variable</th>
      <th>Valores_unicos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Unnamed: 0</td>
      <td>12976</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id</td>
      <td>12976</td>
    </tr>
    <tr>
      <th>2</th>
      <td>superficie</td>
      <td>619</td>
    </tr>
    <tr>
      <th>3</th>
      <td>fecha</td>
      <td>2272</td>
    </tr>
    <tr>
      <th>4</th>
      <td>lat</td>
      <td>6118</td>
    </tr>
    <tr>
      <th>5</th>
      <td>lng</td>
      <td>6121</td>
    </tr>
    <tr>
      <th>6</th>
      <td>idprovincia</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>idmunicipio</td>
      <td>268</td>
    </tr>
    <tr>
      <th>8</th>
      <td>causa</td>
      <td>5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>muertos</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>heridos</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>time_ctrl</td>
      <td>251</td>
    </tr>
    <tr>
      <th>12</th>
      <td>time_ext</td>
      <td>251</td>
    </tr>
    <tr>
      <th>13</th>
      <td>personal</td>
      <td>34</td>
    </tr>
    <tr>
      <th>14</th>
      <td>medios</td>
      <td>9</td>
    </tr>
    <tr>
      <th>15</th>
      <td>gastos</td>
      <td>3</td>
    </tr>
    <tr>
      <th>16</th>
      <td>ALTITUD</td>
      <td>4</td>
    </tr>
    <tr>
      <th>17</th>
      <td>TMEDIA</td>
      <td>287</td>
    </tr>
    <tr>
      <th>18</th>
      <td>PRECIPITACION</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19</th>
      <td>TMIN</td>
      <td>276</td>
    </tr>
    <tr>
      <th>20</th>
      <td>TMAX</td>
      <td>333</td>
    </tr>
    <tr>
      <th>21</th>
      <td>VELMEDIA</td>
      <td>5</td>
    </tr>
    <tr>
      <th>22</th>
      <td>RACHA</td>
      <td>73</td>
    </tr>
    <tr>
      <th>23</th>
      <td>SOL</td>
      <td>149</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Trimestre</td>
      <td>4</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Mes</td>
      <td>12</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Año</td>
      <td>15</td>
    </tr>
    <tr>
      <th>27</th>
      <td>DIR_VIENTO</td>
      <td>8</td>
    </tr>
    <tr>
      <th>28</th>
      <td>PRES_RANGE</td>
      <td>149</td>
    </tr>
  </tbody>
</table>
</div>



Las variables Unnamed e id tienen tantos valores únicos como observaciones contienen, por lo que decidimos eliminar ambas variables:


```python
datos_galicia1 = datos_galicia1.drop(columns=["Unnamed: 0"])
datos_galicia1 = datos_galicia1.drop(columns=["id"])
```


```python
print('Después de haber eliminado las dos variables, nuestro dataset tiene un total de {} lineas y {} columnas'.format(datos_galicia1.shape[0],datos_galicia1.shape[1]))
```

    Después de haber eliminado las dos variables, nuestro dataset tiene un total de 12976 lineas y 27 columnas
    

Teniendo en cuenta los valores únicos que componen cada variable y el tipo de dato, dividimos entre las variables categóricas y numéricas


```python
lista_numericas=datos_galicia1._get_numeric_data()
lista_categoricas=datos_galicia1.select_dtypes(include = ["object"])
```

Comprobamos


```python
len(lista_categoricas.columns)
```




    10




```python
len(lista_numericas.columns)
```




    17



ha incluido correctamente todas las columnas. Vemos que incluye cada lista


```python
lista_categoricas.columns
```




    Index(['fecha', 'idprovincia', 'idmunicipio', 'causa', 'gastos', 'ALTITUD',
           'VELMEDIA', 'Trimestre', 'Mes', 'DIR_VIENTO'],
          dtype='object')




```python
lista_numericas.columns
```




    Index(['superficie', 'lat', 'lng', 'muertos', 'heridos', 'time_ctrl',
           'time_ext', 'personal', 'medios', 'TMEDIA', 'PRECIPITACION', 'TMIN',
           'TMAX', 'RACHA', 'SOL', 'Año', 'PRES_RANGE'],
          dtype='object')



Vamos a ver como se distribuyen los valores en las variables categoricas

## Análisis de variables categóricas


```python
for i in lista_categoricas:
    print(datos_galicia1[i].value_counts())
```

    2005-03-19    59
    2004-03-28    59
    2001-09-18    58
    2002-03-28    56
    2002-09-02    55
                  ..
    2009-06-28     1
    2006-09-13     1
    2015-07-04     1
    2008-07-18     1
    2001-03-18     1
    Name: fecha, Length: 2272, dtype: int64
    Ourense       5762
    A Coruña      3549
    Pontevedra    2062
    Lugo          1603
    Name: idprovincia, dtype: int64
    VIANA DO BOLO          436
    MANZANEDA              352
    CHANDREXA DE QUEIXA    295
    MUIÃâOS             256
    SANTA COMBA            234
                          ... 
    CARIÃâO               1
    RIBADUMIA                1
    MUGARDOS                 1
    BARREIROS                1
    BEADE                    1
    Name: idmunicipio, Length: 268, dtype: int64
    intencionado         11293
    causa desconocida      830
    negligencia            534
    fuego reproducido      215
    rayo                   104
    Name: causa, dtype: int64
    NO INFO    10779
    < 5K        1769
    >5K          428
    Name: gastos, dtype: int64
    Superior a 125    5762
    Inferior a 80     3549
    Entre 80-125      2062
    NO INFO           1603
    Name: ALTITUD, dtype: int64
    < 2 m/s    7366
    2-4 m/s    3926
    4-6 m/s    1125
    6-8 m/s     385
    > 8 m/s     174
    Name: VELMEDIA, dtype: int64
    Q3    5256
    Q1    3554
    Q2    2825
    Q4    1341
    Name: Trimestre, dtype: int64
    agosto        2296
    marzo         2070
    septiembre    1929
    abril         1358
    febrero       1309
    julio         1031
    junio          864
    octubre        655
    mayo           603
    diciembre      448
    noviembre      238
    enero          175
    Name: Mes, dtype: int64
    NE    2727
    N     2722
    W     2395
    NW    1843
    E     1353
    S      767
    SW     691
    SE     478
    Name: DIR_VIENTO, dtype: int64
    

La variable idmunicipio incluye demasiadas categorias, la eliminamos


```python
datos_galicia1 = datos_galicia1.drop(columns=['idmunicipio'])
```

También vamos a prescindir de fecha, ya que la tenemos representada en varias variables : Trimestre / MES / Año


```python
datos_galicia1 = datos_galicia1.drop(columns=['fecha'])
```


```python
print('Nos quedamos con {} lineas y {} columnas'.format(datos_galicia1.shape[0],datos_galicia1.shape[1]))
```

    Nos quedamos con 12976 lineas y 25 columnas
    

### Transformamos las variables categóricas -> codificación One-Hot y orden numérico

Las variables categóricas necesitan ser pasadas a categóricas para poder tratarlas. Hay varias formas de hacerlos pero las más comunes suelen ser transformarlas a variables ordinales o realizar la codificación One-Hot

Poner UN orden numérico en ocasiones dificulta la predicción ya que da diferentes pesos a las distintas categorías de una variable, por ello, esta transformación la vamos a utilizar para aquellas variables que sí sigan un orden en sus categoría

Por otro lado, para aquellas que no sigan un orden, utilizaremos la codificación One-Hot.
Este método consiste en crear una nueva variable binaria por cada categoria existente en la variable inicial, donde 
1 serán las observaciones que pertenezcan a esa categoría y 0 las demás.

En muchas tareas, tales como la regresión lineal, es común usar k-1 variables binarias en lugar de k, donde k es el número total de categorías. Esto se debe a que estamos añadiendo una variable extra redundante que no es más que una combinación lineal de las otras y seguramente afectará de manera negativa al rendimiento del modelo. Además, al eliminar una variable no estamos perdiendo información, ya que se entiende que, si el resto de las categorías contienen un 0, la categoría correspondiente es la de la variable eliminada.

* Se ha seleccionado esta opción tras probar con todas las variables categóricas en codificación One-Hot y todas en orden numérico

Vemos las variables categóricas de las que disponemos


```python
print('______________________________________________________________\n'
      '\n\t Variables categóricas \n'
      '______________________________________________________________\n')
for x in lista_categoricas: 
    print(x)
```

    ______________________________________________________________
    
    	 Variables categóricas 
    ______________________________________________________________
    
    fecha
    idprovincia
    idmunicipio
    causa
    gastos
    ALTITUD
    VELMEDIA
    Trimestre
    Mes
    DIR_VIENTO
    


```python
'''categoricas con codificacion one-hot:  idprovincia, Trimestre, Mes, DIR_VIENTO'''
```




    'categoricas con codificacion one-hot:  idprovincia, Trimestre, Mes, DIR_VIENTO'




```python
dummies= pd.get_dummies(datos_galicia1['idprovincia'], drop_first = True)
datos_galicia1 = pd.concat([datos_galicia1, dummies], axis = 1)

dummies2= pd.get_dummies(datos_galicia1['Trimestre'], drop_first = True)
datos_galicia1 = pd.concat([datos_galicia1, dummies2], axis = 1)

dummies3= pd.get_dummies(datos_galicia1['Mes'], drop_first = True)
datos_galicia1 = pd.concat([datos_galicia1, dummies3], axis = 1)

dummies4= pd.get_dummies(datos_galicia1['DIR_VIENTO'], drop_first = True)
datos_galicia1 = pd.concat([datos_galicia1, dummies4], axis = 1)
```


```python
len(dummies.columns)+len(dummies2.columns)+len(dummies3.columns)+len(dummies4.columns)
```




    24




```python
25+24
```




    49




```python
datos_galicia1.shape
```




    (12976, 49)



Eliminamos las variables que hemos creado con one-hot


```python
datos_galicia1 = datos_galicia1.drop(columns=['idprovincia'])
datos_galicia1 = datos_galicia1.drop(columns=['Trimestre'])
datos_galicia1 = datos_galicia1.drop(columns=['Mes'])
datos_galicia1 = datos_galicia1.drop(columns=['DIR_VIENTO'])
```


```python
49-4
```




    45




```python
datos_galicia1.shape
```




    (12976, 45)




```python
'''variables orden numérico: gastos , ALTITUD, VELMEDIA'''
```




    'variables orden numérico: gastos , ALTITUD, VELMEDIA'




```python
datos_galicia1['gastos'].value_counts()
```




    NO INFO    10779
    < 5K        1769
    >5K          428
    Name: gastos, dtype: int64




```python
datos_galicia1.gastos.replace(("NO INFO","< 5K ",">5K"),
                      (1,2,3),inplace=True)
```


```python
datos_galicia1['ALTITUD'].value_counts()
```




    Superior a 125    5762
    Inferior a 80     3549
    Entre 80-125      2062
    NO INFO           1603
    Name: ALTITUD, dtype: int64




```python
datos_galicia1.ALTITUD.replace(("NO INFO","Inferior a 80","Entre 80-125","Superior a 125"),
                      (1,2,3,4),inplace=True)
```


```python
datos_galicia1['VELMEDIA'].value_counts()
```




    < 2 m/s    7366
    2-4 m/s    3926
    4-6 m/s    1125
    6-8 m/s     385
    > 8 m/s     174
    Name: VELMEDIA, dtype: int64




```python
datos_galicia1.VELMEDIA.replace(("< 2 m/s","2-4 m/s","4-6 m/s","6-8 m/s","> 8 m/s"),
                      (1,2,3,4,5),inplace=True)
```

Comprobamos los tipos de las variables a ver si solo tenemos numericas


```python
datos_galicia1.dtypes
```




    superficie       float64
    lat              float64
    lng              float64
    causa             object
    muertos            int64
    heridos            int64
    time_ctrl        float64
    time_ext         float64
    personal           int64
    medios             int64
    gastos             int64
    ALTITUD            int64
    TMEDIA           float64
    PRECIPITACION      int64
    TMIN             float64
    TMAX             float64
    VELMEDIA           int64
    RACHA            float64
    SOL              float64
    Año                int64
    PRES_RANGE       float64
    Lugo               uint8
    Ourense            uint8
    Pontevedra         uint8
    Q2                 uint8
    Q3                 uint8
    Q4                 uint8
    agosto             uint8
    diciembre          uint8
    enero              uint8
    febrero            uint8
    julio              uint8
    junio              uint8
    marzo              uint8
    mayo               uint8
    noviembre          uint8
    octubre            uint8
    septiembre         uint8
    N                  uint8
    NE                 uint8
    NW                 uint8
    S                  uint8
    SE                 uint8
    SW                 uint8
    W                  uint8
    dtype: object



Vemos cómo ha quedado el dataset final


```python
datos_galicia1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>superficie</th>
      <th>lat</th>
      <th>lng</th>
      <th>causa</th>
      <th>muertos</th>
      <th>heridos</th>
      <th>time_ctrl</th>
      <th>time_ext</th>
      <th>personal</th>
      <th>medios</th>
      <th>...</th>
      <th>noviembre</th>
      <th>octubre</th>
      <th>septiembre</th>
      <th>N</th>
      <th>NE</th>
      <th>NW</th>
      <th>S</th>
      <th>SE</th>
      <th>SW</th>
      <th>W</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.0</td>
      <td>43.703581</td>
      <td>-8.038777</td>
      <td>negligencia</td>
      <td>0</td>
      <td>0</td>
      <td>3.55</td>
      <td>3.55</td>
      <td>14</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.5</td>
      <td>43.186836</td>
      <td>-8.685470</td>
      <td>intencionado</td>
      <td>0</td>
      <td>0</td>
      <td>2.05</td>
      <td>2.05</td>
      <td>5</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.5</td>
      <td>43.699889</td>
      <td>-7.984566</td>
      <td>negligencia</td>
      <td>0</td>
      <td>0</td>
      <td>1.50</td>
      <td>1.50</td>
      <td>9</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.1</td>
      <td>42.758649</td>
      <td>-8.917814</td>
      <td>causa desconocida</td>
      <td>0</td>
      <td>0</td>
      <td>3.10</td>
      <td>3.10</td>
      <td>18</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.5</td>
      <td>43.063218</td>
      <td>-9.235604</td>
      <td>intencionado</td>
      <td>0</td>
      <td>0</td>
      <td>1.35</td>
      <td>1.35</td>
      <td>14</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 45 columns</p>
</div>




```python
print('Finalmente, el dataset tiene un total de {} lineas y {} columnas'.format(datos_galicia.shape[0],datos_galicia.shape[1]))
```

    Finalmente, el dataset tiene un total de 12976 lineas y 29 columnas
    

# Modelado

Definimos nuestra variable objetivo -> Causa


```python
datos_galicia1['target']=datos_galicia1['causa']
```


```python
datos_galicia1 = datos_galicia1.drop(columns=['causa'])
```

Vamos a ver como se distribuye nuestra variable objetivo


```python
print(datos_galicia1.groupby('target').size())
```

    target
    causa desconocida      830
    fuego reproducido      215
    intencionado         11293
    negligencia            534
    rayo                   104
    dtype: int64
    

Podemos verlo también de una forma más visual


```python
count_classes = datos_galicia1.value_counts(datos_galicia1['target'])
count_classes.plot(kind = 'bar', figsize=(10,4),rot=0, color = ['blue'])
plt.title("Frecuencia de las observaciones")
plt.xlabel("Clases",labelpad=14)
plt.ylabel("Número de observaciones",labelpad=14)

```




    Text(0, 0.5, 'Número de observaciones')




    
![png](output_76_1.png)
    


Como podemos comprobar, nuestra variable está desbalanceada y por lo tanto, tendremos que tenerlo en cuenta

Dividimos los datos en Train y Test y separamos ambas entre x -> entradas ( variables explicativas) e y-> salidas ( variable objetivo)
Nuestro conjunto de Train es el entrenamiento, en Test probaremos los resultados de nuestras predicciones.



```python
X_train, X_test, y_train, y_test = train_test_split(
                                        datos_galicia1.drop('target', axis = 'columns'),
                                        datos_galicia1['target'],
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True,
                                        stratify = datos_galicia1['target'])
```

Creamos train y test teniendo en cuenta el desbalanceo mediante el parámetro stratify.

Comprobamos:


```python
print('______________________________________________________________\n'
      '\n Número de observaciones en el dataset de TRAIN y TEST \n'
      '______________________________________________________________\n')
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

for k in y_train.unique():
    print('\nTRAIN - % de incendios con causa =',k,': ', 
          round(((y_train[y_train==k].count() / y_train.shape[0])*100),2))
    print('TEST  - % de incendios con causa =',k,': ', 
          round(((y_test[y_test==k].count() / y_test.shape[0])*100),2))
```

    ______________________________________________________________
    
     Número de observaciones en el dataset de TRAIN y TEST 
    ______________________________________________________________
    
    Train set: (10380, 44) (10380,)
    Test set: (2596, 44) (2596,)
    
    TRAIN - % de incendios con causa = 1 :  87.03
    TEST  - % de incendios con causa = 1 :  87.02
    
    TRAIN - % de incendios con causa = 4 :  1.66
    TEST  - % de incendios con causa = 4 :  1.66
    
    TRAIN - % de incendios con causa = 3 :  4.11
    TEST  - % de incendios con causa = 3 :  4.12
    
    TRAIN - % de incendios con causa = 2 :  6.4
    TEST  - % de incendios con causa = 2 :  6.39
    
    TRAIN - % de incendios con causa = 5 :  0.8
    TEST  - % de incendios con causa = 5 :  0.81
    

Vemos que cada categoría se ha dividido de manera proporcional en la parte train y test

## Vamos a lanzar varios modelos con todas las variables


```python
print("\n En un inicio contamos con el siguiente número de variables :")
print("\n  ", len(datos_galicia1.columns))
```

    
     En un inicio contamos con el siguiente número de variables :
    
       45
    

 #### -- sin normalizar

### * MODELO 1 -  Support Vector Machines = SVM

conjunto de métodos de aprendizaje supervisados. realiza un método one-against-one.

NearMiss es una técnica de submuestreo. En lugar de volver a muestrear la clase minoritaria, utilizando una distancia, esto hará que la clase mayoritaria sea igual a la clase minoritaria.


```python
modelo_clf = svm.SVC()
modelo_clf.fit(X_train, y_train)


modelo_clf.predict(X_test)

modelo1 = ('MODELO 1 :Score modelo SVM : {} con todas las variables: {}'.format(modelo_clf.score(X_test,y_test),len(datos_galicia1.columns)))
modelo1
```




    'MODELO 1 :Score modelo SVM : 0.8701848998459168 con todas las variables: 45'



--------------------------------------

### * MODELO 2 -  Árbol de decisión

Modelo que se basa en una combinación y subdivision en ramas de las variables de una forma binaria para la toma de la decisión con mayor probabilidad de que ocurra un suceso¶


```python
modelo_arbol = DecisionTreeClassifier().fit(X_train, y_train)
y_pred        = modelo_arbol.predict(X_test)

modelo2 = ('MODELO 2 : Score modelo Árbol de decisión : {} con todas las variables: {}'.format(modelo_arbol.score(X_test,y_test),len(datos_galicia1.columns)))
modelo2
```




    'MODELO 2 : Score modelo Árbol de decisión : 0.7654083204930663 con todas las variables: 45'



---------------------------------------------------

### * MODELO 3 -  Random Forest

Conjunto de árboles de decisión


```python
modelo_ranfor = RandomForestClassifier().fit(X_train, y_train)
y_pred     = modelo_ranfor.predict(X_test)

modelo3 = ('MODELO 3 : Score modelo Random Forest : {} con todas las variables: {}'.format(modelo_ranfor.score(X_test,y_test),len(datos_galicia1.columns)))
modelo3
```




    'MODELO 3 : Score modelo Random Forest : 0.8694144838212635 con todas las variables: 45'



-------------------------------------

### * MODELO 4 -  GradientBoosting

está formado por un conjunto de árboles de decisión individuales, entrenados de forma secuencial, de forma que cada nuevo árbol trata de mejorar los errores de los árboles anteriores.
La diferencia con Random Forest es que utiliza árboles más débiles, con menos profundidad.


```python
modelo_gbrt = GradientBoostingClassifier(random_state=0, n_estimators=500,
                                  max_depth=1, learning_rate=0.01)
modelo_gbrt.fit(X_train, y_train)


modelo4 = ('MODELO 4 : Score modelo Gradient Boosting : {} con todas las variables : {} '.format(modelo_gbrt.score(X_test,y_test),len(datos_galicia1.columns)))
modelo4
```




    'MODELO 4 : Score modelo Gradient Boosting : 0.8697996918335902 con todas las variables : 45 '



De momento, contando con todas las variables, los modelos ajustan muy bien, estamos por encima de un 85% de precisión en la predicción. Los que mejores resultados arrojan son SVM,Random Forest y Gradient Boosting . vamos a continuar con ellos.

 #### -- normalizando

Vamos a comprobar si normalizando varios de los mejores , mejoran los resultados


```python
X_train = np.asarray(X_train).astype(np.float32)
```


```python
X_train, X_test, y_train, y_test = train_test_split(
                                        datos_galicia1.drop('target', axis = 'columns'),
                                        datos_galicia1['target'],
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True,
                                        stratify = datos_galicia1['target'])

norm= tf.keras.layers.experimental.preprocessing.Normalization(axis = -1,dtype=None,mean = None,variance=None)

norm.adapt(X_train)
x_train_norm = norm(X_train)
```


```python
#SVM
X_train, X_test, y_train, y_test = train_test_split(
                                        datos_galicia1.drop('target', axis = 'columns'),
                                        datos_galicia1['target'],
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True,
                                        stratify = datos_galicia1['target'])

modelo_clf_norm = svm.SVC(decision_function_shape='ovo')
modelo_clf_norm.fit(x_train_norm, y_train)


modelo_clf_norm.predict(X_test)

modelo_clf_norm.score(X_test,y_test)
```




    0.8701848998459168




```python
#Random Forest
modelo_ranfor_norm = RandomForestClassifier(bootstrap = True, criterion= 'entropy', max_depth=None, n_estimators=150,class_weight='balanced').fit(x_train_norm, y_train)
y_pred     = modelo_ranfor_norm.predict(X_test)

modelo_ranfor_norm.score(X_test,y_test)
```




    0.8701848998459168



No se ven diferencias, por lo que es mejor optar por no normalizarlo

## Selección de variables

Como hemos indicado anteriormente, disponemos de demasiadas variables, por lo que vamos a intentar simplicar nuestros modelos y con ello, mejorar su predicción.

Vamos a utilizar tres métodos de selección de variables: 

- Teniendo en cuenta la correlación entre variables predictoras y la correlación de esas variables con la variable objetivo
- Mejores variables por el método de chi-cuadrado
- Mejores variables por el algoritmo de F

En primer lugar, vamos a utilizar GridSearch para ver si estamos utilizando los mejores parámetros posibles en nuestros modelos teniendo en cuenta nuestra base de datos


```python
#SVM
param_grid = { 'class_weight'   : [None, 'balanced'],
              'decision_function_shape': ['ovo', 'ovr']
             }


# Búsqueda por grid search con validación cruzada

model_grid = GridSearchCV(
        estimator  = svm.SVC(),
        param_grid = param_grid,
        scoring    = 'roc_auc_ovr',  #'roc_auc_ovr',
        n_jobs     = -1,
        cv         = RepeatedKFold(n_splits=5, n_repeats=3, random_state=19), 
    )

model_grid.fit(X = X_train, y = y_train)

print('Los mejores hiperparámetros para el modelo SVM son: \n', model_grid.best_params_)
```




    "#SVM\nparam_grid = { 'class_weight'   : [None, 'balanced'],\n              'decision_function_shape': ['ovo', 'ovr']\n             }\n\n\n# Búsqueda por grid search con validación cruzada\n\nmodel_grid = GridSearchCV(\n        estimator  = svm.SVC(),\n        param_grid = param_grid,\n        scoring    = 'roc_auc_ovr',  #'roc_auc_ovr',\n        n_jobs     = -1,\n        cv         = RepeatedKFold(n_splits=5, n_repeats=3, random_state=19), \n    )\n\nmodel_grid.fit(X = X_train, y = y_train)\n\nprint('Los mejores hiperparámetros para el modelo SVM son: \n', model_grid.best_params_)"




```python
#RANDOM FOREST
param_grid = {'n_estimators': [30, 50, 100, 150],
              #'max_features': [5, 7, 9],
              'max_depth'   : [None, 5, 7, 10],
              'bootstrap': [True, False],
              'criterion': ['gini', 'entropy']
             }


# Búsqueda por grid search con validación cruzada

model_grid = GridSearchCV(
        estimator  = RandomForestClassifier(class_weight='balanced'),
        param_grid = param_grid,
        scoring    = 'roc_auc_ovr',  #'roc_auc_ovr',
        n_jobs     = -1,
        cv         = RepeatedKFold(n_splits=5, n_repeats=3, random_state=19), 
    )

model_grid.fit(X = X_train, y = y_train)

print('Los mejores hiperparámetros para el modelo Random Forest son: \n', model_grid.best_params_)
```




    "#RANDOM FOREST\nparam_grid = {'n_estimators': [30, 50, 100, 150],\n              #'max_features': [5, 7, 9],\n              'max_depth'   : [None, 5, 7, 10],\n              'bootstrap': [True, False],\n              'criterion': ['gini', 'entropy']\n             }\n\n\n# Búsqueda por grid search con validación cruzada\n\nmodel_grid = GridSearchCV(\n        estimator  = RandomForestClassifier(class_weight='balanced'),\n        param_grid = param_grid,\n        scoring    = 'roc_auc_ovr',  #'roc_auc_ovr',\n        n_jobs     = -1,\n        cv         = RepeatedKFold(n_splits=5, n_repeats=3, random_state=19), \n    )\n\nmodel_grid.fit(X = X_train, y = y_train)\n\nprint('Los mejores hiperparámetros para el modelo Random Forest son: \n', model_grid.best_params_)"



Tenemos muchas variables, vamos a intentar reducirlas para ver si mejoran los modelos

### -> Reducir variables por correlaciones


```python
def tidy_corr_matrix(corr_mat):
    '''
    Función para convertir una matrix de correlación de pandas en formato tidy
    '''
    corr_mat = corr_mat.stack().reset_index()
    corr_mat.columns = ['variable_1','variable_2','r']
    corr_mat = corr_mat.loc[corr_mat['variable_1'] != corr_mat['variable_2'], :]
    corr_mat['abs_r'] = np.abs(corr_mat['r'])
    corr_mat = corr_mat.sort_values('abs_r', ascending=False)
    
    return(corr_mat)



corr_matrix = datos_galicia1.select_dtypes(include=['float64', 'int']).corr(method='pearson')
tidy_corr_matrix(corr_matrix).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable_1</th>
      <th>variable_2</th>
      <th>r</th>
      <th>abs_r</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>47</th>
      <td>time_ext</td>
      <td>time_ctrl</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>37</th>
      <td>time_ctrl</td>
      <td>time_ext</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>82</th>
      <td>TMAX</td>
      <td>TMEDIA</td>
      <td>0.910006</td>
      <td>0.910006</td>
    </tr>
    <tr>
      <th>62</th>
      <td>TMEDIA</td>
      <td>TMAX</td>
      <td>0.910006</td>
      <td>0.910006</td>
    </tr>
    <tr>
      <th>61</th>
      <td>TMEDIA</td>
      <td>TMIN</td>
      <td>0.885689</td>
      <td>0.885689</td>
    </tr>
    <tr>
      <th>71</th>
      <td>TMIN</td>
      <td>TMEDIA</td>
      <td>0.885689</td>
      <td>0.885689</td>
    </tr>
    <tr>
      <th>83</th>
      <td>TMAX</td>
      <td>TMIN</td>
      <td>0.613526</td>
      <td>0.613526</td>
    </tr>
    <tr>
      <th>73</th>
      <td>TMIN</td>
      <td>TMAX</td>
      <td>0.613526</td>
      <td>0.613526</td>
    </tr>
    <tr>
      <th>33</th>
      <td>time_ctrl</td>
      <td>superficie</td>
      <td>0.423137</td>
      <td>0.423137</td>
    </tr>
    <tr>
      <th>3</th>
      <td>superficie</td>
      <td>time_ctrl</td>
      <td>0.423137</td>
      <td>0.423137</td>
    </tr>
  </tbody>
</table>
</div>



Podemos ver estas correlaciones de una manera más visual


```python
corr_matrix = datos_galicia1[['time_ext','time_ctrl','TMAX','TMEDIA','TMIN','superficie']].corr()
```


```python
correlation_mat = corr_matrix.corr()

sns.heatmap(correlation_mat, annot = True)

plt.show()
```


    
![png](output_121_0.png)
    



```python
'''time_ext/time_cntrl
TMAX/TMEDIA
TMIN/TMEDIA 
TMAX/TMIN
time_ctrl/superficie
 -- me voy a quedar solo con TMEDIA, superficie y time_cntr
'''
datos_galicia_reduc =  datos_galicia1.drop(['TMAX','TMIN', 'time_ext'], axis = 'columns')
```

Vamos a eliminar tambien las variables que tengan menos correlación con la variable objetivo


```python
#vemos los tipos de datos de la target
datos_galicia1['target'].value_counts()
```




    intencionado         11293
    causa desconocida      830
    negligencia            534
    fuego reproducido      215
    rayo                   104
    Name: target, dtype: int64



Todavía están como tipo object por lo que la pasamos a numérica


```python
datos_galicia1.target.replace(("intencionado","causa desconocida","negligencia","fuego reproducido","rayo"),
                      (1,2,3,4,5),inplace=True)

datos_galicia_reduc.target.replace(("intencionado","causa desconocida","negligencia","fuego reproducido","rayo"),
                      (1,2,3,4,5),inplace=True)
```


```python
datos_galicia_reduc['target'].value_counts()
```




    1    11293
    2      830
    3      534
    4      215
    5      104
    Name: target, dtype: int64



Calculamos la correlac


```python
corr = abs(datos_galicia_reduc.corr())
corr[['target']].sort_values(by = 'target',ascending = False).style.background_gradient()
```




<style  type="text/css" >
#T_6c828_row0_col0{
            background-color:  #023858;
            color:  #f1f1f1;
        }#T_6c828_row1_col0{
            background-color:  #ede7f2;
            color:  #000000;
        }#T_6c828_row2_col0{
            background-color:  #ede8f3;
            color:  #000000;
        }#T_6c828_row3_col0{
            background-color:  #f0eaf4;
            color:  #000000;
        }#T_6c828_row4_col0{
            background-color:  #f2ecf5;
            color:  #000000;
        }#T_6c828_row5_col0{
            background-color:  #f4edf6;
            color:  #000000;
        }#T_6c828_row6_col0{
            background-color:  #f4eef6;
            color:  #000000;
        }#T_6c828_row7_col0{
            background-color:  #f5eef6;
            color:  #000000;
        }#T_6c828_row8_col0,#T_6c828_row9_col0{
            background-color:  #f5eff6;
            color:  #000000;
        }#T_6c828_row10_col0,#T_6c828_row11_col0{
            background-color:  #f6eff7;
            color:  #000000;
        }#T_6c828_row12_col0,#T_6c828_row13_col0{
            background-color:  #f7f0f7;
            color:  #000000;
        }#T_6c828_row14_col0,#T_6c828_row15_col0{
            background-color:  #f8f1f8;
            color:  #000000;
        }#T_6c828_row16_col0{
            background-color:  #f9f2f8;
            color:  #000000;
        }#T_6c828_row17_col0,#T_6c828_row18_col0{
            background-color:  #faf2f8;
            color:  #000000;
        }#T_6c828_row19_col0,#T_6c828_row20_col0,#T_6c828_row21_col0,#T_6c828_row22_col0{
            background-color:  #faf3f9;
            color:  #000000;
        }#T_6c828_row23_col0,#T_6c828_row24_col0,#T_6c828_row25_col0{
            background-color:  #fbf3f9;
            color:  #000000;
        }#T_6c828_row26_col0{
            background-color:  #fbf4f9;
            color:  #000000;
        }#T_6c828_row27_col0,#T_6c828_row28_col0,#T_6c828_row29_col0,#T_6c828_row30_col0{
            background-color:  #fcf4fa;
            color:  #000000;
        }#T_6c828_row31_col0,#T_6c828_row32_col0{
            background-color:  #fdf5fa;
            color:  #000000;
        }#T_6c828_row33_col0{
            background-color:  #fef6fa;
            color:  #000000;
        }#T_6c828_row34_col0,#T_6c828_row35_col0{
            background-color:  #fef6fb;
            color:  #000000;
        }#T_6c828_row36_col0,#T_6c828_row37_col0,#T_6c828_row38_col0,#T_6c828_row39_col0,#T_6c828_row40_col0,#T_6c828_row41_col0{
            background-color:  #fff7fb;
            color:  #000000;
        }</style><table id="T_6c828_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >target</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_6c828_level0_row0" class="row_heading level0 row0" >target</th>
                        <td id="T_6c828_row0_col0" class="data row0 col0" >1.000000</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row1" class="row_heading level0 row1" >medios</th>
                        <td id="T_6c828_row1_col0" class="data row1 col0" >0.123753</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row2" class="row_heading level0 row2" >TMEDIA</th>
                        <td id="T_6c828_row2_col0" class="data row2 col0" >0.121941</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row3" class="row_heading level0 row3" >personal</th>
                        <td id="T_6c828_row3_col0" class="data row3 col0" >0.099976</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row4" class="row_heading level0 row4" >Año</th>
                        <td id="T_6c828_row4_col0" class="data row4 col0" >0.086746</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row5" class="row_heading level0 row5" >Lugo</th>
                        <td id="T_6c828_row5_col0" class="data row5 col0" >0.079416</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row6" class="row_heading level0 row6" >ALTITUD</th>
                        <td id="T_6c828_row6_col0" class="data row6 col0" >0.075265</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row7" class="row_heading level0 row7" >lat</th>
                        <td id="T_6c828_row7_col0" class="data row7 col0" >0.072122</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row8" class="row_heading level0 row8" >Ourense</th>
                        <td id="T_6c828_row8_col0" class="data row8 col0" >0.067900</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row9" class="row_heading level0 row9" >Q2</th>
                        <td id="T_6c828_row9_col0" class="data row9 col0" >0.067850</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row10" class="row_heading level0 row10" >mayo</th>
                        <td id="T_6c828_row10_col0" class="data row10 col0" >0.062214</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row11" class="row_heading level0 row11" >junio</th>
                        <td id="T_6c828_row11_col0" class="data row11 col0" >0.061929</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row12" class="row_heading level0 row12" >julio</th>
                        <td id="T_6c828_row12_col0" class="data row12 col0" >0.056630</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row13" class="row_heading level0 row13" >SOL</th>
                        <td id="T_6c828_row13_col0" class="data row13 col0" >0.056471</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row14" class="row_heading level0 row14" >marzo</th>
                        <td id="T_6c828_row14_col0" class="data row14 col0" >0.048430</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row15" class="row_heading level0 row15" >diciembre</th>
                        <td id="T_6c828_row15_col0" class="data row15 col0" >0.045835</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row16" class="row_heading level0 row16" >febrero</th>
                        <td id="T_6c828_row16_col0" class="data row16 col0" >0.042348</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row17" class="row_heading level0 row17" >Q4</th>
                        <td id="T_6c828_row17_col0" class="data row17 col0" >0.040505</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row18" class="row_heading level0 row18" >PRES_RANGE</th>
                        <td id="T_6c828_row18_col0" class="data row18 col0" >0.038575</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row19" class="row_heading level0 row19" >noviembre</th>
                        <td id="T_6c828_row19_col0" class="data row19 col0" >0.036887</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row20" class="row_heading level0 row20" >agosto</th>
                        <td id="T_6c828_row20_col0" class="data row20 col0" >0.035497</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row21" class="row_heading level0 row21" >Q3</th>
                        <td id="T_6c828_row21_col0" class="data row21 col0" >0.035287</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row22" class="row_heading level0 row22" >gastos</th>
                        <td id="T_6c828_row22_col0" class="data row22 col0" >0.034162</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row23" class="row_heading level0 row23" >septiembre</th>
                        <td id="T_6c828_row23_col0" class="data row23 col0" >0.032436</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row24" class="row_heading level0 row24" >PRECIPITACION</th>
                        <td id="T_6c828_row24_col0" class="data row24 col0" >0.031382</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row25" class="row_heading level0 row25" >Pontevedra</th>
                        <td id="T_6c828_row25_col0" class="data row25 col0" >0.029970</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row26" class="row_heading level0 row26" >RACHA</th>
                        <td id="T_6c828_row26_col0" class="data row26 col0" >0.025799</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row27" class="row_heading level0 row27" >SW</th>
                        <td id="T_6c828_row27_col0" class="data row27 col0" >0.025689</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row28" class="row_heading level0 row28" >superficie</th>
                        <td id="T_6c828_row28_col0" class="data row28 col0" >0.024701</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row29" class="row_heading level0 row29" >muertos</th>
                        <td id="T_6c828_row29_col0" class="data row29 col0" >0.023089</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row30" class="row_heading level0 row30" >time_ctrl</th>
                        <td id="T_6c828_row30_col0" class="data row30 col0" >0.022969</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row31" class="row_heading level0 row31" >enero</th>
                        <td id="T_6c828_row31_col0" class="data row31 col0" >0.021724</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row32" class="row_heading level0 row32" >lng</th>
                        <td id="T_6c828_row32_col0" class="data row32 col0" >0.016770</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row33" class="row_heading level0 row33" >NW</th>
                        <td id="T_6c828_row33_col0" class="data row33 col0" >0.011557</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row34" class="row_heading level0 row34" >W</th>
                        <td id="T_6c828_row34_col0" class="data row34 col0" >0.010068</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row35" class="row_heading level0 row35" >N</th>
                        <td id="T_6c828_row35_col0" class="data row35 col0" >0.008629</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row36" class="row_heading level0 row36" >SE</th>
                        <td id="T_6c828_row36_col0" class="data row36 col0" >0.006073</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row37" class="row_heading level0 row37" >octubre</th>
                        <td id="T_6c828_row37_col0" class="data row37 col0" >0.004512</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row38" class="row_heading level0 row38" >VELMEDIA</th>
                        <td id="T_6c828_row38_col0" class="data row38 col0" >0.004423</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row39" class="row_heading level0 row39" >NE</th>
                        <td id="T_6c828_row39_col0" class="data row39 col0" >0.003609</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row40" class="row_heading level0 row40" >heridos</th>
                        <td id="T_6c828_row40_col0" class="data row40 col0" >0.003361</td>
            </tr>
            <tr>
                        <th id="T_6c828_level0_row41" class="row_heading level0 row41" >S</th>
                        <td id="T_6c828_row41_col0" class="data row41 col0" >0.002379</td>
            </tr>
    </tbody></table>




```python
#Seleccionamos las que tengan más del 5% de correlación
datos_galicia_reduc = datos_galicia_reduc [['medios', 'TMEDIA','personal','Año','Lugo',
                                           'ALTITUD','lat','Ourense','Q2','mayo','junio','julio','SOL','target']]
```


```python
len(datos_galicia_reduc.columns)
```




    14




```python
X_train, X_test, y_train, y_test = train_test_split(
                                        datos_galicia_reduc.drop('target', axis = 'columns'),
                                        datos_galicia_reduc['target'],
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True,
                                        stratify = datos_galicia_reduc['target'])
```

### * Modelo 5 - SVM  reducidas variables por correlación


```python
modelo_clf_reduc = svm.SVC(class_weight= None, decision_function_shape='ovo')
modelo_clf_reduc.fit(X_train, y_train)

modelo_clf_reduc.predict(X_test)


modelo5 = ('MODELO 5 : Score modelo SVM reducido teniendo en cuenta correlaciones : {} con {} variables'.format(modelo_clf_reduc.score(X_test,y_test),len(datos_galicia_reduc.columns)))
modelo5
```




    'MODELO 5 : Score modelo SVM reducido teniendo en cuenta correlaciones : 0.8701848998459168 con 14 variables'



----------

### * Modelo 6 -  RF reducido reducidas variables por correlación

Conjuntos de arboles de decisión


```python
modelo_ranfor_reduc = RandomForestClassifier(bootstrap= True, criterion= 'entropy', max_depth= None, n_estimators= 150, class_weight='balanced').fit(X_train, y_train)
y_pred     = modelo_ranfor_reduc.predict(X_test)


modelo6 = ('MODELO 6 :Score modelo Random Forest reducido teniendo en cuenta correlaciones : {} con {} variables'.format(modelo_ranfor_reduc.score(X_test,y_test),len(datos_galicia_reduc.columns)))
modelo6
```




    'MODELO 6 :Score modelo Random Forest reducido teniendo en cuenta correlaciones : 0.8701848998459168 con 14 variables'



---------

### * Modelo 7 => Gradient Boosting reducido reducidas variables por correlación


```python
modelo_gbrt_reduc = GradientBoostingClassifier(random_state=0, n_estimators=500,
                                  max_depth=1, learning_rate=0.01)
modelo_gbrt_reduc.fit(X_train, y_train)

modelo7 = ('MODELO 7 : Score modelo Gradient Boosting reducido teniendo en cuenta correlaciones : {} con {} variables'.format(modelo_gbrt_reduc.score(X_test,y_test),len(datos_galicia_reduc.columns)))
modelo7
```




    'MODELO 7 : Score modelo Gradient Boosting reducido teniendo en cuenta correlaciones : 0.8697996918335902 con 14 variables'




```python

```

### - Selección de las mejores variables a través de la prueba F

Recordemos las variables que tenemos:


```python
datos_galicia1.columns
```




    Index(['superficie', 'lat', 'lng', 'muertos', 'heridos', 'time_ctrl',
           'time_ext', 'personal', 'medios', 'gastos', 'ALTITUD', 'TMEDIA',
           'PRECIPITACION', 'TMIN', 'TMAX', 'VELMEDIA', 'RACHA', 'SOL', 'Año',
           'PRES_RANGE', 'Lugo', 'Ourense', 'Pontevedra', 'Q2', 'Q3', 'Q4',
           'agosto', 'diciembre', 'enero', 'febrero', 'julio', 'junio', 'marzo',
           'mayo', 'noviembre', 'octubre', 'septiembre', 'N', 'NE', 'NW', 'S',
           'SE', 'SW', 'W', 'target'],
          dtype='object')




```python
k = 10  # número de atributos a seleccionar
columnas = list(datos_galicia1.columns.values)
seleccionadas = SelectKBest(f_classif, k=k).fit(X_train, y_train)
atrib = seleccionadas.get_support()
atributos = [columnas[i] for i in list(atrib.nonzero()[0])]
atributos
```




    ['superficie',
     'lat',
     'lng',
     'muertos',
     'heridos',
     'time_ctrl',
     'time_ext',
     'personal',
     'medios',
     'PRECIPITACION']



Seleccionamos esas columnas del dataset


```python
datos_theBest = datos_galicia1.copy()
```


```python
datos_theBest = datos_theBest[['superficie',
 'lat',
 'lng',
 'muertos',
 'heridos',
 'time_ctrl',
 'time_ext',
 'personal',
 'medios',
 'PRECIPITACION',
 'target']]
```


```python
len(datos_theBest.columns)
```




    11




```python
X_train, X_test, y_train, y_test = train_test_split(
                                        datos_theBest.drop('target', axis = 'columns'),
                                        datos_theBest['target'],
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True,
                                        stratify = datos_theBest['target'])

```


```python

```

### *  Modelo 8 - SMV seleccion de las mejores variables por F


```python
clf_theBest = svm.SVC(class_weight= None, decision_function_shape='ovo')
clf_theBest.fit(X_train, y_train)


modelo8 = ('MODELO 8 : Score modelo SMV estadistico F : {} con {} variables'.format(clf_theBest.score(X_test,y_test),len(datos_theBest.columns)))
modelo8
```




    'MODELO 8 : Score modelo SMV estadistico F : 0.8701848998459168 con 11 variables'



-----

### * Modelo 9 - Random Forest seleccion de las mejores variables por F


```python
modelo_ranfor_theBest = RandomForestClassifier(bootstrap = True, criterion= 'entropy', max_depth=None, n_estimators=150,class_weight='balanced').fit(X_train, y_train)
y_pred     = modelo_ranfor_theBest.predict(X_test)

modelo9 = ('MODELO 9: Score modelo SMV estadistico F : {} con {} variables'.format(modelo_ranfor_theBest.score(X_test,y_test),len(datos_theBest.columns)))
modelo9
```




    'MODELO 9: Score modelo SMV estadistico F : 0.8705701078582434 con 11 variables'



------

### * Modelo 10 - Gradient Boosting seleccion de las mejores variables por F


```python
modelo_gbrt_theBest = GradientBoostingClassifier(random_state=0, n_estimators=500,
                                  max_depth=1, learning_rate=0.01)
modelo_gbrt_theBest.fit(X_train, y_train)


modelo10 = ('MODELO 10 : Score modelo SMV estadistico F : {} con {} variables'.format(modelo_gbrt_theBest.score(X_test,y_test),len(datos_theBest.columns)))
modelo10
```




    'MODELO 10 : Score modelo SMV estadistico F : 0.8701848998459168 con 11 variables'




```python

```

### -> Selección de las mejores variables a través la importancia de las variables predictoras 


```python
#Con todas las variables
X_train, X_test, y_train, y_test = train_test_split(
                                        datos_galicia1.drop('target', axis = 'columns'),
                                        datos_galicia1['target'],
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True,
                                        stratify = datos_theBest['target'])

```


```python
modelo_ranfor.feature_importances_
```




    array([0.05976211, 0.08644232, 0.08108318, 0.00035509, 0.00025465,
           0.06508303, 0.06669248, 0.06385987, 0.04019237, 0.00768777,
           0.01077811, 0.06062158, 0.00757393, 0.0567882 , 0.0608992 ,
           0.01549273, 0.05088886, 0.05703113, 0.04186735, 0.05302345,
           0.0044928 , 0.00442551, 0.00425036, 0.00683879, 0.0049764 ,
           0.00281112, 0.00506298, 0.00075697, 0.00113726, 0.00317905,
           0.00527654, 0.00447586, 0.00487579, 0.00578402, 0.00066268,
           0.00303944, 0.00464499, 0.00869813, 0.00919536, 0.00732868,
           0.00460768, 0.00414083, 0.00469752, 0.00826384])




```python
modelo_gbrt.feature_importances_
```




    array([0.02500385, 0.1556113 , 0.05660735, 0.00535549, 0.        ,
           0.00935404, 0.01010786, 0.14579285, 0.01989083, 0.        ,
           0.01959691, 0.03786522, 0.00385744, 0.01280531, 0.11218875,
           0.        , 0.00868079, 0.00777312, 0.25547996, 0.        ,
           0.02042585, 0.00341354, 0.0052917 , 0.03783206, 0.00642794,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.00240043, 0.00709785, 0.        , 0.03080115, 0.        ,
           0.00033841, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        ])




```python
datos_galicia1.columns
```




    Index(['superficie', 'lat', 'lng', 'muertos', 'heridos', 'time_ctrl',
           'time_ext', 'personal', 'medios', 'gastos', 'ALTITUD', 'TMEDIA',
           'PRECIPITACION', 'TMIN', 'TMAX', 'VELMEDIA', 'RACHA', 'SOL', 'Año',
           'PRES_RANGE', 'Lugo', 'Ourense', 'Pontevedra', 'Q2', 'Q3', 'Q4',
           'agosto', 'diciembre', 'enero', 'febrero', 'julio', 'junio', 'marzo',
           'mayo', 'noviembre', 'octubre', 'septiembre', 'N', 'NE', 'NW', 'S',
           'SE', 'SW', 'W', 'target'],
          dtype='object')




```python
#seleccionamos los que esten por encima de 0.05
datos_gbrt = datos_galicia1[['lat','lng','personal','medios','TMAX','Año', 'target']]
```


```python
#seleccionamos los que esten por encima de 0.05
datos_rf = datos_galicia1[['superficie','lat','lng','time_ctrl','personal','medios','TMEDIA','RACHA','SOL','Año','PRES_RANGE','target']]
```

### * Modelo 11 - GradientBoosting con las variables de mayor importancia


```python
X_train, X_test, y_train, y_test = train_test_split(
                                        datos_gbrt.drop('target', axis = 'columns'),
                                        datos_gbrt['target'],
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True,
                                        stratify = datos_gbrt['target'])


modelo_gbrt_mejores = GradientBoostingClassifier(random_state=0, n_estimators=500,
                                  max_depth=1, learning_rate=0.01)
modelo_gbrt_mejores.fit(X_train, y_train)


modelo11 = ('MODELO 11 : Score modelo Gradient Boosting respecto a las variables con mayor importancia : {} con {} variables'.format(modelo_gbrt_mejores.score(X_test,y_test),len(datos_gbrt.columns)))
modelo11
```




    'MODELO 11 : Score modelo Gradient Boosting respecto a las variables con mayor importancia : 0.8701848998459168 con 7 variables'



-------------

### * Modelo 12 - Random Forest con las variables de mayor importancia


```python
X_train, X_test, y_train, y_test = train_test_split(
                                        datos_rf.drop('target', axis = 'columns'),
                                        datos_rf['target'],
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True,
                                        stratify = datos_rf['target'])

modelo_ranfor_mejores = RandomForestClassifier(bootstrap = True, criterion= 'entropy', max_depth=None, n_estimators=150,class_weight='balanced').fit(X_train, y_train)
y_pred     = modelo_ranfor_mejores.predict(X_test)


modelo12 = ('MODELO 12 : Score modelo Random Forest respecto a las variables con mayor importancia : {} con {} variables'.format(modelo_ranfor_mejores.score(X_test,y_test),len(datos_rf.columns)))
modelo12
```




    'MODELO 12 : Score modelo Random Forest respecto a las variables con mayor importancia : 0.8713405238828967 con 12 variables'



* Este método de selección de variables no puede hacerse con SVM

## Selección del modelo

Vamos a seleccionar el modelo mediante la validación cruzada de los modelos que mejor resultado nos hayan dado, tanto por su cálculo numérico como su posterior representación gráfica


```python
lista_modelos = [modelo1,modelo2,modelo3,modelo4,modelo5,modelo6,modelo7,modelo8,modelo9,modelo10,modelo11,modelo12]

for i in lista_modelos:
    print(i)
```

    MODELO 1 :Score modelo SVM : 0.8701848998459168 con todas las variables: 45
    MODELO 2 : Score modelo Árbol de decisión : 0.7654083204930663 con todas las variables: 45
    MODELO 3 : Score modelo Random Forest : 0.8694144838212635 con todas las variables: 45
    MODELO 4 : Score modelo Gradient Boosting : 0.8697996918335902 con todas las variables : 45 
    MODELO 5 : Score modelo SVM reducido teniendo en cuenta correlaciones : 0.8701848998459168 con 14 variables
    MODELO 6 :Score modelo Random Forest reducido teniendo en cuenta correlaciones : 0.8701848998459168 con 14 variables
    MODELO 7 : Score modelo Gradient Boosting reducido teniendo en cuenta correlaciones : 0.8697996918335902 con 14 variables
    MODELO 8 : Score modelo SMV estadistico F : 0.8701848998459168 con 11 variables
    MODELO 9: Score modelo SMV estadistico F : 0.8705701078582434 con 11 variables
    MODELO 10 : Score modelo SMV estadistico F : 0.8701848998459168 con 11 variables
    MODELO 11 : Score modelo Gradient Boosting respecto a las variables con mayor importancia : 0.8701848998459168 con 7 variables
    MODELO 12 : Score modelo Random Forest respecto a las variables con mayor importancia : 0.8713405238828967 con 12 variables
    

### * Validación cruzada

En SVM no se ha visto modificado la precisión del modelo modificando los parámetros, por lo que haremos la validación cruzada del modelo que mas "limpio" consideramos


```python
#Modelo 8
results_modelo_clf_theBest = cross_val_score(estimator=clf_theBest, X=X_train, y=y_train, cv=5)
```


```python
print(' Score: {}' .format(results_modelo_clf_theBest.mean()))
print(' Desviación: {}'.format(results_modelo_clf_theBest.std()))
```

     Score: 0.8703275529865125
     Desviación: 0.00019267822736028782
    

Por otro lado, vamos a hacerla para aquellos mejores modelos de RF


```python
#Modelo 9
results_modelo_ranfor_theBest = cross_val_score(estimator=modelo_ranfor_theBest, X=X_train, y=y_train, cv=5)
```


```python
print(' Score: {}' .format(results_modelo_ranfor_theBest.mean()))
print(' Desviación: {}'.format(results_modelo_ranfor_theBest.std()))
```

     Score: 0.8711946050096339
     Desviación: 0.0006534036592606193
    


```python

```


```python
#Modelo 12
results_modelo_ranfor_mejores = cross_val_score(estimator=modelo_ranfor_mejores, X=X_train, y=y_train, cv=5)
```


```python
print(' Score: {}' .format(results_modelo_ranfor_mejores.mean()))
print(' Desviación: {}'.format(results_modelo_ranfor_mejores.std()))
```

     Score: 0.870712909441233
     Desviación: 0.0004912350205773353
    


```python

```


```python
#Modelo 11
results_modelo_gbrt_mejores = cross_val_score(estimator=modelo_gbrt_mejores, X=X_train, y=y_train, cv=5)
```


```python
print(' Score: {}' .format(results_modelo_gbrt_mejores.mean()))
print(' Desviación: {}'.format(results_modelo_gbrt_mejores.std()))
```

     Score: 0.869942196531792
     Desviación: 0.0010104131485261774
    

Todos tienen varianzas muy bajas y muy buenos resultados, podría ser válido cualquiera de ellos.
Vamos a verlo de una manera más visual antes de elegir


### * Visualización de los modelos


```python
# Listado con los modelos a evaluar
def get_models():
    models=dict()
    models['Random_forest'] = RandomForestClassifier(bootstrap = True, criterion= 'entropy', max_depth=None,
                                                     n_estimators=150,class_weight='balanced').fit(X_train, y_train)
    models['SVC'] = svm.SVC(class_weight= None, decision_function_shape='ovo')
    models['Gradient_Boosting'] = GradientBoostingClassifier(random_state=0, n_estimators=500,
                                  max_depth=1, learning_rate=0.01)
    return models
    
    
    
# Evaluar los modelos usando la Validación cruzada
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
    scores_cv = cross_val_score(model, X, y, scoring= 'accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores_cv


```


```python
# Lista con los modelos a evaluar
models = get_models()
```


```python
# Evaluar los modelos y guardar los resultados (score, media y desviación típica)
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, X_train, y_train)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
```

    >Random_forest 0.871 (0.001)
    >SVC 0.870 (0.000)
    >Gradient_Boosting 0.870 (0.001)
    


```python
# Visualizar graficamente los modelos evaluados - BOXPLOT

pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
```


    
![png](output_196_0.png)
    


Los tres modelos seleccionados tienen una capacidad predictiva por encima del 85% y varianzas muy bajas, lo que demuestra que sus predicciones no son aleatorias.
El modelo que mejores resultados llega a obtener y más sencilla es su comprensión es Random Forest, por lo que el modelo seleccionado finalmente será el MODELO 12.

## Predicción


```python
X_train, X_test, y_train, y_test = train_test_split(
                                        datos_rf.drop('target', axis = 'columns'),
                                        datos_rf['target'],
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True,
                                        stratify = datos_rf['target'])

```


```python
modelo_ranfor_mejores = RandomForestClassifier(bootstrap = True, criterion= 'entropy', max_depth=None, n_estimators=150,class_weight='balanced').fit(X_train, y_train)
y_pred     = modelo_ranfor_mejores.predict(X_test)

modelo_ranfor_mejores.score(X_test, y_test)
```




    0.8713405238828967




```python
y_pred = modelo_ranfor_mejores.predict(X_test)
```


```python
y_pred
```




    array([1, 1, 1, ..., 1, 1, 1], dtype=int64)



#### Visualización del modelo


```python
def saca_metricas(y1, y2):
    print('Matriz de confusión: ')
    print(multilabel_confusion_matrix(y1, y2, labels=y_test.unique()))
    print('\n Accuracy')
    print(accuracy_score(y1, y2))
    print('\n Precision')
    print(precision_score(y1, y2, average='weighted'))
    print('\n Recall')
    print(recall_score(y1, y2, average='weighted'))
    print('\n f1')
    print(f1_score(y1, y2, average='weighted'))
    '''false_positive_rate, recall, thresholds = roc_curve(y1, y2)
    roc_auc = auc(false_positive_rate, recall)
    print('\n AUC')
    print(roc_auc)
    plt.plot(false_positive_rate, recall, 'b')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title('AUC = %0.2f' % roc_auc)'''
```


```python
saca_metricas(y_test, y_pred)
```

    Matriz de confusión: 
    [[[   5  332]
      [   2 2257]]
    
     [[2575    0]
      [  18    3]]
    
     [[2489    0]
      [ 107    0]]
    
     [[2552    1]
      [  43    0]]
    
     [[2429    1]
      [ 164    2]]]
    
     Accuracy
    0.8713405238828967
    
     Precision
    0.8093159336555452
    
     Recall
    0.8713405238828967
    
     f1
    0.8137698645378528
    

    C:\Users\rtx9652\Anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    

## Importancia de los predictores


```python
importancia_pred = pd.DataFrame(
                            {'predictor': datos_rf.drop(columns = "target").columns,
                             'importancia': modelo_ranfor_mejores.feature_importances_}
                            )

```


```python
print("Importancia de los predictores en el modelo")
print("-------------------------------------------")
importancia_pred.sort_values('importancia', ascending=False)
```

    Importancia de los predictores en el modelo
    -------------------------------------------
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>predictor</th>
      <th>importancia</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>TMEDIA</td>
      <td>0.163695</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lng</td>
      <td>0.122532</td>
    </tr>
    <tr>
      <th>1</th>
      <td>lat</td>
      <td>0.116276</td>
    </tr>
    <tr>
      <th>8</th>
      <td>SOL</td>
      <td>0.092015</td>
    </tr>
    <tr>
      <th>3</th>
      <td>time_ctrl</td>
      <td>0.087114</td>
    </tr>
    <tr>
      <th>10</th>
      <td>PRES_RANGE</td>
      <td>0.085111</td>
    </tr>
    <tr>
      <th>4</th>
      <td>personal</td>
      <td>0.076662</td>
    </tr>
    <tr>
      <th>7</th>
      <td>RACHA</td>
      <td>0.075555</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Año</td>
      <td>0.070233</td>
    </tr>
    <tr>
      <th>0</th>
      <td>superficie</td>
      <td>0.068720</td>
    </tr>
    <tr>
      <th>5</th>
      <td>medios</td>
      <td>0.042087</td>
    </tr>
  </tbody>
</table>
</div>




```python
datos_rf.dtypes
```




    superficie    float64
    lat           float64
    lng           float64
    time_ctrl     float64
    personal        int64
    medios          int64
    TMEDIA        float64
    RACHA         float64
    SOL           float64
    Año             int64
    PRES_RANGE    float64
    target          int64
    dtype: object




```python
'''Guardamos el dataset final del modelo'''
#datos_rf.to_csv('dataset_modelo.csv')'''
```




    'Guardamos el dataset final del modelo'


