

# Predict House Prices Using TensorFlow 2.0

![House](house.jpg)

This project predicts house sale prices using ANN (Artificial Neural Networks) using [Tensorflow](https://www.tensorflow.org) 2.0. 

This dataset contains house sale prices for King County, which includes Seattle. The data is about homes sold between May 2014 and May 2015.

You can find the data from [Kaggle](https://www.kaggle.com/harlfoxem/housesalesprediction#kc_house_data.csv).

You can open a new colab [here](https://github.com/rohitcricket/TensorFlow2.0-PredictHousePrices/blob/master/Predict_House_Prices_Using_ANNs.ipynb).

### Step 1: Open a [Colab](https://colab.research.google.com) python notebook

### Step 2: Import TensorFlow and Python Libraries


```
!pip install tensorflow-gpu==2.0.0.alpha0
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

### Step 3: Import the dataset

You will need to mount your drive using the following commands:
For more information regarding mounting, please check this out [here](https://stackoverflow.com/questions/46986398/import-data-into-google-colaboratory).


```
from google.colab import drive
drive.mount('/content/drive')
```

Upload the data file from Kaggle to your Google drive and then access it

```
house_df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/kc-house-data.csv', encoding = 'ISO-8859-1')
```

### Step 4: Visualize the dataset using Seaborn, a python library
```
sns.scatterplot(x = 'sqft_living', y = 'price', data = house_df)
```
See more steps in the colab.

### Step 5: Create testing and training data set and clean the data. 
See steps in the colab.

### Step 6: Train the Model. 
See steps in the colab.

### Step 7: Evaluate the Model. 
See steps in the colab.

### Step 8: Improve the Model
If you are not satisfied with the results, then you can increase the number of independent variables and retrain the same model. See steps in the colab.
