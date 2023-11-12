Exp.No : 05 
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
Date : 09.10.2023 
<br>
# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
- The given problem is to predict the google stock price based on time.
- For this we are provided with a dataset which contains features like
    - Date
    - Opening Price
    - Highest Price
    - Lowest Price
    - Closing Price
    - Adjusted Closing
    - Price and
    - Volume
- Based on the given features, develop a RNN model to predict the price of stocks in future

## Neural Network Model
<p align="center">
<img src="https://github.com/Kaushika-Anandh/rnn-stock-price-prediction/blob/main/3.jpg" width="650" height="350">
</p>

<br>

## DESIGN STEPS

- **Step 1:** Import the required packages
- **Step 2:** Load the dataset
- **Step 3:** Perform the necessary data preprocessing 
- **Step 4:** Build and fit the data in the Learning model
- **Step 5:** Predict using the fit model
- **Step 6:** Check the error value of the predicted pricing model 


## PROGRAM
> Developed by : Kaushika A <br>
>Register No. : 212221230048

**importing libraries**
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential
```

**loading datasets**
```python
dataset_train = pd.read_csv('trainset.csv')
dataset_train.columns
```
```python
dataset_train.head()
train_set = dataset_train.iloc[:,1:2].values
print(type(train_set))
train_set.shape
```

**scaling data**
```python
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)
training_set_scaled.shape
```

```python
X_train_array = []
y_train_array = []
for i in range(60, 1259):
    X_train_array.append(training_set_scaled[i-60:i,0])
    y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))
X_train.shape
```

**network model**
```python
length = 60
n_features = 1

model = Sequential([layers.SimpleRNN(42,input_shape=(60,1)),
                    layers.Dense(1)])
model.compile(optimizer='adam',loss='mse')
model.summary()
model.fit(X_train1,y_train,epochs=20, batch_size=32)
```

```python
dataset_test = pd.read_csv('testset.csv')
test_set = dataset_test.iloc[:,1:2].values
test_set.shape
```

```python
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
y_test = []
for i in range(60,1384):
    X_test.append(inputs_scaled[i-60:i,0])
    y_test.append(inputs_scaled[i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
X_test.shape
```

```python
predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)
plt.plot(np.arange(0,1384),inputs, color='purple', label = 'Test data')
plt.plot(np.arange(60,1384),predicted_stock_price, color='skyblue', label = 'Predicted stock price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
```

**metrics**
```python
from sklearn.metrics import mean_squared_error as mse
mse(y_test,predicted_stock_price)
```
## OUTPUT

### True Stock Price, Predicted Stock Price vs time

<img src="https://github.com/Kaushika-Anandh/rnn-stock-price-prediction/blob/main/1.png" width="380" height="280">


### Mean Square Error

<img src="https://github.com/Kaushika-Anandh/rnn-stock-price-prediction/blob/main/2.PNG" width="350" height="50">


## RESULT
Thus, a Recurrent Neural Network model for stock price prediction is developed.
