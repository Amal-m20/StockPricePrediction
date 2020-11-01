## Description: This program uses an artificial recurrent neural network called Long Short Term Memory (LSTM)
##              to predict the closing stock price of a corporation using the past 60 day stock price.

# Import the libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#user input on what company to use
company = input("Enter company: ")
startdate = input("Enter start date: ")
enddate = input("Enter end date: ")
#Get the stock quote
#df = dataframe
df = web.DataReader(company, data_source='yahoo', start = startdate, end=enddate)
#show the data
print(df)

#visualizing the closing price history
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Price USD ($)', fontsize = 18)

show_graph = input("Would like to visualize the closing price history? [y/n]: ")

if show_graph == 'y':
    plt.show()

#Create a new dataframe with only the closed column
data = df.filter(['Close'])
#Convert the dataframe to a numpy array
dataset = data.values
#Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8)

#print(training_data_len)

#scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scale_data = scaler.fit_transform(dataset)

#print(scale_data)

#Create the training data set
#create scaled traning data set
train_data = scale_data[0:training_data_len , :]

#split the data into x_train and y_train data sets

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
   #if i<=60:
   #    print(x_train)
   #    print(y_train)
   #    print()

#convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#print(x_train.shape)

#Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
#train the model
model.fit(x_train,y_train, batch_size=1, epochs=1)

#create the testing data set
#create a new array containing scaled values from index 1543 to 2003
test_data = scale_data[training_data_len - 60:, :]
#create the data sets x_text and y_test
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

#convert the data to a numpy array
x_test = np.array(x_test)

#reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
print(rmse)

#plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Price USD($)', fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Validation', 'Prediction'], loc = 'lower right')

show_graph2 = input("Would like to visualize the model? [y/n]: ")
if show_graph2 == 'y':
    plt.show()

#show the valid and predicted prices
print(valid)

#get the quote
company_quote = web.DataReader(company, data_source='yahoo', start = startdate, end=enddate)
# Create a new dataframe
new_df = company_quote.filter(['Close'])
#get the last 60 days closing price values and convert the dataframe to an array
last60_days = new_df[-60:].values

#scale the data to be values between 0 and 1
last60_days_scaled = scaler.transform(last60_days)

#create an empty list
X_test  = []
#append the past 60 days 
X_test.append(last60_days_scaled)

#convert the X_test to a numpy array
X_test = np.array(X_test)
#reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#get the predicted scale price
pred_price = model.predict(X_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)


#Check if the date exixt in the dataset and print it
df2 = web.DataReader('AAPL', data_source='yahoo', start='01/01/2001')
new_df = df2.reset_index()
wrongdate = True
while(wrongdate == True):
    prediction_date = input('Enter prediction date: ')
    if(new_df['Date'].isin([prediction_date])).any():
        wrongdate = False
        company_quote2 = web.DataReader('AAPL', data_source='yahoo', start = prediction_date, end=prediction_date)
        print(company_quote2['Close'])
    else:
        print('Date does not exist in database')
