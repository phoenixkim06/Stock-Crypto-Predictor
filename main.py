import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

def predictCrypto(choice, crypto_currency, future_day, start):
    future_day = int((float(future_day)))
    against_currency = 'USD'

    #start/end date
    start = dt.datetime(2016, 1, 1)
    end = dt.datetime.now()

    #pull data off of the web
    #                     #crypto, #source, #start, #end
    #check if it is doing it for a stock or crypto
    if choice == "1":
        data = web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo', start, end)
    else:
        data = web.DataReader(crypto_currency, 'yahoo', start, end)

    #prepare data
    scaler = MinMaxScaler(feature_range=(0, 1))
    #list of closing values
    scaled_data = scaler.fit_transform((data['Close'].values.reshape(-1,1)))

    #using 60 days to predict
    prediction_days = 60
    global x_train
    global y_train
    x_train, y_train = [], []

    for x in range(prediction_days, len(scaled_data) - future_day):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x + future_day, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    #creating neural network
    model =Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    #Testing the model
    test_start = dt.datetime(2020, 1, 1)
    test_end = dt.datetime.now()

    #determines what data to choose depending on whether the user chooses to predict crypto or stock prices
    if choice == "1":
        test_data = web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo', test_start, test_end)
        actual_values = test_data['Close'].values
    else:
        test_data = web.DataReader(crypto_currency, 'yahoo', test_start, test_end)
        actual_values = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.fit_transform(model_inputs)

    x_test = []
    for x in range(prediction_days, len(model_inputs) + 1):
        x_test.append(model_inputs[x-prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    prediction = model.predict(x_test)
    prediction = scaler.inverse_transform(prediction)

    plt.plot(actual_values, color = 'black', label = 'Actual Values')
    plt.plot(prediction, color = 'green', label = 'Prediction')
    plt.title(f'{crypto_currency}-{against_currency}')
    plt.xlabel('Time')
    plt.ylabel('Conversion')
    plt.legend(loc='upper left')
    plt.show()

    #Prediction for next day
    real_data = [model_inputs[len(model_inputs) - prediction_days:len(model_inputs) + 1, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)

    #prediction
    print(f"Prediction price of {crypto_currency}: {prediction}")

#fucntion that tests whether or not the user inputs are valid or not
def testing_valid(choice, userChoice, start = dt.datetime(2016, 1, 1), end = dt.datetime.now()):
    if choice == "1":
        try:
            data = web.DataReader(f'{userChoice}-USD', 'yahoo', start, end)
            return True
        except:
            return False
    else:
        try:
            data = web.DataReader(userChoice, 'yahoo', start, end)
            return True
        except:
            return False

print("Crypto/Stock Price Prediction")
print("*DISCLAIMER: Not Intended for financial purposes*")
print("\n Crypto: 1 \n Stock: 2 \n")
#function that runs the user inputs
def runProgram():
    #menu
    choice = input("Pick either 1 or 2 for predicting crypto or stock prices: ")
    while choice != "1" and choice != "2":
        choice = input("Pick either 1 or 2 for predicting crypto or stock prices: ")

    #menu for crypto
    if choice == "1":
        print("\n Bitcoin: BTC \n Cardano: ADA \n Ethereum: ETH \n Ripple: XRP \n Litecoin: LTC \n")
        crypto_currency = input("Pick a crypto from the menu above or enter in your own cryptocurrency: ")

    #menu for stocks
    else:
        print("\n Amazon: AMZN \n Google: GOOGL \n Apple: AAPL \n Tesla: TSLA \n Microsoft: MSFT \n")
        crypto_currency = input("Pick a stock from the menu above or enter in your own stock: ")

    crypto_currency = crypto_currency.upper()
    while (testing_valid(choice, crypto_currency)) == False:
        if choice == "1":
            crypto_currency = input("Pick a crypto from the menu above or enter in your own cryptocurrency: ")
        else:
            crypto_currency = input("Pick a stock from the menu above or enter in your own stock: ")

    future_day = input("Pick a number of days in the future that you would like to predict the price for (1-180 days): ")

    while future_day.isdigit() == False or int(future_day) < 0 or int(future_day) > 180:
        future_day = input("Invalid number of days in the future. Put in a specific number of days: ")

    todays_date = dt.datetime.today()
    using_year = todays_date.year
    using_month = todays_date.month
    using_day = todays_date.day
    custom = None
    monthRange = list(range(1, 13))
    dayRange = list(range(1, 32))
    if int(future_day) >= 120:
        using_year = todays_date.year - 3
        start = dt.datetime(using_year, using_month, using_day)
        custom = input("Would you like to use a custom start date? Type in either Y or N: ")
        custom = custom.upper()
        while custom != "Y" and custom != "N":
            custom = input("Invalid input. Type in either Y or N: ")
            custom = custom.upper()
        if custom == "Y":
            using_year = input("Enter in a year in its full length. Ex: 2015: ")
            while using_year.isdigit() == False:
                using_year = input("Invalid input. Enter in a 4 digit year. Ex: 2015: ")
            while len(using_year) != 4:
                using_year = input("Invalid input. Enter in a 4 digit year. Ex: 2015: ")
            using_year = int(using_year)

            using_month = input("Enter in a month number. Ex: January = 1, December = 12: ")
            while using_month.isdigit() == False:
                using_month = input("Invalid input. Enter in a numerical month. Ex: January = 1, December = 12: ")
            while int(using_month) not in monthRange:
                using_month = input("Invalid input. Enter in a numerical month. Ex: January = 1, December = 12: ")
            using_month = int(using_month)

            using_day = input("Enter in a day of the month. Ex: 15th = 15: ")
            while using_day.isdigit() == False:
                using_day = input("Invalid input. Enter in a numerical day of the month. Ex: 15th = 15: ")
            while int(using_day) not in dayRange:
                using_day = input("Invalid input. Enter in a numerical day of the month. Ex: 15th = 15: ")
            using_day = int(using_day)
            start = dt.datetime(using_year, using_month, using_day)
        else:
            start = dt.datetime(using_year, using_month, using_day)

    elif int(future_day) >= 90:
        using_year = todays_date.year - 2
        start = dt.datetime(using_year, using_month, using_day)
        custom = input("Would you like to use a custom start date? Type in either Y or N: ")
        custom = custom.upper()
        while custom != "Y" and custom != "N":
            custom = input("Invalid input. Type in either Y or N: ")
            custom = custom.upper()
        if custom == "Y":
            using_year = input("Enter in a year in its full length. Ex: 2015: ")
            while using_year.isdigit() == False:
                using_year = input("Invalid input. Enter in a 4 digit year. Ex: 2015: ")
            while len(using_year) != 4:
                using_year = input("Invalid input. Enter in a 4 digit year. Ex: 2015: ")
            using_year = int(using_year)

            using_month = input("Enter in a month number. Ex: January = 1, December = 12: ")
            while using_month.isdigit() == False:
                using_month = input("Invalid input. Enter in a numerical month. Ex: January = 1, December = 12: ")
            while int(using_month) not in monthRange:
                using_month = input("Invalid input. Enter in a numerical month. Ex: January = 1, December = 12: ")
            using_month = int(using_month)

            using_day = input("Enter in a day of the month. Ex: 15th = 15: ")
            while using_day.isdigit() == False:
                using_day = input("Invalid input. Enter in a numerical day of the month. Ex: 15th = 15: ")
            while int(using_day) not in dayRange:
                using_day = input("Invalid input. Enter in a numerical day of the month. Ex: 15th = 15: ")
            using_day = int(using_day)
            start = dt.datetime(using_year, using_month, using_day)
        else:
            start = dt.datetime(using_year, using_month, using_day)
    elif int(future_day) >= 60:
        using_year = todays_date.year - 2
        start = dt.datetime(using_year, using_month, using_day)
        custom = input("Would you like to use a custom start date? Type in either Y or N: ")
        custom = custom.upper()
        while custom != "Y" and custom != "N":
            custom = input("Invalid input. Type in either Y or N: ")
            custom = custom.upper()
        if custom == "Y":
            using_year = input("Enter in a year in its full length. Ex: 2015: ")
            while using_year.isdigit() == False:
                using_year = input("Invalid input. Enter in a 4 digit year. Ex: 2015: ")
            while len(using_year) != 4:
                using_year = input("Invalid input. Enter in a 4 digit year. Ex: 2015: ")
            using_year = int(using_year)

            using_month = input("Enter in a month number. Ex: January = 1, December = 12: ")
            while using_month.isdigit() == False:
                using_month = input("Invalid input. Enter in a numerical month. Ex: January = 1, December = 12: ")
            while int(using_month) not in monthRange:
                using_month = input("Invalid input. Enter in a numerical month. Ex: January = 1, December = 12: ")
            using_month = int(using_month)

            using_day = input("Enter in a day of the month. Ex: 15th = 15: ")
            while using_day.isdigit() == False:
                using_day = input("Invalid input. Enter in a numerical day of the month. Ex: 15th = 15: ")
            while int(using_day) not in dayRange:
                using_day = input("Invalid input. Enter in a numerical day of the month. Ex: 15th = 15: ")
            using_day = int(using_day)
            start = dt.datetime(using_year, using_month, using_day)
        else:
            start = dt.datetime(using_year, using_month, using_day)
    else:
        using_year = todays_date.year - 1
        start = dt.datetime(using_year, using_month, using_day)
        custom = input("Would you like to use a custom start date? Type in either Y or N: ")
        custom = custom.upper()
        while custom != "Y" and custom != "N":
            custom = input("Invalid input. Type in either Y or N: ")
            custom = custom.upper()
        if custom == "Y":
            using_year = input("Enter in a year in its full length. Ex: 2015: ")
            while using_year.isdigit() == False:
                using_year = input("Invalid input. Enter in a 4 digit year. Ex: 2015: ")
            while len(using_year) != 4:
                using_year = input("Invalid input. Enter in a 4 digit year. Ex: 2015: ")
            using_year = int(using_year)

            using_month = input("Enter in a month number. Ex: January = 1, December = 12: ")
            while using_month.isdigit() == False:
                using_month = input("Invalid input. Enter in a numerical month. Ex: January = 1, December = 12: ")
            while int(using_month) not in monthRange:
                using_month = input("Invalid input. Enter in a numerical month. Ex: January = 1, December = 12: ")
            using_month = int(using_month)

            using_day = input("Enter in a day of the month. Ex: 15th = 15: ")
            while using_day.isdigit() == False:
                using_day = input("Invalid input. Enter in a numerical day of the month. Ex: 15th = 15: ")
            while int(using_day) not in dayRange:
                using_day = input("Invalid input. Enter in a numerical day of the month. Ex: 15th = 15: ")
            using_day = int(using_day)
            start = dt.datetime(using_year, using_month, using_day)
        else:
            start = dt.datetime(using_year, using_month, using_day)
    predictCrypto(choice, crypto_currency, future_day, start)

runProgram()
