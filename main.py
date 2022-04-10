import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tkinter as tk
from tkinter import *
from tkinter import ttk
r = tk.Tk()
#Basic Formatting + Title
r.title("Stock/Crypto Prediction")
r.geometry("350x550")
r.configure(bg="#222222")
r.resizable(0, 0)

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

#function that takes in the inputs and returns prediction
def predictCrypto(choice, crypto_currency, future_day, start = dt.datetime(2016, 1, 1)):
    future_day = int((float(future_day)))
    against_currency = 'USD'

    #start/end date
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

    #get all of the stock prices
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

    #graph formatting
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
    return prediction

stockCryptoRadioButton = IntVar()
#button on whether to select crypto or stock
#1 = crypto, 2 = stock
Radiobutton(r, text='Crypto', variable = stockCryptoRadioButton, value = "1", bg = "#222222", fg = "#ff7452", activebackground = "#222222", activeforeground = "#ff7452", command = lambda: printCrypto()).place(x = 0, y = 0)
Radiobutton(r, text='Stock', variable = stockCryptoRadioButton, value = "2", bg = "#222222", fg = "#ff7452", activebackground = "#222222", activeforeground = "#ff7452", command = lambda: printStocks()).place(x = 65, y = 0)

#button to select whether to choose a default or custom start date
#1 = default, 2 = custom
dates = IntVar()
Radiobutton(r, text='Default Date', variable = dates, value = "1", bg = "#222222", fg = "#c270c0", activebackground = "#222222", activeforeground = "#c270c0", command = lambda: destroyComboBox()).place(x = 0, y = 320)
Radiobutton(r, text='Custom Date', variable = dates, value = "2", bg = "#222222", fg = "#c270c0", activebackground = "#222222", activeforeground = "#c270c0", command = lambda: displayComboBox()).place(x = 110, y = 320)

#example cryptos/stocks
cryptoList = "Example Crypto:\n Bitcoin: BTC\n Cardano: ADA\n Ethereum: ETH\n Ripple: XRP\n Litecoin: LTC"
stockList = "Example Stocks:\n Amazon: AMZN\n Google: GOOGL\n Apple: AAPL\n Tesla: TSLA\n Microsoft: MSFT"
stockCryptoLabel = Label(r, bg = "#222222", fg = "#abf5d1").place(x = 0, y = 40)


#Entry box for the ticker symbol
Label(r, text = "Enter Ticker Symbol:", bg = "#222222", fg = "#028fa3").place(x = 0, y = 180)
stockCryptoEntry = StringVar(r)
Entry(r, width = 35, textvariable = stockCryptoEntry, bg = "#222222", fg = "#d9e2ec").place(x = 3, y = 200)

#Entry box for the # of days in the future to predict
Label(r, text = "Enter # of Days Into the Future to Predict:", bg = "#222222", fg = "#028fa3").place(x = 0, y = 240)
numberOfFutureDays = StringVar(r)
Entry(r, width = 35, textvariable = numberOfFutureDays, bg = "#222222", fg = "#d9e2ec").place(x = 3, y = 260)
Label(r, text = "Choose a Data Colletion Start Date:", bg = "#222222", fg = "#ffab00").place(x = 0, y = 300)
#boxes to select custom dates
comboBOX1 = ttk.Combobox(r)
comboBOX2 = ttk.Combobox(r)
comboBOX3 = ttk.Combobox(r)
#lists with months, years, and days
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
years = ["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022"]
days = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
        "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"]

#show custom date boxes
def displayComboBox():
    global comboBOX1
    global comboBOX2
    global comboBOX3
    global monthvar
    global yearvar
    global dayvar
    global months
    monthvar = StringVar(r)
    yearvar = StringVar(r)
    dayvar = StringVar(r)
    #formatting/style of comboboxes
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TCombobox", fieldbackground = "#222222", background = "#222222", foreground = "#d9e2ec", darkcolor = "#222222", lightcolor = "#222222")
    #select boxes for custom date
    comboBOX1 = ttk.Combobox(r, values = months, width = 13, textvariable = monthvar)
    comboBOX1.set("Pick a Month")
    comboBOX1.place(x = 3, y = 350)
    comboBOX2 = ttk.Combobox(r, values = years, width = 13, textvariable = yearvar)
    comboBOX2.set("Pick a Year")
    comboBOX2.place(x = 119, y = 350)
    comboBOX3 = ttk.Combobox(r, values = days, width = 13, textvariable = dayvar)
    comboBOX3.set("Pick a Day")
    comboBOX3.place(x = 236, y = 350)

#function that gets rid of custom date choices if default choice is chosen
def destroyComboBox():
    global comboBOX1
    global comboBOX2
    global comboBOX3
    global monthvar
    global yearvar
    global dayvar
    comboBOX1.destroy()
    comboBOX2.destroy()
    comboBOX3.destroy()
    monthvar = None
    yearvar = None
    dayvar = None

#submit button
Button(r, text = "Submit for Prediction", bg = "#57d9a3", fg = "#222222", activebackground = "#57d9a3", activeforeground = "#222222", command = lambda: getValues()).place(x = 3, y = 400)

#print the crypto list
def printCrypto():
    global stockCryptoLabel
    if (stockCryptoLabel != None):
        stockCryptoLabel.destroy()
        stockCryptoLabel = Label(r, text = cryptoList, bg = "#222222", fg = "#abf5d1")
        stockCryptoLabel.place(x = 0, y = 40)
    else:
        stockCryptoLabel = Label(r, text = cryptoList, bg = "#222222", fg = "#abf5d1")
        stockCryptoLabel.place(x = 0, y = 40)

#print the stock list
def printStocks():
    global stockCryptoLabel
    if (stockCryptoLabel != None):
        stockCryptoLabel.destroy()
        stockCryptoLabel = Label(r, text = stockList, bg = "#222222", fg = "#abf5d1")
        stockCryptoLabel.place(x = 0, y = 40)
    else:
        stockCryptoLabel = Label(r, text = stockList, bg = "#222222", fg = "#abf5d1")
        stockCryptoLabel.place(x = 0, y = 40)

#reset the status text for each entry box
status1 = Label(r, text = "                                                                                                ", bg = "#222222").place(x = 0, y = 220)
status2 = Label(r, text = "                                                                                                ", bg = "#222222").place(x = 0, y = 280)

#status message
final = Label(r, text = "*May take a couple minutes to get results", bg = "#222222", fg = "#d9e2ec").place(x = 0, y = 440)

#function that gets the values the user inputs
def getValues():
    global final
    status1 = Label(r, text="                                                                                                ", bg="#222222").place(x=0, y=220)
    status2 = Label(r, text="                                                                                                ", bg="#222222").place(x=0, y=280)
    final = Label(r, text = "                                                                                                          ", bg = "#222222", fg = "#d9e2ec").place(x = 0, y = 440)
    global stockCryptoRadioButton
    global stockCryptoEntry
    global numberOfFutureDays
    global monthvar
    global yearvar
    global dayvar
    global dates
    global months
    global monthNum
    #CRYPTO
    monthNumbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    temp1 = False
    temp2 = False
    #gets crypto entry and re-prompt the user if needed
    if stockCryptoRadioButton.get() == 1: #crypto
        try: #custom
            #gets the proper month numerical value
            for i in range(len(months)):
                if months[i] == monthvar.get():
                    monthNum = i + 1
            #print out value
            try:
                web.DataReader(f'{stockCryptoEntry.get().upper()}-{"USD"}', 'yahoo', dt.datetime(2016, 1, 1), dt.datetime.now())
                temp1 = True
            except:
                status1 = Label(r, text = "Invalid crypto. Enter in an existing cryptocurrency.", bg = "#222222", fg = "#d9e2ec").place(x = 0, y = 220)
                temp1 = False
            try:
                int(numberOfFutureDays.get())
                temp2 = True
            except:
                status2 = Label(r, text = "Invalid number of future days. Enter in an integer.", bg = "#222222", fg = "#d9e2ec").place(x = 0, y = 280)
                temp2 = False
            if (temp1 == True and temp2 == True):
                Label(r, text="Prediction Price of " + stockCryptoEntry.get().upper() + ": " + str(predictCrypto("1", stockCryptoEntry.get().upper(), numberOfFutureDays.get(), start = dt.datetime(int(yearvar.get()), monthNum, int(dayvar.get()))))).place(x=0, y=440)
        except: #default
            try:
                data = web.DataReader(f'{stockCryptoEntry.get().upper()}-{"USD"}', 'yahoo', dt.datetime(2016, 1, 1), dt.datetime.now())
                temp1 = True
            except:
                status1 = Label(r, text = "Invalid crypto. Enter in an existing cryptocurrency.", bg = "#222222", fg = "#d9e2ec").place(x = 0, y = 220)
                temp1 = False
            try:
                int(numberOfFutureDays.get())
                temp2 = True
            except:
                status2 = Label(r, text = "Invalid number of future days. Enter in an integer.", bg = "#222222", fg = "#d9e2ec").place(x = 0, y = 280)
                temp2 = False
            if (temp1 == True and temp2 == True):
                Label(r, text="Prediction Price of " + stockCryptoEntry.get().upper() + ": " + str(predictCrypto("1", stockCryptoEntry.get().upper(), numberOfFutureDays.get()))).place(x=0, y=440)
    #gets stock entry and re-prompt the user if needed
    if stockCryptoRadioButton.get() == 2: #stock
        try: #custom
            #gets the proper month numerical value
            for i in range(len(months)):
                if months[i] == monthvar.get():
                    monthNum = i + 1
            #print out value
            try:
                data = web.DataReader(stockCryptoEntry.get().upper(), 'yahoo', dt.datetime(2016, 1, 1), dt.datetime.now())
                temp1 = True
            except:
                status1 = Label(r, text = "Invalid stock. Enter in an existing stock.", bg = "#222222", fg = "#d9e2ec").place(x = 0, y = 220)
                temp1 = False
            try:
                numbers = int(numberOfFutureDays.get())
                temp2 = True
            except:
                status2 = Label(r, text = "Invalid number of future days. Enter in an integer.", bg = "#222222", fg = "#d9e2ec").place(x = 0, y = 280)
                temp2 = False
            if (temp1 == True and temp2 == True):
                Label(r, text="Prediction Price of " + stockCryptoEntry.get().upper() + ": " + str(predictCrypto("2", stockCryptoEntry.get().upper(), numberOfFutureDays.get(), start=dt.datetime(int(yearvar.get()), monthNum, int(dayvar.get()))))).place(x=0, y=440)
        except: #default
            try:
                data = web.DataReader(stockCryptoEntry.get().upper(), 'yahoo', dt.datetime(2016, 1, 1), dt.datetime.now())
                temp1 = True
            except:
                status1 = Label(r, text = "Invalid stock. Enter in an existing stock.", bg = "#222222", fg = "#d9e2ec").place(x = 0, y = 220)
                temp1 = False
            try:
                numbers = int(numberOfFutureDays.get())
                temp2 = True
            except:
                status2 = Label(r, text = "Invalid number of future days. Enter in an integer.", bg = "#222222", fg = "#d9e2ec").place(x = 0, y = 280)
                temp2 = False
            if (temp1 == True and temp2 == True):
                Label(r, text="Prediction Price of " + stockCryptoEntry.get().upper() + ": " + str(predictCrypto("2", stockCryptoEntry.get().upper(), numberOfFutureDays.get())), bg = "#222222", fg = "#d9e2ec").place(x=0, y=440)

r.mainloop()
