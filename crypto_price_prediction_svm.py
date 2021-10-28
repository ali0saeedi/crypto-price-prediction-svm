import numpy as np
import pandas as pd
import requests
import json
import time

def predict(prediction_counts,timeFrame,symbol):
    end_time = int(time.time())
    start_time = int(time.time())-100000000
    res = requests.get('https://api.kucoin.com/api/v1/market/candles?type='+timeFrame+'&symbol='+ symbol +'&startAt='+str(start_time)+'&endAt='+str(end_time))
    price_array = json.loads(res.content)['data']
    price_array=price_array[::-1]
    for i in range(len(price_array)):
        price_array[i][0] = i
    
    df = pd.DataFrame(price_array,columns=['index', 'opening_price', 'closing_price', 'highest_price', 'lowest_price', 'Transaction_volume', 'Transaction_amount'])
    df = df.drop(['opening_price', 'highest_price', 'lowest_price','Transaction_volume','Transaction_amount'], axis=1)
    
    x = np.array(df.drop(['closing_price'],1))
    x = x[:len(df)-prediction_counts]
    
    y = np.array(df['closing_price'])
    y = y[:-prediction_counts]

    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.2)
 
    prediction_counts_array = []
    for i in range(len(x),len(x) + prediction_counts):
        tmp_arr = [i]
        prediction_counts_array.append(tmp_arr)
    # print(prediction_counts_array)

    from sklearn.svm import SVR
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.00001)
    svr_rbf.fit(xtrain, ytrain)
    svm_prediction = svr_rbf.predict(prediction_counts_array)

    prediction_price_first = float(svm_prediction[0])
    prediction_price_last = float(svm_prediction[-1])
    real_price_first = float(df._get_column_array(1)[1500-len(svm_prediction)])
    real_price_last = float(df._get_column_array(1)[1499])
    print("prediction_price_first",prediction_price_first)
    print("prediction_price_last",prediction_price_last)
    print("real_price_first",real_price_first)
    print("real_price_last",real_price_last)
    prediction_price_change_percentage=((prediction_price_last-prediction_price_first)/prediction_price_first)*100
    print("Prediction Price Change : ", (prediction_price_change_percentage))
    print("Real Price Change : ",(real_price_last-real_price_first)/real_price_first*100 )
    return prediction_price_first,prediction_price_last, prediction_price_change_percentage


print(predict(100,'3min','BTC-USDT'))