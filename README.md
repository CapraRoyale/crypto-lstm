# crypto-lstm

## Using Tensorflow's LSTM (Long Short-term Memory) RNN (Recurrent Neural Network) to predict a cryptocurrencies next-day closing price

---

 [This notebook](/lstm_crypto_predictor-fng.ipynb) predicts the value of Bitcoin based on the ['Fear and Greed' index](https://alternative.me/crypto/fear-and-greed-index/)

 [This notebook](/lstm_crypto_predictor-closing.ipynb) predicts the value of Bitcoin based on a rolling 10-day window of closing prices

You can see, given the results, that the prediction based on closing prices has a smaller margin of error, averaging about 75% of the actual value in it's prediction, versus the Fear and Greed index at 61%.
