import os
import sys
import numpy as np
import pandas as pd
from crycompare import *
from scipy.misc import imread
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential
from skimage.transform import resize
from keras.layers import Dense, LSTM, Dropout
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from poloniex import Poloniex, PoloniexCommandException
from time import sleep, gmtime, strftime
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
pd.set_option('display.max_columns', 10000)
pd.set_option('expand_frame_repr', False)


class CoinBot:
    
    def __init__(self, coins, apikey, secret, wb=2000, wf=6):
        self.coins = coins
        self.apikey = apikey
        self.secret = secret
        self.wb = wb
        self.wf = wf
        
    def train(self, lr=1e-4, verbose=0):
        wb = self.wb
        wf = self.wf
        lc = len(self.coins)
        nbars = 30
        df_dict = {}
        for i in range(lc):
            inc = (i+1)*nbars//lc
            coin = self.coins[i]
            if verbose:
                print('Data downloading ['+'='*(inc-1)+'>'+'.'*(nbars-inc)+'] {:.0f}%'.format((i+1)/lc*100),end='\r')
                sys.stdout.flush()
            histo = History().histoHour(coin,'USD',limit=9999)
            if histo['Data']:
                df_histo = pd.DataFrame(histo['Data'])
                df_histo.index = pd.to_datetime(df_histo['time'],unit='s')
                del df_histo['time']
                del df_histo['volumefrom']
                del df_histo['volumeto']
                
                df_dict[coin] = df_histo
        
        if verbose:
            print('Data downloading ['+'='*30+'] Finished!')
        df = pd.concat(df_dict.values(), axis=1, keys=df_dict.keys())
        df = df.dropna().interpolate(method='nearest')[-wb:]
        rt = pd.DataFrame({coin: df[coin].close/df[coin].open-1 for coin in self.coins}, index=df.index)
        
        opt = []
        for i in range(wb-wf):
            rt_temp = rt.ix[i:i+wf,:]
            df_temp = df.ix[i:i+wf,:]
            sharpe = rt_temp.mean().values/df_temp.std()[::4].values
            opt.append((np.max(sharpe), np.argmax(sharpe)))
        
        def prep(data, pred=False):
            X = np.hstack([to_categorical(data[i:-wf+i], nb_classes=lc) for i in range(wf)])
            y = to_categorical(data[wf:], nb_classes=lc)
            if not pred:
                return X.reshape(wb-2*wf,1,lc*wf), y.reshape(wb-2*wf,1,lc)
            else:
                return np.hstack([X[-1,lc:], y[-1]]).reshape(1,1,lc*wf)
        
        data = np.array([w for r,w in opt])
        X, y = prep(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        model = Sequential()
        model.add(LSTM(16*lc, return_sequences=True, input_shape=X_train.shape[1:]))
        model.add(Dropout(.3))
        model.add(Dense(4*lc, activation='relu'))
        model.add(Dropout(.3))
        model.add(Dense(lc, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer=Adam(lr=lr))
        history = model.fit(X_train, y_train, nb_epoch=20, batch_size=16, validation_data=(X_test, y_test), verbose=verbose)

        if verbose:
            fig = plt.figure(figsize=(15,6))
            ax = fig.add_subplot(111)
            ax.plot(history.history['loss'], lw=3, alpha=1, color='c', label='train loss')
            ax.plot(history.history['val_loss'], lw=3, alpha=.3, color='c', label='test loss')
            ax.plot(history.history['acc'], lw=3, alpha=1, color='r', label='train acc')
            ax.plot(history.history['val_acc'], lw=3, alpha=.3, color='r', label='test acc')
            ax.set_yticks(np.arange(0,1.1,.1))
            plt.legend(loc='upper left',frameon=False, ncol=4)
            plt.tight_layout()
            plt.show()

        w = pd.DataFrame(model.predict(prep(data, True), verbose=0)[0], columns=self.coins, index=['weight'])
        w = w/w.sum(axis=1)[0]
        self.portfolio = w.values[0]
        self.sharpe = (w @ sharpe)[0]
    
    def run(self, transaction=True):
        agent = Poloniex(self.apikey, self.secret)
        BTC2USD = Price().price('BTC','USD')['USD']
        balance = agent.returnBalances()
        portfolio = {}
        for coin in self.coins:
            c_balance = balance[coin]
            if coin=='BTC':
                portfolio[coin] = (BTC2USD, c_balance)
            else:
                portfolio[coin] = (BTC2USD * agent.returnTicker()['BTC_{}'.format(coin)]['last'], c_balance)
        
        df = pd.DataFrame(portfolio, index=range(2))
        df = df.append(df.ix[0] * df.ix[1], ignore_index=True)
        df = df.join(pd.Series(['/','/',round(df.ix[2].sum(),2)], name='total'))
        df = df.append(df.ix[2]/df.ix[2,-1], ignore_index=True)
        df.index = ['price','balance','value','weight']
        df_str = df.round(2).replace(0,'/').astype(str).apply(lambda s: s.str.rjust(8))
        nbars = len(self.coins) * 10 + 17
        print('#'*nbars)
        print('{:-^{width}}'.format(' CURRENT PORTFOLIO ({}) '.format(strftime('%Y-%m-%d %H:%M:%S', gmtime())), width=nbars))
        print(df_str)
        
        self.train()
        print('{:-^{width}}'.format(' SUGGESTED PORTFOLIO (EXP. SHARPE = {:.8f}) '.format(self.sharpe), width=nbars))
        order = pd.DataFrame([self.portfolio * df.ix[-2,-1] / df.ix[0,:-1], df.ix[1,:-1]], index=range(2))
        order = order.append(order.ix[0,:] - order.ix[1,:], ignore_index=True)
        order.index = ['target', 'current', 'order']
        print(order.round(3).replace(0,'/').astype(str).apply(lambda s: s.str.rjust(8)))
        
        if transaction:
            print('{:-^{width}}'.format(' TRANSACTIONS ', width=nbars))
            def mkt_order(coin, amount, slippage=.1):
                currencyPair = 'BTC_{}'.format(coin)
                ticker = agent.returnTicker()
                if amount:
                    if amount > 0:
                        rate = float(ticker[currencyPair]['lowestAsk']) * (1 + slippage)  # in btc
                        fee = rate * amount * 0.0025  # in btc
                        print('{} buy {:.3f} {} at {:.3f} USD/{}, fee = {:.6f} USD'.
                              format(strftime('%Y-%m-%d %H:%M:%S', gmtime()), amount, coin, rate * BTC2USD, coin, fee * BTC2USD), end='')
                        agent.buy(currencyPair=currencyPair, rate=rate, amount=amount, fillOrKill=1)
                    elif amount < 0:
                        rate = float(ticker[currencyPair]['highestBid']) * (1 - slippage)  # in btc
                        fee = -rate * amount * 0.0025  # in btc
                        print('{} sell {:.3f} {} at {:.3f} USD/{}, fee = {:.6f} USD'.
                              format(strftime('%Y-%m-%d %H:%M:%S', gmtime()), -amount, coin, rate * BTC2USD, coin, fee * BTC2USD), end='')
                        agent.sell(currencyPair=currencyPair, rate=rate, amount=-amount, fillOrKill=1)
            
            unset = order.ix[-1,np.setdiff1d(self.coins,['BTC'])].sort_values()
            fail = 0
            for coin in unset.index:
                try:
                    amount = unset[coin]
                    mkt_order(coin, amount)
                    if amount:
                        print('... succeed')
                except PoloniexCommandException:
                    print('... fail (not enough fund)')
                    fail += 1
            fail += sum([len(agent.returnOpenOrders(currencyPair='BTC_{}'.format(coin))) for coin in self.coins if not coin=='BTC'])
            if fail: print('note: {} order(s) not successfully set'.format(fail))
        
        balance = agent.returnBalances()
        portfolio = {}
        for coin in self.coins:
            c_balance = balance[coin]
            if coin=='BTC':
                portfolio[coin] = (BTC2USD, c_balance)
            else:
                portfolio[coin] = (BTC2USD * agent.returnTicker()['BTC_{}'.format(coin)]['last'], c_balance)
        
        df = pd.DataFrame(portfolio, index=range(2))
        df = df.append(df.ix[0] * df.ix[1], ignore_index=True)
        df = df.join(pd.Series(['/','/',round(df.ix[2].sum(),2)], name='total'))
        df = df.append(df.ix[2]/df.ix[2,-1], ignore_index=True)
        df.index = ['price','balance','value','weight']
        df_str = df.round(2).replace(0,'/').astype(str).apply(lambda s: s.str.rjust(8))
        print('{:-^{width}}'.format(' UPDATED PORTFOLIO ({}) '.format(strftime('%Y-%m-%d %H:%M:%S', gmtime())), width=nbars))
        print(df_str)
        
        print('#'*nbars)
    
    def start(self, transaction=True):
        nbar = len(self.coins) * 10 + 17
        img = pd.DataFrame(resize(imread('brand.png', mode='L'), (int(nbar/4.5), nbar), mode='reflect'))
        img[img > 0] = '#'
        img[img == 0] = ' '
        img = img.values
        line = []
        for i in range(img.shape[0]): line.append(''.join(img[i,:])+'\n')
        copyright = ' VERSION 0.0.1 ' + u'\u00a9' + ' 2017 ALLEN FROSTLINE ALL RIGHTS RESERVED '
        lenc = len(copyright)
        line[-1] = line[-1][:-lenc-2] + copyright + '#'
        img = ''.join(line)
        print(img)
        
        while True:
            try:
                self.run(transaction)
                sleep(3600 * self.wf)  # run every wf hours
            except KeyboardInterrupt:
                print('\n\nGOODBYE.')
                sys.exit()
            else:
                print('{} unknown error'.format(strftime('%Y-%m-%d %H:%M:%S', gmtime())))
                sleep(10)  # run again after 10 seconds if error

                
def main():
    coins = ['BTC', 'XRP', 'ETH', 'LTC', 'BCH', 'XMR', 'ZEC', 'DASH', 'ETC']
    apikey = ''
    secret = ''
    t = CoinBot(coins, apikey, secret)
    t.start(False)


if __name__ == '__main__':
    main()
