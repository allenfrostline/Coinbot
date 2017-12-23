import os
import sys
import numpy as np
from config import *
from time import sleep
from pytz import timezone
from datetime import datetime
from timeout_decorator import timeout, TimeoutError
from poloniex import Poloniex, PoloniexCommandException


def exceptionHandler(timeOut, again):
    def decorator(func):
        @timeout(timeOut)
        def wrapper(*args, **kwargs):
            global agent
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                print(strTime() + ' Good bye.')
                sys.exit()
            except TimeoutError as e:
                print(strTime() + ' Time out.')
                agent = Poloniex(apikey, secret)
                sleep(.1)
                if again: return wrapper(*args, **kwargs)
            except (PoloniexCommandException, Exception) as e:
                print(strTime(), e)
                agent = Poloniex(apikey, secret)
                sleep(.1)
                if again: return wrapper(*args, **kwargs)
        return wrapper
    return decorator


@exceptionHandler(timeOut=30, again=True)
def sim(path):
    global pairs
    lastState = path[-1]
    amountFrom = lastState[0]
    fromCoin = lastState[1]
    ffromCoin = None if (len(path) == 1) else path[-2][1]
    mask = [fromCoin in pair.split('_') for pair in pairs]
    pairs_filtered = pairs[mask]
    pathList = []
    for pair in pairs_filtered:
        temp = pair.split('_')
        toCoin = temp[1] if fromCoin == temp[0] else temp[0]
        if toCoin == ffromCoin: continue
        ret = orderConfig(toCoin, fromCoin, amountFrom)
        if ret is None: continue
        pair, rates, amounts, reverse = ret
        non_reverse = 1 - reverse
        rate = rates[non_reverse]
        amount = amounts[non_reverse] * (1 - feeRate)
        amountTo = amount * rate**non_reverse * (1 - slippage)
        amountBase = amount * rate
        if amountBase < minOrder(temp[0]):
            pairs = np.delete(pairs, np.where(pairs == pair))
            del orderBooks[pair]
            continue
        if toCoin == baseCoin: pathList.append(path + [(amountTo, toCoin)])
        elif len(path) < maxLen - 1:
            temp = sim(path + [(amountTo, toCoin)])
            if not temp: continue
            if type(temp[0]) is tuple: pathList.append(temp)
            else: pathList += temp
    if len(pathList) > 1: return pathList
    elif len(pathList) == 1: return pathList[0]


def optimum(pathList):
    rtioList = [path[-1][0] / path[0][0] for path in pathList]
    optRank = np.argmax(rtioList)
    optPath = pathList[optRank]
    optRetn = rtioList[optRank] - 1
    return optPath, optRetn


@exceptionHandler(timeOut=10, again=False)
def order(baseCoin, coin, amount):
    global pairs
    if (baseCoin == coin) or (amount == 0):
        print(strTime(), 'Null order')
        return False
    ret = orderConfig(baseCoin, coin, abs(amount))
    if not ret: return False
    pair, rates, amounts, reverse = ret
    if ((not reverse) and (amount > 0)) or (reverse and (amount < 0)):  # buy pair, index = 0
        rate = rates[0] * (1 + slippage)
        amount = amounts[0] * (1 - feeRate)
        symbol = ' Buy '
        func = agent.buy
    else:  # sell pair, index = 1
        rate = rates[1] * (1 - slippage)
        amount = amounts[1] * (1 - feeRate)
        symbol = ' Sell '
        func = agent.sell
    baseCoin, coin = pair.split('_')
    amountBase = amount * rate
    if amountBase < minOrder(baseCoin):
        # print(strTime(), 'Order size {:.6f} too small.'.format(amountBase))
        return False
    print(strTime() + symbol + GRAY + '{:.6f}'.format(amount) + RESET + coin + ' @ ' + GRAY + '{:.6f}'.format(rate) + RESET + baseCoin + '/' + coin + '.')
    func(pair, rate, amount)
    return True


@exceptionHandler(timeOut=10, again=True)
def closeAll():
    openOrders = agent.returnOpenOrders()
    for o in openOrders:
        for n in openOrders[o]:
            agent.cancelOrder(int(n['orderNumber']))
    balances = availableBalances()
    for coin in balances:
        if coin in [baseCoin, 'BTC']: continue
        order('BTC', coin, -balances[coin])
    if not (baseCoin == 'BTC'):
        balances = availableBalances()
        order(baseCoin, 'BTC', -balances['BTC'])
    return availableBalances()


@exceptionHandler(timeOut=60, again=True)
def calcMv():
    global orderBooks
    for pair in pairs:
        orderBook = agent.returnOrderBook(pair, 1000)
        orderBooks[pair] = {
            'asks': [[float(x[0]), float(x[1])] for x in orderBook['asks']],
            'bids': [[float(x[0]), float(x[1])] for x in orderBook['bids']]
        }
    n_BTC = 0
    balances = closeAll()
    USDT_BTC = (orderBooks['USDT_BTC']['asks'][0][0] + orderBooks['USDT_BTC']['bids'][0][0]) / 2
    for coin in balances:
        b = balances[coin]
        if coin == 'BTC': n_BTC += b
        elif coin == 'USDT': n_BTC += b / USDT_BTC
        else: n_BTC += b * (orderBooks['BTC_' + coin]['asks'][0][0] + orderBooks['BTC_' + coin]['bids'][0][0]) / 2
    mv = n_BTC * USDT_BTC
    return mv, balances


@exceptionHandler(timeOut=10, again=True)
def availableBalances():
    return agent.returnAvailableAccountBalances(account='exchange')['exchange']


@exceptionHandler(timeOut=10, again=False)
def orderConfig(baseCoin, coin, amount):
    # rates for {pair} to buy/sell {amount} {coin}
    global orderBooks, pairs
    assert amount >= 0, 'Amount cannot be negative.'
    pair = baseCoin + '_' + coin
    if (pair in orderBooks):
        reverse = 0
    else:
        pair = coin + '_' + baseCoin
        reverse = 1
    rates = []
    amounts = []
    types = ['asks', 'bids']
    for t in types:
        cum = 0
        for (p, a) in orderBooks[pair][t]:
            p_or_1 = p**reverse
            cum += a * p_or_1
            if cum >= amount:
                rates.append(p)
                amounts.append(amount / p_or_1)
                break
    if len(rates) < 2:  # depth is not enough
        print(strTime() + ' Order book depth of {} not enough, deleted.'.format(pair))
        pairs = np.delete(pairs, np.where(pairs == pair))
        del orderBooks[pair]
        return None
    return pair, rates, amounts, reverse


minOrder = lambda coin: 0.5 if coin == 'USDT' else 0.0001
now = lambda: datetime.now(timezone('Europe/Amsterdam'))
strTime = lambda: '[' + GREEN + now().strftime('%Y-%m-%d %H:%M:%S') + RESET + ']'


@exceptionHandler(timeOut=30, again=False)
def trade(mv0):
    global orderBooks, seconds, balances, mv, start, end
    end = now()
    seconds.append((end - start).seconds)
    totalRounds = len(seconds)
    if totalRounds > 10:
        seconds.pop(0)
        totalRounds -= 1
    totalSeconds = sum(seconds)
    start = now()
    pathList = sim([(balances[baseCoin], baseCoin)])
    if pathList is None: return False
    freq = 60 / (totalSeconds / totalRounds)
    optPath, optRetn = optimum(pathList)
    lenTrans = len(optPath) - 1
    pathStr = ' \u2192 '.join([YELLOW + state[1] + RESET for state in optPath])
    print(strTime() + ' $' + RED + '{:.4f}'.format(mv) + RESET + '/' + RED + '{:+.2f}'.format(mv / mv0 * 10000 - 10000) + RESET + 'bp [' + '{}'.format(pathStr) + '] ' + CYAN + '{:+.4f}'.  format(optRetn * 10000) + RESET + 'bp (' + YELLOW + '{:.2f}'.format(freq) + RESET + 'rpm)')
    if optRetn > threshold:
        if sys.platform == 'darwin': os.system('afplay /System/Library/Sounds/Ping.aiff')
        for i in range(lenTrans):
            fromCoin = optPath[i][1]
            toCoin = optPath[i + 1][1]
            amount = balances[fromCoin]
            if not order(toCoin, fromCoin, -amount): break
            balances = availableBalances()
        if sys.platform == 'darwin': os.system('afplay /System/Library/Sounds/Tink.aiff')
    mv, balances = calcMv()


@exceptionHandler(timeOut=30, again=False)
def updateConfig(mv0=None):
    print(strTime(), 'Coinbot initializing...')
    global agent, feeRate, start, orderBooks, pairs, balances, mv
    orderBooks = {}
    agent = Poloniex(apikey, secret)
    pairs = np.array(agent.returnTicker())
    feeRate = agent.returnFeeInfo()['takerFee']
    start = now()
    mv, balances = calcMv()
    print(strTime(), '{:<10}:'.format('Base coin'), baseCoin)
    print(strTime(), '{:<10}:'.format('Max length'), maxLen)
    print(strTime(), '{:<10}:'.format('Slippage'), '{:.2f}bp'.format(slippage * 10000))
    print(strTime(), '{:<10}:'.format('Threshold'), '{:.2f}bp'.format(threshold * 10000))
    print(strTime(), '{:<10}:'.format('Value'), '${:.4f}'.format(mv))
    if mv0: print(strTime(), '{:<10}:'.format('Init Value'), '${:.4f}'.format(mv0))
    else: mv0 = mv
    return mv0


if sys.stdout.isatty():
    RED = '\u001b[1m\u001b[38;5;196m'
    GRAY = '\u001b[1m\u001b[38;5;240m'
    GREEN = '\u001b[1m\u001b[38;5;46m'
    YELLOW = '\u001b[1m\u001b[38;5;220m'
    CYAN = '\u001b[1m\u001b[38;5;51m'
    RESET = '\u001b[0m'
else:
    RED = GRAY = GREEN = YELLOW = CYAN = RESET = ''

seconds = []
