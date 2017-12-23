import os
import imp
import config
import arbitrage
from config import initValue


if __name__ == '__main__':
    mtimeConfig = 0
    mtimeArbitrage = 0
    mv0 = initValue
    while True:
        mtimeConfigNew = os.path.getmtime('config.py')
        mtimeArbitrageNew = os.path.getmtime('arbitrage.py')
        if (mtimeConfigNew != mtimeConfig) or (mtimeArbitrageNew != mtimeArbitrage):
            mtimeConfig = mtimeConfigNew
            mtimeArbitrage = mtimeArbitrageNew
            imp.reload(config)
            imp.reload(arbitrage)
            from config import *
            from arbitrage import *
            mv0 = updateConfig(mv0)
        trade(mv0)
