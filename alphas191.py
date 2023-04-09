import numpy as np
from numpy import log
from scipy.stats import rankdata
from alphas import Alphas
from datas import *

def Log(sr):
    #自然对数函数
    return np.log(sr)

def Rank(sr):
    #列-升序排序并转化成百分比
    return sr.rank(axis=1, method='min', pct=True)

def Delta(sr,period):
    #period日差分
    return sr.diff(period)

def Delay(sr,period):
    #period阶滞后项
    return sr.shift(period)

def Corr(x,y,window):
    #window日滚动相关系数
    #当一个变量值为常量，另一个变量值可变化时，此时无法计算相关度，使用0 进行填充
    r = x.rolling(window).corr(y).fillna(0)
    #同时将起始 window-1 个窗口赋值为空
    r.iloc[:(window-1), :] = None
    return r

def Cov(x,y,window):
    #window日滚动协方差
    return x.rolling(window).cov(y)

def Sum(sr,window):
    #window日滚动求和
    return sr.rolling(window).sum()

def Prod(sr,window):
    #window日滚动求乘积
    return sr.rolling(window).apply(lambda x: np.prod(x))

def Mean(sr,window):
    #window日滚动求均值
    return sr.rolling(window).mean()

def Std(sr,window):
    #window日滚动求标准差
    return sr.rolling(window).std()

def Tsrank(sr, window):
    #window日序列末尾值的顺位
    return sr.rolling(window).apply(lambda x: rankdata(x)[-1])
               
def Tsmax(sr, window):
    #window日滚动求最大值    
    return sr.rolling(window).max()

def Tsmin(sr, window):
    #window日滚动求最小值    
    return sr.rolling(window).min()

def Sign(sr):
    #符号函数
    return np.sign(sr)

def Max(sr1,sr2):
    return np.maximum(sr1, sr2)

def Min(sr1,sr2):
    return np.minimum(sr1, sr2)

def Rowmax(sr):
    return sr.max(axis=1)

def Rowmin(sr):
    return sr.min(axis=1)

def Sma(sr,n,m):
    #sma均值
    return sr.ewm(alpha=m/n, adjust=False).mean()

def Abs(sr):
    #求绝对值
    return sr.abs()

def Sequence(n):
    #生成 1~n 的等差序列
    return np.arange(1,n+1)

def Regbeta(sr,x):
    window = len(x)
    return sr.rolling(window).apply(lambda y: np.polyfit(x, y, deg=1)[0])

def Decaylinear(sr, window):  
    weights = np.array(range(1, window+1))
    sum_weights = np.sum(weights)
    return sr.rolling(window).apply(lambda x: np.sum(weights*x) / sum_weights)

def Lowday(sr,window):
    return sr.rolling(window).apply(lambda x: len(x) - x.values.argmin())

def Highday(sr,window):
    return sr.rolling(window).apply(lambda x: len(x) - x.values.argmax())

def Wma(sr,window):
    weights = np.array(range(window-1,-1, -1))
    weights = np.power(0.9,weights)
    sum_weights = np.sum(weights)

    return sr.rolling(window).apply(lambda x: np.sum(weights*x) / sum_weights)

def Count(cond,window):
    return cond.rolling(window).apply(lambda x: x.sum())

def Sumif(sr,window,cond):
    sr[~cond] = 0
    return sr.rolling(window).sum()

def Returns(df):
    return df.rolling(2).apply(lambda x: x.iloc[-1] / x.iloc[0]) - 1


class Alphas191(Alphas):
    def __init__(self, df_data):
        self.open = df_data['open'] # 开盘价
        self.high = df_data['high'] # 最高价
        self.low = df_data['low'] # 最低价
        self.close = df_data['close'] # 收盘价
        self.volume = df_data['volume'] # 成交量
        self.returns = Returns(df_data['close']) # 日收益率
        self.vwap = df_data['vwap']  # 成交均价
        self.close_prev = df_data['close'].shift(1)#前一天收盘价        
        self.amount = df_data['amount']#交易额
        
        self.benchmark_open = df_data['benchmark_open']#指数开盘价series
        self.benchmark_close = df_data['benchmark_close']#指数收盘价series
        # self.value = df_data['value']#公司总市值

    def alpha001(self): #平均1751个数据
        ##### (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))#### 
        return (-1 * Corr(Rank(Delta(log(self.volume), 1)), Rank(((self.close - self.open) / self.open)), 6))
    
    def alpha002(self): #1783
        ##### -1 * delta((((close-low)-(high-close))/(high-low)),1))####
        return -1*Delta((((self.close-self.low)-(self.high-self.close))/(self.high-self.low)),1) 
    
    def alpha003(self): 
        ##### SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6) ####
        cond1 = (self.close == Delay(self.close,1))
        cond2 = (self.close > Delay(self.close,1))
        cond3 = (self.close < Delay(self.close,1))
        part = self.close.copy(deep=True)
        part.loc[:, :] = None
        part[cond1] = 0
        part[cond2] = self.close - Min(self.low,Delay(self.close,1))
        part[cond3] = self.close - Max(self.high,Delay(self.close,1))
        return Sum(part, 6)
    
    def alpha004(self):  
        #####((((SUM(CLOSE, 8) / 8) + STD(CLOSE, 8)) < (SUM(CLOSE, 2) / 2)) ? (-1 * 1) : (((SUM(CLOSE, 2) / 2) <((SUM(CLOSE, 8) / 8) - STD(CLOSE, 8))) ? 1 : (((1 < (VOLUME / MEAN(VOLUME,20))) || ((VOLUME /MEAN(VOLUME,20)) == 1)) ? 1 : (-1 * 1))))
        cond1 = ((Sum(self.close, 8)/8 + Std(self.close, 8)) < Sum(self.close, 2)/2)
        cond2 = ((Sum(self.close, 8)/8 + Std(self.close, 8)) > Sum(self.close, 2)/2)
        cond3 = ((Sum(self.close, 8)/8 + Std(self.close, 8)) == Sum(self.close, 2)/2)
        cond4 = (self.volume/Mean(self.volume, 20) >= 1)
        part = self.close.copy(deep=True) 
        part.loc[:, :] = None
        part[cond1] = -1
        part[cond2] = 1
        part[cond3] = -1
        part[cond3 & cond4] = 1
        
        return part
    
    def alpha005(self): #1447
        ####(-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))###
        return -1*Tsmax(Corr(Tsrank(self.volume, 5),Tsrank(self.high, 5),5), 3)
    
    def alpha006(self): #1779
        ####(RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4)))* -1)### 
        return -1*Rank(Sign(Delta(((self.open * 0.85) + (self.high * 0.15)), 4)))
    
    def alpha007(self): #1782
        ####((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))###
        return ((Rank(Tsmax((self.vwap - self.close), 3)) + Rank(Tsmin((self.vwap - self.close), 3))) * Rank(Delta(self.volume, 3)))
    
    def alpha008(self): #1779
        ####RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)###    
        return Rank(Delta(((((self.high + self.low) / 2) * 0.2) + (self.vwap * 0.8)), 4) * -1)
    
    def alpha009(self): #1790
        ####SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)###  
        return Sma(((self.high+self.low)/2-(Delay(self.high,1)+Delay(self.low,1))/2)*(self.high-self.low)/self.volume,7,2)
    
    def alpha010(self):    
        ####(RANK(MAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2),5))###
        cond = (self.returns < 0)
        part = self.returns.copy(deep=True) 
        part.loc[:, :] = None
        part[cond] = Std(self.returns, 20)
        part[~cond] = self.close
        part = part**2
        
        return Rank(Tsmax(part, 5))
    
    def alpha011(self): #1782
        ####SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,6)###   
        return Sum(((self.close-self.low)-(self.high-self.close))/(self.high-self.low)*self.volume,6)
    
    def alpha012(self): #1779
        ####(RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))###   
        return (Rank((self.open - (Sum(self.vwap, 10) / 10)))) * (-1 * (Rank(Abs((self.close - self.vwap)))))
    
    def alpha013(self): #1790
        ####(((HIGH * LOW)^0.5) - VWAP)###
        return (((self.high * self.low)**0.5) - self.vwap)
    
    def alpha014(self): #1776
        ####CLOSE-DELAY(CLOSE,5)###
        return self.close-Delay(self.close,5)
    
    def alpha015(self): #1790
        ####OPEN/DELAY(CLOSE,1)-1###
        return self.open/Delay(self.close,1)-1
    
    def alpha016(self): #1736   
        ####(-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5))###
        return (-1 * Tsmax(Rank(Corr(Rank(self.volume), Rank(self.vwap), 5)), 5))
        
    def alpha017(self): #1776   
        ####RANK((VWAP - MAX(VWAP, 15)))^DELTA(CLOSE, 5)###
        return Rank((self.vwap - Tsmax(self.vwap, 15)))**Delta(self.close, 5)
    
    def alpha018(self): #1776   
        ####CLOSE/DELAY(CLOSE,5)###
        return self.close/Delay(self.close,5)  
    
    def alpha019(self):  
        ####(CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE))###
        cond1 = (self.close < Delay(self.close,5))
        cond2 = (self.close == Delay(self.close,5))
        cond3 = (self.close > Delay(self.close,5))
        part = self.close.copy(deep=True) 
        part.loc[:, :] = None
        part[cond1] = (self.close-Delay(self.close,5))/Delay(self.close,5)
        part[cond2] = 0
        part[cond3] = (self.close-Delay(self.close,5))/self.close
        
        return part
       
    def alpha020(self): #1773      
        ####(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100###
        return (self.close-Delay(self.close,6))/Delay(self.close,6)*100
    
    def alpha021(self):  #reg？
        ####REGBETA(MEAN(CLOSE,6),SEQUENCE(6))###        
        return Regbeta(Mean(self.close,6), Sequence(6))
    
    def alpha022(self): #1736  
        ####SMA(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)###
        return Sma(((self.close-Mean(self.close,6))/Mean(self.close,6)-Delay((self.close-Mean(self.close,6))/Mean(self.close,6),3)),12,1)
     
    def alpha023(self):  
        ####SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1) / (SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1) + SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1))*100###
        cond = (self.close > Delay(self.close,1))
        part1 = self.close.copy(deep=True) 
        part1.loc[:, :] = None
        part1[cond] = Std(self.close,20)
        part1[~cond] = 0
        part2 = self.close.copy(deep=True) 
        part2.loc[:, :] = None
        part2[~cond] = Std(self.close,20)
        part2[cond] = 0
        
        return 100*Sma(part1,20,1)/(Sma(part1,20,1) + Sma(part2,20,1))
        
    def alpha024(self): #1776  
        ####SMA(CLOSE-DELAY(CLOSE,5),5,1)###
        return Sma(self.close-Delay(self.close,5),5,1)
    
    def alpha025(self):  #886  
        ####((-1 * RANK((DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR((VOLUME / MEAN(VOLUME,20)), 9)))))) * (1 + RANK(SUM(RET, 250))))###
        return ((-1 * Rank((Delta(self.close, 7) * (1 - Rank(Decaylinear((self.volume / Mean(self.volume,20)), 9)))))) * (1 + Rank(Sum(self.returns, 250))))
    
    def alpha026(self):   
        ####((((SUM(CLOSE, 7) / 7) - CLOSE)) + ((CORR(VWAP, DELAY(CLOSE, 5), 230))))###
        return ((((Sum(self.close, 7) / 7) - self.close)) + ((Corr(self.vwap, Delay(self.close, 5), 230))))
    
    def alpha027(self):  
        ####WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)###
        A = (self.close-Delay(self.close,3))/Delay(self.close,3)*100+(self.close-Delay(self.close,6))/Delay(self.close,6)*100
        return Wma(A, 12)
    
    def alpha028(self):   #1728 
        ####3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)-2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(MAX(HIGH,9)-TSMAX(LOW,9))*100,3,1),3,1)###
        return 3*Sma((self.close-Tsmin(self.low,9))/(Tsmax(self.high,9)-Tsmin(self.low,9))*100,3,1)-2*Sma(Sma((self.close-Tsmin(self.low,9))/(Tsmax(self.high,9)-Tsmax(self.low,9))*100,3,1),3,1)
    
    def alpha029(self):   #1773 
        ####(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME###
        return (self.close-Delay(self.close,6))/Delay(self.close,6)*self.volume
    
    def alpha030(self):  #reg？
        ####WMA((REGRESI(CLOSE/DELAY(CLOSE)-1,MKT,SMB,HML， 60))^2,20)###
        return 0
    
    def alpha031(self):   #1714
        ####(CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100###
        return (self.close-Mean(self.close,12))/Mean(self.close,12)*100
    
    def alpha032(self):   #1505
        ####(-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))###
        return (-1 * Sum(Rank(Corr(Rank(self.high), Rank(self.volume), 3)), 3))
    
    def alpha033(self):   #904  数据量较少
        ####((((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)) * RANK(((SUM(RET, 240) - SUM(RET, 20)) / 220))) *TSRANK(VOLUME, 5))###
        return ((((-1 * Tsmin(self.low, 5)) + Delay(Tsmin(self.low, 5), 5)) * Rank(((Sum(self.returns, 240) - Sum(self.returns, 20)) / 220))) *Tsrank(self.volume, 5))
    
    def alpha034(self):   #1714
        ####MEAN(CLOSE,12)/CLOSE###
        return Mean(self.close,12)/self.close
    
    def alpha035(self):   #1790    (OPEN * 0.65) +(OPEN *0.35)有问题
        ####(MIN(RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)), RANK(DECAYLINEAR(CORR((VOLUME), ((OPEN * 0.65) +(OPEN *0.35)), 17),7))) * -1)###
        return (Min(Rank(Decaylinear(Delta(self.open, 1), 15)), Rank(Decaylinear(Corr((self.volume), ((self.open * 0.65) +(self.open *0.35)), 17),7))) * -1)
     
    def alpha036(self):   #1714
        ####RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP),6), 2))###
        return Rank(Sum(Corr(Rank(self.volume), Rank(self.vwap),6 ), 2))
    
    def alpha037(self):   #1713
        ####(-1 * RANK(((SUM(OPEN, 5) * SUM(RET, 5)) - DELAY((SUM(OPEN, 5) * SUM(RET, 5)), 10))))###
        return (-1 * Rank(((Sum(self.open, 5) * Sum(self.returns, 5)) - Delay((Sum(self.open, 5) * Sum(self.returns, 5)), 10))))
    
    def alpha038(self):  
        ####(((SUM(HIGH, 20) / 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0)
        cond = ((Sum(self.high, 20) / 20) < self.high)
        part = self.close.copy(deep=True)
        part.loc[:, :] = None
        part[cond] = -1 * Delta(self.high, 2)
        part[~cond] = 0
        
        return part
    
    def alpha039(self):   #1666
        ####((RANK(DECAYLINEAR(DELTA((CLOSE), 2),8)) - RANK(DECAYLINEAR(CORR(((VWAP * 0.3) + (OPEN * 0.7)),SUM(MEAN(VOLUME,180), 37), 14), 12))) * -1)###
        return ((Rank(Decaylinear(Delta((self.close), 2),8)) - Rank(Decaylinear(Corr(((self.vwap * 0.3) + (self.open * 0.7)),Sum(Mean(self.volume,180), 37), 14), 12))) * -1)
    
    def alpha040(self):  
        ####SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26)/SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26)*100###
        cond = (self.close > Delay(self.close,1))
        part1 = self.close.copy(deep=True)
        part1.loc[:, :] = None
        part1[cond] = self.volume
        part1[~cond] = 0
        part2 = self.close.copy(deep=True)
        part2.loc[:, :] = None
        part2[~cond] = self.volume
        part2[cond] = 0

        return Sum(part1,26)/Sum(part2,26)*100
    
    def alpha041(self):   #1782
        ####(RANK(MAX(DELTA((VWAP), 3), 5))* -1)###
        return (Rank(Tsmax(Delta((self.vwap), 3), 5))* -1)
    
    def alpha042(self):   #1399  数据量较少
        ####((-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10))###
        return ((-1 * Rank(Std(self.high, 10))) * Corr(self.high, self.volume, 10))
    
    def alpha043(self):  
        ####SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6)###
        cond1 = (self.close > Delay(self.close,1))
        cond2 = (self.close < Delay(self.close,1))
        cond3 = (self.close == Delay(self.close,1))
        part = self.close.copy(deep=True) # pd.Series(np.zeros(self.close.shape))
        part.loc[:, :] = None
        part[cond1] = self.volume
        part[cond2] = -self.volume
        part[cond3] = 0
        
        return Sum(part,6)
    
    def alpha044(self):   #1748
        ####(TSRANK(DECAYLINEAR(CORR(((LOW )), MEAN(VOLUME,10), 7), 6),4) + TSRANK(DECAYLINEAR(DELTA((VWAP),3), 10), 15))###
        return (Tsrank(Decaylinear(Corr(((self.low)), Mean(self.volume,10), 7), 6),4) + Tsrank(Decaylinear(Delta((self.vwap),3), 10), 15))
    
    def alpha045(self):   #1070  数据量较少
        ####(RANK(DELTA((((CLOSE * 0.6) + (OPEN *0.4))), 1)) * RANK(CORR(VWAP, MEAN(VOLUME,150), 15)))###
        return (Rank(Delta((((self.close * 0.6) + (self.open *0.4))), 1)) * Rank(Corr(self.vwap, Mean(self.volume,150), 15)))
    
    def alpha046(self):   #1630
        ####(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)###
        return (Mean(self.close,3)+Mean(self.close,6)+Mean(self.close,12)+Mean(self.close,24))/(4*self.close)
    
    def alpha047(self):   #1759
        ####SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1)###
        return Sma((Tsmax(self.high,6)-self.close)/(Tsmax(self.high,6)-Tsmin(self.low,6))*100,9,1)
    
    def alpha048(self):   #1657
        ####(-1*((RANK(((SIGN((CLOSE - DELAY(CLOSE, 1))) + SIGN((DELAY(CLOSE, 1) - DELAY(CLOSE, 2)))) + SIGN((DELAY(CLOSE, 2) - DELAY(CLOSE, 3)))))) * SUM(VOLUME, 5)) / SUM(VOLUME, 20))###
        return (-1*((Rank(((Sign((self.close - Delay(self.close, 1))) + Sign((Delay(self.close, 1) - Delay(self.close, 2)))) + Sign((Delay(self.close, 2) - Delay(self.close, 3)))))) * Sum(self.volume, 5)) / Sum(self.volume, 20))
    
    def alpha049(self):  
        ####SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12) / (SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12) + SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
        cond = ((self.high + self.low) > (Delay(self.high,1) + Delay(self.low,1)))
        part1 = self.close.copy(deep=True) # pd.Series(np.zeros(self.close.shape))
        part1.loc[:, :] = None
        part1[cond] = 0
        part1[~cond] = Max(Abs(self.high - Delay(self.high,1)), Abs(self.low - Delay(self.low,1)))
        part2 = self.close.copy(deep=True)
        part2.loc[:, :] = None
        part2[~cond] = 0
        part2[cond] = Max(Abs(self.high - Delay(self.high,1)), Abs(self.low - Delay(self.low,1)))
        
        return Sum(part1, 12) / (Sum(part1, 12) + Sum(part2, 12))
    
    def alpha050(self):  
        ####SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))-SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))###
        cond = ((self.high + self.low) <= (Delay(self.high,1) + Delay(self.low,1)))
        part1 = self.close.copy(deep=True)
        part1.loc[:, :] = None
        part1[cond] = 0
        part1[~cond] = Max(Abs(self.high - Delay(self.high,1)), Abs(self.low - Delay(self.low,1)))
        part2 = self.close.copy(deep=True)
        part2.loc[:, :] = None
        part2[~cond] = 0
        part2[cond] = Max(Abs(self.high - Delay(self.high,1)), Abs(self.low - Delay(self.low,1)))
        
        return (Sum(part1, 12) - Sum(part2, 12)) / (Sum(part1, 12) + Sum(part2, 12)) 

    def alpha051(self):  
        ####SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12) / (SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))###
        cond = ((self.high + self.low) <= (Delay(self.high,1) + Delay(self.low,1)))
        part1 = self.close.copy(deep=True)
        part1.loc[:, :] = None
        part1[cond] = 0
        part1[~cond] = Max(Abs(self.high - Delay(self.high,1)), Abs(self.low - Delay(self.low,1)))
        part2 = self.close.copy(deep=True)
        part2.loc[:, :] = None
        part2[~cond] = 0
        part2[cond] = Max(Abs(self.high - Delay(self.high,1)), Abs(self.low - Delay(self.low,1)))
        
        return Sum(part1, 12) / (Sum(part1, 12) + Sum(part2, 12))
    
    def alpha052(self):   #1611
        ####SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-L),26)*100###
        return Sum(Max(self.high-Delay((self.high+self.low+self.close)/3,1),0),26)/Sum(Max(Delay((self.high+self.low+self.close)/3,1)-self.low, 0),26)*100
    
    def alpha053(self):  
        ####COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100###
        cond = (self.close > Delay(self.close,1))
        return Count(cond, 12) / 12 * 100
    
    def alpha054(self):   #1729
        ####(-1 * RANK((STD(ABS(CLOSE - OPEN)) + (CLOSE - OPEN)) + CORR(CLOSE, OPEN,10)))###
        return (-1 * Rank(((Abs(self.close - self.open)).std() + (self.close - self.open)) + Corr(self.close, self.open,10)))
    
    def alpha055(self):  #公式有问题
        ###SUM(16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1)) & ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2 + ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1))),20)
        A = Abs(self.high - Delay(self.close, 1))
        B = Abs(self.low - Delay(self.close, 1))
        C = Abs(self.high - Delay(self.low, 1))
        cond1 = ((A > B) & (A > C))
        cond2 = ((B > C) & (B > A))
        cond3 = ((C >= A) & (C >= B))
        part0 = 16*(self.close + (self.close - self.open)/2 - Delay(self.open,1))
        part1 = self.close.copy(deep=True)
        part1.loc[:,:] = 0
        part1[cond1] = Abs(self.high - Delay(self.close, 1)) + Abs(self.low - Delay(self.close, 1))/2 + Abs(Delay(self.close, 1)-Delay(self.open, 1))/4
        part1[cond2] = Abs(self.low - Delay(self.close, 1)) + Abs(self.high - Delay(self.close, 1))/2 + Abs(Delay(self.close, 1)-Delay(self.open, 1))/4
        part1[cond3] = Abs(self.high - Delay(self.low, 1)) + Abs(Delay(self.close, 1)-Delay(self.open, 1))/4
        part2=Max(Abs(self.high-Delay(self.close,1)),Abs(self.low-Delay(self.close,1)))
        
        return Sum(part0/part1*part2,20)
    
    def alpha056(self):  
        ####(RANK((OPEN - TSMIN(OPEN, 12))) < RANK((RANK(CORR(SUM(((HIGH + LOW) / 2), 19),SUM(MEAN(VOLUME,40), 19), 13))^5)))###
        A = Rank((self.open - Tsmin(self.open, 12)))
        B = Rank((Rank(Corr(Sum(((self.high + self.low) / 2), 19),Sum(Mean(self.volume,40), 19), 13))**5))
        cond = (A < B)
        part = self.close.copy(deep=True)
        part.loc[:, :] = None
        part[cond] = 1
        part[~cond] = 0
        return part
    
    def alpha057(self):   #1736
        ####SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)###
        return Sma((self.close-Tsmin(self.low,9))/(Tsmax(self.high,9)-Tsmin(self.low,9))*100,3,1)
    
    def alpha058(self):  
        ####COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100###
        cond = (self.close > Delay(self.close,1))

        return Count(cond,20)/20*100
        
    
    def alpha059(self):  
        ####SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),20)###
        cond1 = (self.close == Delay(self.close,1))
        cond2 = (self.close > Delay(self.close,1))
        cond3 = (self.close < Delay(self.close,1))
        part = self.close.copy(deep=True)
        part.loc[:, :] = None
        part[cond1] = 0
        part[cond2] = self.close - Min(self.low,Delay(self.close,1))
        part[cond3] = self.close - Max(self.low,Delay(self.close,1))
        
        return Sum(part, 20)
    
    def alpha060(self):   #1635
        ####SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,20)###
        return Sum(((self.close-self.low)-(self.high-self.close))/(self.high-self.low)*self.volume,20)

    def alpha061(self):   #1790
        ####(MAX(RANK(DECAYLINEAR(DELTA(VWAP, 1), 12)),RANK(DECAYLINEAR(RANK(CORR((LOW),MEAN(VOLUME,80), 8)), 17))) * -1)###
        return (Max(Rank(Decaylinear(Delta(self.vwap, 1), 12)),Rank(Decaylinear(Rank(Corr((self.low),Mean(self.volume,80), 8)), 17))) * -1)
    
    def alpha062(self):   #1479
        ####(-1 * CORR(HIGH, RANK(VOLUME), 5))###
        return (-1 * Corr(self.high, Rank(self.volume), 5))
    
    def alpha063(self):   #1789
        ####SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100###
        return Sma(Max(self.close-Delay(self.close,1),0),6,1)/Sma(Abs(self.close-Delay(self.close,1)),6,1)*100
    
    def alpha064(self):   #1774
        ####(MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 4), 4)),RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(VOLUME,60)), 4), 13), 14))) * -1)###
        return (Max(Rank(Decaylinear(Corr(Rank(self.vwap), Rank(self.volume), 4), 4)),Rank(Decaylinear(Tsmax(Corr(Rank(self.close), Rank(Mean(self.volume,60)), 4), 13), 14))) * -1)
    
    def alpha065(self):   #1759
        ####MEAN(CLOSE,6)/CLOSE###
        return Mean(self.close,6)/self.close
    
    def alpha066(self):   #1759
        ####(CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100###
        return (self.close-Mean(self.close,6))/Mean(self.close,6)*100
    
    def alpha067(self):   #1759
        ####SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100###
        a1 = Sma(Max(self.close-Delay(self.close,1),0),24,1)
        a2 = Sma(Abs(self.close-Delay(self.close,1)),24,1)
        return a1/a2*100
    
    def alpha068(self):   #1790
        ####SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2)###
        return Sma(((self.high+self.low)/2-(Delay(self.high,1)+Delay(self.low,1))/2)*(self.high-self.low)/self.volume,15,2)
    
    def alpha069(self):  
        ####(SUM(DTM,20)>SUM(DBM,20)？ (SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20)： (SUM(DTM,20)=SUM(DBM,20)？0： (SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)))###
        ####DTM (OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1))))
        ####DBM (OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1))))
        cond1 = (self.open <= Delay(self.open,1))
        cond2 = (self.open >= Delay(self.open,1))
        
        DTM = self.close.copy(deep=True)
        DTM.loc[:, :] = None
        DTM[cond1] = 0
        DTM[~cond1] = Max((self.high-self.open),(self.open-Delay(self.open,1)))
        
        DBM = self.close.copy(deep=True)
        DBM.loc[:, :] = None
        DBM[cond2] = 0
        DBM[~cond2] = Max((self.open-self.low),(self.open-Delay(self.open,1)))
        
        cond3 = (Sum(DTM,20) > Sum(DBM,20))
        cond4 = (Sum(DTM,20)== Sum(DBM,20))
        cond5 = (Sum(DTM,20) < Sum(DBM,20))
        part = self.close.copy(deep=True)
        part.loc[:, :] = None
        part[cond3] = (Sum(DTM,20)-Sum(DBM,20))/Sum(DTM,20)
        part[cond4] = 0
        part[cond5] = (Sum(DTM,20)-Sum(DBM,20))/Sum(DBM,20)
        return part
    
    def alpha070(self):   #1759
        ####STD(AMOUNT,6)###
        return Std(self.amount,6)
    
    def alpha071(self):   #1630
        ####(CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100###
        return (self.close-Mean(self.close,24))/Mean(self.close,24)*100
    
    def alpha072(self):   #1759
        ####SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)###
        return Sma((Tsmax(self.high,6)-self.close)/(Tsmax(self.high,6)-Tsmin(self.low,6))*100,15,1)
    
    def alpha073(self):   #1729
        ####((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE), VOLUME, 10), 16), 4), 5) - RANK(DECAYLINEAR(CORR(VWAP, MEAN(VOLUME,30), 4),3))) * -1)###
        return ((Tsrank(Decaylinear(Decaylinear(Corr((self.close), self.volume, 10), 16), 4), 5) - Rank(Decaylinear(Corr(self.vwap, Mean(self.volume,30), 4),3))) * -1) 
    
    def alpha074(self):   #1402
        ####(RANK(CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)), 20), SUM(MEAN(VOLUME,40), 20), 7)) + RANK(CORR(RANK(VWAP), RANK(VOLUME), 6)))###
        return (Rank(Corr(Sum(((self.low * 0.35) + (self.vwap * 0.65)), 20), Sum(Mean(self.volume,40), 20), 7)) + Rank(Corr(Rank(self.vwap), Rank(self.volume), 6)))
    
    def alpha075(self):  
        ####COUNT(CLOSE>OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)/COUNT(BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)###
        return Count(((self.close>self.open)&(self.benchmark_close<self.benchmark_open)),50)/Count((self.benchmark_close<self.benchmark_open),50)
    
    def alpha076(self):   #1650
        ####STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)###
        return Std(Abs((self.close/Delay(self.close,1)-1))/self.volume,20)/Mean(Abs((self.close/Delay(self.close,1)-1))/self.volume,20)
    
    def alpha077(self):   #1797
        #### MIN(RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH) - (VWAP + HIGH)), 20)),RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 3), 6)))###
        return  Min(Rank(Decaylinear(((((self.high + self.low) / 2) + self.high) - (self.vwap + self.high)), 20)),Rank(Decaylinear(Corr(((self.high + self.low) / 2), Mean(self.volume,40), 3), 6)))
       
    def alpha078(self):   #1637
        ####((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))###
        return ((self.high+self.low+self.close)/3-Mean((self.high+self.low+self.close)/3,12))/(0.015*Mean(Abs(self.close-Mean((self.high+self.low+self.close)/3,12)),12))
    
    def alpha079(self):   #1789
        ####SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100###
        return Sma(Max(self.close-Delay(self.close,1),0),12,1)/Sma(Abs(self.close-Delay(self.close,1)),12,1)*100
    
    def alpha080(self):   #1776
        ####(VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100###
        return (self.volume-Delay(self.volume,5))/Delay(self.volume,5)*100
    
    def alpha081(self):   #1797
        ####SMA(VOLUME,21,2)###
        return Sma(self.volume,21,2)
    
    def alpha082(self):   #1759
        ####SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)###
        return Sma((Tsmax(self.high,6)-self.close)/(Tsmax(self.high,6)-Tsmin(self.low,6))*100,20,1)
    
    def alpha083(self):   #1766
        ####(-1 * RANK(COVIANCE(RANK(HIGH), RANK(VOLUME), 5)))###
        return (-1 * Rank(Cov(Rank(self.high), Rank(self.volume), 5)))
    
    def alpha084(self):  
        ####SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),20)###
        cond1 = (self.close > Delay(self.close,1))
        cond2 = (self.close < Delay(self.close,1))
        cond3 = (self.close == Delay(self.close,1))  
        part = self.close.copy(deep=True)
        part.loc[:, :] = None
        part[cond1] = self.volume
        part[cond2] = 0
        part[cond3] = -self.volume 
        return Sum(part, 20)
    
    def alpha085(self):   #1657
        ####(TSRANK((VOLUME / MEAN(VOLUME,20)), 20) * TSRANK((-1 * DELTA(CLOSE, 7)), 8))###
        return (Tsrank((self.volume / Mean(self.volume,20)), 20) * Tsrank((-1 * Delta(self.close, 7)), 8))
    
    def alpha086(self):  
        ####((0.25 < (((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10))) ? (-1 * 1) :(((((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10)) < 0) ?1 : ((-1 * 1) *(CLOSE - DELAY(CLOSE, 1)))))
        A = (((Delay(self.close, 20) - Delay(self.close, 10)) / 10) - ((Delay(self.close, 10) - self.close) / 10))
        cond1 = (A > 0.25)
        cond2 = (A < 0.0)
        cond3 = ((0 <= A) & (A <= 0.25))
        part = self.close.copy(deep=True)
        part.loc[:, :] = None
        part[cond1] = -1
        part[cond2] = 1
        part[cond3] = -1*(self.close - Delay(self.close, 1))
        return part

    def alpha087(self):   #1741
        ####((RANK(DECAYLINEAR(DELTA(VWAP, 4), 7)) + TSRANK(DECAYLINEAR(((((LOW * 0.9) + (LOW * 0.1)) - VWAP) /(OPEN - ((HIGH + LOW) / 2))), 11), 7)) * -1)###
        return ((Rank(Decaylinear(Delta(self.vwap, 4), 7)) + Tsrank(Decaylinear(((((self.low * 0.9) + (self.low * 0.1)) - self.vwap) /(self.open - ((self.high + self.low) / 2))), 11), 7)) * -1)
  
    def alpha088(self):   #1745
        ####(CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100###
        return (self.close-Delay(self.close,20))/Delay(self.close,20)*100
    
    def alpha089(self):   #1797
        ####2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))###
        return 2*(Sma(self.close,13,2)-Sma(self.close,27,2)-Sma(Sma(self.close,13,2)-Sma(self.close,27,2),10,2))
    
    def alpha090(self):   #1745
        ####(RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) * -1)###
        return (Rank(Corr(Rank(self.vwap), Rank(self.volume), 5)) * -1)
    
    def alpha091(self):   #1745
        ####((RANK((CLOSE - MAX(CLOSE, 5)))*RANK(CORR((MEAN(VOLUME,40)), LOW, 5))) * -1)###
        return ((Rank((self.close - Tsmax(self.close, 5)))*Rank(Corr((Mean(self.volume,40)), self.low, 5))) * -1)
    
    def alpha092(self):   #1786
        ####(MAX(RANK(DECAYLINEAR(DELTA(((CLOSE * 0.35) + (VWAP *0.65)), 2), 3)),TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)), CLOSE, 13)), 5), 15)) * -1)###
        return (Max(Rank(Decaylinear(Delta(((self.close * 0.35) + (self.vwap *0.65)), 2), 3)),Tsrank(Decaylinear(Abs(Corr((Mean(self.volume,180)), self.close, 13)), 5), 15)) * -1)
    
    def alpha093(self):  
        ####SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)###
        cond = (self.open >= Delay(self.open,1))
        part = self.close.copy(deep=True)
        part.loc[:, :] = None
        part[cond] = 0
        part[~cond] = Max((self.open-self.low),(self.open-Delay(self.open,1)))
        return Sum(part, 20)
    
    def alpha094(self):  
        ####SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)###
        cond1 = (self.close > Delay(self.close,1))
        cond2 = (self.close < Delay(self.close,1))
        cond3 = (self.close == Delay(self.close,1))
        part = self.close.copy(deep=True)
        part.loc[:, :] = None
        part[cond1] = self.volume
        part[cond2] = -1*self.volume
        part[cond3] = 0
        return Sum(part, 30)
    
    def alpha095(self):   #1657
        ####STD(AMOUNT,20)###
        return Std(self.amount,20)
    
    def alpha096(self):   #1736
        ####SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)###
        return Sma(Sma((self.close-Tsmin(self.low,9))/(Tsmax(self.high,9)-Tsmin(self.low,9))*100,3,1),3,1)
    
    def alpha097(self):   #1729
        ####STD(VOLUME,10)###
        return Std(self.volume,10)
    
    def alpha098(self):  
        ####((((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) < 0.05) || ((DELTA((SUM(CLOSE, 100) / 100), 100) /DELAY(CLOSE, 100)) == 0.05)) ? (-1 * (CLOSE - TSMIN(CLOSE, 100))) : (-1 * DELTA(CLOSE, 3)))###
        cond = (Delta(Sum(self.close,100)/100, 100)/Delay(self.close, 100) <= 0.05)
        part = self.close.copy(deep=True)
        part.loc[:, :] = None
        part[cond] = -1 * (self.close - Tsmin(self.close, 100))
        part[~cond] = -1 * Delta(self.close, 3)
        return part
    
    def alpha099(self):   #1766
        ####(-1 * Rank(Cov(Rank(self.close), Rank(self.volume), 5)))###
        return (-1 * Rank(Cov(Rank(self.close), Rank(self.volume), 5)))
    
    def alpha100(self):   #1657
        ####Std(self.volume,20)###
        return Std(self.volume,20)
    
    def alpha101(self):  
        ###((RANK(CORR(CLOSE, SUM(MEAN(VOLUME,30), 37), 15)) < RANK(CORR(RANK(((HIGH * 0.1) + (VWAP * 0.9))),RANK(VOLUME), 11))) * -1)
        rank1 = Rank(Corr(self.close, Sum(Mean(self.volume,30), 37), 15))
        rank2 = Rank(Corr(Rank(((self.high * 0.1) + (self.vwap * 0.9))),Rank(self.volume), 11))
        cond = (rank1<rank2)
        part = self.close.copy(deep=True)
        part.loc[:, :] = None
        part[cond] = 1
        part[~cond] = 0
        return part
    
    def alpha102(self):   #1790
        ####SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100###
        return Sma(Max(self.volume-Delay(self.volume,1),0),6,1)/Sma(Abs(self.volume-Delay(self.volume,1)),6,1)*100
    
    def alpha103(self):  
        ####((20-LOWDAY(LOW,20))/20)*100###
        return ((20-Lowday(self.low,20))/20)*100
    
    def alpha104(self):   #1657
        ####(-1 * (DELTA(CORR(HIGH, VOLUME, 5), 5) * RANK(STD(CLOSE, 20))))###
        return (-1 * (Delta(Corr(self.high, self.volume, 5), 5) * Rank(Std(self.close, 20))))
    
    def alpha105(self):   #1729
        ####(-1 * CORR(RANK(OPEN), RANK(VOLUME), 10))###
        return (-1 * Corr(Rank(self.open), Rank(self.volume), 10))
    
    def alpha106(self):   #1745
        ####CLOSE-DELAY(CLOSE,20)###
        return self.close-Delay(self.close,20)
    
    def alpha107(self):   #1790
        ####(((-1 * RANK((OPEN - DELAY(HIGH, 1)))) * RANK((OPEN - DELAY(CLOSE, 1)))) * RANK((OPEN - DELAY(LOW, 1))))###
        return (((-1 * Rank((self.open - Delay(self.high, 1)))) * Rank((self.open - Delay(self.close, 1)))) * Rank((self.open - Delay(self.low, 1))))
    
    def alpha108(self):   #1178   
        ####((RANK((HIGH - MIN(HIGH, 2)))^RANK(CORR((VWAP), (MEAN(VOLUME,120)), 6))) * -1)###
        return ((Rank((self.high - Tsmin(self.high, 2)))**Rank(Corr((self.vwap), (Mean(self.volume,120)), 6))) * -1)
    
    def alpha109(self):   #1797
        ####SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)###
        return Sma(self.high-self.low,10,2)/Sma(Sma(self.high-self.low,10,2),10,2)
    
    def alpha110(self):   #1650
        ####SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100###
        return Sum(Max(self.high-Delay(self.close,1),0),20)/Sum(Max(Delay(self.close,1)-self.low,0),20)*100
      
    def alpha111(self):   #1789
        ####SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)###
        return Sma(self.volume*((self.close-self.low)-(self.high-self.close))/(self.high-self.low),11,2)-Sma(self.volume*((self.close-self.low)-(self.high-self.close))/(self.high-self.low),4,2)
    
    def alpha112(self):  
        ####(SUM((CLOSE-DELAY(CLOSE,1)>0? CLOSE-DELAY(CLOSE,1):0),12) - SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))/(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12) + SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))*100     
        cond = (self.close-Delay(self.close,1) > 0)
        part1 = self.close.copy(deep=True)
        part1.loc[:, :] = None
        part1[cond] = self.close-Delay(self.close,1)
        part1[~cond] = 0
        part2 = self.close.copy(deep=True)
        part2.loc[:, :] = None
        part2[~cond] = Abs(self.close-Delay(self.close,1))
        part2[cond] = 0
        return (Sum(part1,12) - Sum(part2,12))/(Sum(part1,12) + Sum(part2,12))*100
    
    def alpha113(self):   #1587
        ####(-1 * ((RANK((SUM(DELAY(CLOSE, 5), 20) / 20)) * CORR(CLOSE, VOLUME, 2)) * RANK(CORR(SUM(CLOSE, 5),SUM(CLOSE, 20), 2))))###
        return (-1 * ((Rank((Sum(Delay(self.close, 5), 20) / 20)) * Corr(self.close, self.volume, 2)) * Rank(Corr(Sum(self.close, 5),Sum(self.close, 20), 2))))
    
    def alpha114(self):   #1751
        ####((RANK(DELAY(((HIGH - LOW) / (SUM(CLOSE, 5) / 5)), 2)) * RANK(RANK(VOLUME))) / (((HIGH - LOW) /(SUM(CLOSE, 5) / 5)) / (VWAP - CLOSE)))###
        return ((Rank(Delay(((self.high - self.low) / (Sum(self.close, 5) / 5)), 2)) * Rank(Rank(self.volume))) / (((self.high - self.low) /(Sum(self.close, 5) / 5)) / (self.vwap - self.close)))
    
    def alpha115(self):   #1527
        ####(RANK(CORR(((HIGH * 0.9) + (CLOSE * 0.1)), MEAN(VOLUME,30), 10))^RANK(CORR(TSRANK(((HIGH + LOW) /2), 4), TSRANK(VOLUME, 10), 7)))###
        return (Rank(Corr(((self.high * 0.9) + (self.close * 0.1)), Mean(self.volume,30), 10))**Rank(Corr(Tsrank(((self.high + self.low) /2), 4), Tsrank(self.volume, 10), 7)))
    
    def alpha116(self):  
        ####REGBETA(CLOSE,SEQUENCE,20)###        
        return Regbeta(self.close, Sequence(20))
    
    def alpha117(self):   #1786
        ####((TSRANK(VOLUME, 32) * (1 - TSRANK(((CLOSE + HIGH) - LOW), 16))) * (1 - TSRANK(RET, 32)))###
        return ((Tsrank(self.volume, 32) * (1 - Tsrank(((self.close + self.high) - self.low), 16))) * (1 - Tsrank(self.returns, 32)))
    
    def alpha118(self):   #1657
        ####SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100###
        return Sum(self.high-self.open,20)/Sum(self.open-self.low,20)*100
    
    def alpha119(self):   #1626
        ####(RANK(DECAYLINEAR(CORR(VWAP, SUM(MEAN(VOLUME,5), 26), 5), 7)) - RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN), RANK(MEAN(VOLUME,15)), 21), 9), 7), 8)))###
        return (Rank(Decaylinear(Corr(self.vwap, Sum(Mean(self.volume,5), 26), 5), 7)) - Rank(Decaylinear(Tsrank(Tsmin(Corr(Rank(self.open), Rank(Mean(self.volume,15)), 21), 9), 7), 8)))
    
    def alpha120(self):   #1797
        ####(RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE)))###
        return (Rank((self.vwap - self.close)) / Rank((self.vwap + self.close)))
    
    def alpha121(self):   #972   数据量较少
        ####((RANK((VWAP - MIN(VWAP, 12)))^TSRANK(CORR(TSRANK(VWAP, 20), TSRANK(MEAN(VOLUME,60), 2), 18), 3)) *-1)###
        return ((Rank((self.vwap - Tsmin(self.vwap, 12)))**Tsrank(Corr(Tsrank(self.vwap, 20), Tsrank(Mean(self.volume,60), 2), 18), 3)) *-1)
    
    def alpha122(self):   #1790
        ####(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))/DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)###
        return (Sma(Sma(Sma(Log(self.close),13,2),13,2),13,2)-Delay(Sma(Sma(Sma(Log(self.close),13,2),13,2),13,2),1))/Delay(Sma(Sma(Sma(Log(self.close),13,2),13,2),13,2),1)
    
    def alpha123(self):  
        ####((RANK(CORR(SUM(((HIGH + LOW) / 2), 20), SUM(MEAN(VOLUME,60), 20), 9)) < RANK(CORR(LOW, VOLUME,6))) * -1)###
        A = Rank(Corr(Sum(((self.high + self.low) / 2), 20), Sum(Mean(self.volume,60), 20), 9))
        B = Rank(Corr(self.low, self.volume,6))
        cond = (A < B)
        part = self.close.copy(deep=True)
        part.loc[:, :] = None
        part[cond] = -1
        part[~cond] = 0
        return part
    
    def alpha124(self):   #1592
        ####(CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)),2)###
        return (self.close - self.vwap) / Decaylinear(Rank(Tsmax(self.close, 30)),2)
     
    def alpha125(self):   #1678
        ####(RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME,80),17), 20)) / RANK(DECAYLINEAR(DELTA(((CLOSE * 0.5) + (VWAP * 0.5)), 3), 16)))###
        return (Rank(Decaylinear(Corr((self.vwap), Mean(self.volume,80),17), 20)) / Rank(Decaylinear(Delta(((self.close * 0.5) + (self.vwap * 0.5)), 3), 16)))
    
    def alpha126(self):   #1797
        ####(CLOSE+HIGH+LOW)/3###
        return (self.close+self.high+self.low)/3
    
    def alpha127(self):  #公式有问题，我们假设mean周期为12
        ####(MEAN((100*(CLOSE-MAX(CLOSE,12))/(MAX(CLOSE,12)))^2),12)^(1/2)###
        return (Mean((100*(self.close-Tsmax(self.close,12))/(Tsmax(self.close,12)))**2,12))**(1/2)
    
    def alpha128(self):  
        #### 100-(100/(1+SUM(((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)/SUM(((HIGH+LOW+CLOSE)/3<DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)))
        A = (self.high+self.low+self.close)/3
        cond = (A > Delay(A,1))        
        part1 = self.close.copy(deep=True)
        part1.loc[:, :] = None
        part1[cond] = A*self.volume
        part1[~cond] = 0
        part2 = self.close.copy(deep=True)
        part2.loc[:, :] = None
        part2[~cond] = A*self.volume
        part2[cond] = 0
        return 100-(100/(1+Sum(part1,14)/Sum(part2,14)))

    def alpha129(self):  
        ####SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)###
        cond = ((self.close-Delay(self.close,1)) < 0)
        part = self.close.copy(deep=True)
        part.loc[:, :] = None
        part[cond] = Abs(self.close-Delay(self.close,1))
        part[~cond] = 0
        return Sum(part, 12)
    
    def alpha130(self):   #1657
        ####(RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 9), 10)) / RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 7),3)))###
        return (Rank(Decaylinear(Corr(((self.high + self.low) / 2), Mean(self.volume,40), 9), 10)) / Rank(Decaylinear(Corr(Rank(self.vwap), Rank(self.volume), 7),3)))
    
    def alpha131(self):   #1030   
        ####(RANK(DELAT(VWAP, 1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50), 18), 18))###
        return (Rank(Delta(self.vwap, 1))**Tsrank(Corr(self.close,Mean(self.volume,50), 18), 18))
       
    def alpha132(self):   #1657
        ####MEAN(AMOUNT,20)###
        return Mean(self.amount,20)
    
    def alpha133(self):  
        ####((20-HIGHDAY(HIGH,20))/20)*100-((20-LOWDAY(LOW,20))/20)*100###
        return ((20-Highday(self.high,20))/20)*100-((20-Lowday(self.low,20))/20)*100
    
    def alpha134(self):   #1760
        ####(CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME###
        return (self.close-Delay(self.close,12))/Delay(self.close,12)*self.volume
    
    def alpha135(self):   #1744
        ####SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)###
        return Sma(Delay(self.close/Delay(self.close,20),1),20,1)
    
    def alpha136(self):   #1729
        ####((-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10))###
        return ((-1 * Rank(Delta(self.returns, 3))) * Corr(self.open, self.volume, 10))
    
    def alpha137(self):  
        ####16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1)) & ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1)))
        A = Abs(self.high- Delay(self.close,1))
        B = Abs(self.low - Delay(self.close,1))
        C = Abs(self.high- Delay(self.low,1))
        D = Abs(Delay(self.close,1)-Delay(self.open,1))          
        cond1 = ((A>B) & (A>C))
        cond2 = ((B>C) & (B>A))
        cond3 = ~cond1 & ~cond2       
        part0 = 16*(self.close + (self.close - self.open)/2 - Delay(self.open,1))
        part1 = self.close.copy(deep=True)
        part1.loc[:, :] = None
        part1[cond1] = A + B/2 + D/4
        part1[cond2] = B + A/2 + D/4
        part1[cond3] = C + D/4  
        part1.replace({0: None}, inplace=True)
        return part0/part1*Max(A,B)

    def alpha138(self):   #1448
        ####((RANK(DECAYLINEAR(DELTA((((LOW * 0.7) + (VWAP *0.3))), 3), 20)) - TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8), TSRANK(MEAN(VOLUME,60), 17), 5), 19), 16), 7)) * -1)###
        return ((Rank(Decaylinear(Delta((((self.low * 0.7) + (self.vwap *0.3))), 3), 20)) - Tsrank(Decaylinear(Tsrank(Corr(Tsrank(self.low, 8), Tsrank(Mean(self.volume,60), 17), 5), 19), 16), 7)) * -1)
    
    def alpha139(self):   #1729
        ####(-1 * CORR(OPEN, VOLUME, 10))###
        return (-1 * Corr(self.open, self.volume, 10))
    
    def alpha140(self):   #1797
        ####MIN(RANK(DECAYLINEAR(((RANK(OPEN) + RANK(LOW)) - (RANK(HIGH) + RANK(CLOSE))), 8)), TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE, 8), TSRANK(MEAN(VOLUME,60), 20), 8), 7), 3))###
        return Min(Rank(Decaylinear(((Rank(self.open) + Rank(self.low)) - (Rank(self.high) + Rank(self.close))), 8)), Tsrank(Decaylinear(Corr(Tsrank(self.close, 8), Tsrank(Mean(self.volume,60), 20), 8), 7), 3))
    
    def alpha141(self):   #1637
        ####(RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME,15)), 9))* -1)###
        return (Rank(Corr(Rank(self.high), Rank(Mean(self.volume,15)), 9))* -1)
    
    def alpha142(self):   #1657
        ####(((-1 * RANK(TSRANK(CLOSE, 10))) * RANK(DELTA(DELTA(CLOSE, 1), 1))) * RANK(TSRANK((VOLUME/MEAN(VOLUME,20)), 5)))###
        return (((-1 * Rank(Tsrank(self.close, 10))) * Rank(Delta(Delta(self.close, 1), 1))) * Rank(Tsrank((self.volume/Mean(self.volume,20)), 5)))
    
    def alpha143(self):   # what fuck
        ####CLOSE>DELAY(CLOSE,1)?(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*SELF:SELF###

        return 0
    
    def alpha144(self):  
        ####SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE,1),20)###
        cond = (self.close<Delay(self.close,1))
        part1 = Abs(self.close/Delay(self.close,1)-1)/self.amount
        return Sumif(part1,20,cond)/Count(cond,20)
    
    def alpha145(self):   #1617
        ####(MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100###
        return (Mean(self.volume,9)-Mean(self.volume,26))/Mean(self.volume,12)*100
    
    def alpha146(self):   #1650  
        ####MEAN((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2),20)*((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2))/SMA(((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2)))^2,61,2)###
        return Mean((self.close-Delay(self.close,1))/Delay(self.close,1)-Sma((self.close-Delay(self.close,1))/Delay(self.close,1),61,2),20)*((self.close-Delay(self.close,1))/Delay(self.close,1)-Sma((self.close-Delay(self.close,1))/Delay(self.close,1),61,2))/Sma(((self.close-Delay(self.close,1))/Delay(self.close,1)-((self.close-Delay(self.close,1))/Delay(self.close,1)-Sma((self.close-Delay(self.close,1))/Delay(self.close,1),61,2)))**2,61,2)

    def alpha147(self):  
        ####REGBETA(MEAN(CLOSE,12),SEQUENCE(12))###
        return Regbeta(Mean(self.close, 12), Sequence(12))
    
    def alpha148(self):  
        ####((RANK(CORR((OPEN), SUM(MEAN(VOLUME,60), 9), 6)) < RANK((OPEN - TSMIN(OPEN, 14)))) * -1)###
        cond = (Rank(Corr((self.open), Sum(Mean(self.volume,60), 9), 6)) < Rank((self.open - Tsmin(self.open, 14))))
        part = self.close.copy(deep=True)
        part.loc[:, :] = None
        part[cond] = -1
        part[~cond] = 0
        return part
    
    def alpha149(self):  
        ####REGBETA(FILTER(CLOSE/DELAY(CLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),FILTER(BANCHMARKINDEXCLOSE/DELAY(BANCHMARKINDEXCLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),252)
        return 0
    
    def alpha150(self):   #1797
        ####(CLOSE+HIGH+LOW)/3*VOLUME###
        return (self.close+self.high+self.low)/3*self.volume
    
    def alpha151(self):   #1745
        ####SMA(CLOSE-DELAY(CLOSE,20),20,1)###
        return Sma(self.close-Delay(self.close,20),20,1)
    
    def alpha152(self):   #1559
        ####SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26),9,1)###
        return Sma(Mean(Delay(Sma(Delay(self.close/Delay(self.close,9),1),9,1),1),12)-Mean(Delay(Sma(Delay(self.close/Delay(self.close,9),1),9,1),1),26),9,1)
    
    def alpha153(self):   #1630
        ####(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4###
        return (Mean(self.close,3)+Mean(self.close,6)+Mean(self.close,12)+Mean(self.close,24))/4
    
    def alpha154(self):  
        ####(((VWAP - MIN(VWAP, 16))) < (CORR(VWAP, MEAN(VOLUME,180), 18)))###
        cond = (((self.vwap - Tsmin(self.vwap, 16))) < (Corr(self.vwap, Mean(self.volume,180), 18)))
        part = self.close.copy(deep=True)
        part.loc[:, :] = None
        part[cond] = 1
        part[~cond] = 0
        return part
    
    def alpha155(self):   #1797
        ####SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)###
        return Sma(self.volume,13,2)-Sma(self.volume,27,2)-Sma(Sma(self.volume,13,2)-Sma(self.volume,27,2),10,2)
    
    def alpha156(self):   #1776
        ####(MAX(RANK(DECAYLINEAR(DELTA(VWAP, 5), 3)), RANK(DECAYLINEAR(((DELTA(((OPEN * 0.15) + (LOW *0.85)),2) / ((OPEN * 0.15) + (LOW * 0.85))) * -1), 3))) * -1)###
        return (Max(Rank(Decaylinear(Delta(self.vwap, 5), 3)), Rank(Decaylinear(((Delta(((self.open * 0.15) + (self.low *0.85)),2) / ((self.open * 0.15) + (self.low * 0.85))) * -1), 3))) * -1)
    
    def alpha157(self):   #1764
        ####(MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1 * RANK(DELTA((CLOSE - 1), 5))))), 2), 1)))), 1), 5) + TSRANK(DELAY((-1 * RET), 6), 5))###
        return (Tsmin(Prod(Rank(Rank(Log(Sum(Tsmin(Rank(Rank((-1 * Rank(Delta((self.close - 1), 5))))), 2), 1)))), 1), 5) + Tsrank(Delay((-1 * self.returns), 6), 5))
    
    def alpha158(self):   #1797
        ####((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE###
        return ((self.high-Sma(self.close,15,2))-(self.low-Sma(self.close,15,2)))/self.close
    
    def alpha159(self):   #1630
        ####((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6)*12*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),12))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),12)*6*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),24))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),24)*6*24)*100/(6*12+6*24+12*24)###
        return ((self.close-Sum(Min(self.low,Delay(self.close,1)),6))/Sum(Max(self.high,Delay(self.close,1))-Min(self.low,Delay(self.close,1)),6)*12*24+(self.close-Sum(Min(self.low,Delay(self.close,1)),12))/Sum(Max(self.high,Delay(self.close,1))-Min(self.low,Delay(self.close,1)),12)*6*24+(self.close-Sum(Min(self.low,Delay(self.close,1)),24))/Sum(Max(self.high,Delay(self.close,1))-Min(self.low,Delay(self.close,1)),24)*6*24)*100/(6*12+6*24+12*24)
    
    def alpha160(self):  
        ####SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)###
        cond = (self.close<=Delay(self.close,1))
        part = self.close.copy(deep=True)
        part.loc[:, :] = None
        part[cond] = Std(self.close,20)
        part[~cond] = 0
        return Sma(part, 20, 1)
    
    def alpha161(self):   #1714
        ####MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)###
        return Mean(Max(Max((self.high-self.low),Abs(Delay(self.close,1)-self.high)),Abs(Delay(self.close,1)-self.low)),12)
    
    def alpha162(self):   #1789
        ####(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))/(MAX(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12)-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))###
        return (Sma(Max(self.close-Delay(self.close,1),0),12,1)/Sma(Abs(self.close-Delay(self.close,1)),12,1)*100-Tsmin(Sma(Max(self.close-Delay(self.close,1),0),12,1)/Sma(Abs(self.close-Delay(self.close,1)),12,1)*100,12))/(Sma(Sma(Max(self.close-Delay(self.close,1),0),12,1)/Sma(Abs(self.close-Delay(self.close,1)),12,1)*100,12,1)-Tsmin(Sma(Max(self.close-Delay(self.close,1),0),12,1)/Sma(Abs(self.close-Delay(self.close,1)),12,1)*100,12))
    
    def alpha163(self):   #1657
        ####RANK(((((-1 * RET) * MEAN(VOLUME,20)) * VWAP) * (HIGH - CLOSE)))###
        return Rank(((((-1 * self.returns) * Mean(self.volume,20)) * self.vwap) * (self.high - self.close)))
    
    def alpha164(self):  
        ####SMA(( ((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1) - MIN( ((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1) ,12) )/(HIGH-LOW)*100,13,2)###
        cond = (self.close>Delay(self.close,1))
        part = self.close.copy(deep=True)
        part.loc[:, :] = None
        part[cond] = 1/(self.close-Delay(self.close,1))
        part[~cond] = 1

        # 部分无交易或涨停跌停情况下，HIGH=LOW, 此时会有除零问题，使用空值解决
        part2 = self.high-self.low
        part2.replace({0: None}, inplace=True)

        return Sma((part - Tsmin(part,12))/(part2)*100, 13, 2)
    
    def alpha165(self):  # rowmax
        ####MAX(SUMAC(CLOSE-MEAN(CLOSE,48)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,48)))/STD(CLOSE,48)###
        p1 = Rowmax(Sum(self.close-Mean(self.close,48), 48))
        p2 = Rowmin(Sum(self.close-Mean(self.close,48), 48))
        p3 = Std(self.close,48)
        return -1*(1/p3.div(p2, axis = 0)).sub(p1, axis=0)
    
    def alpha166(self):  #公式有问题
        ####-20* ( 20-1 ) ^1.5*SUM(CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20),20)/((20-1)*(20-2)(SUM((CLOSE/DELAY(CLOSE,1),20)^2,20))^1.5)
        p1 = -20* ( 20-1 )**1.5*Sum(self.close/Delay(self.close,1)-1-Mean(self.close/Delay(self.close,1)-1,20),20)
        p2 = ((20-1)*(20-2)*(Sum(Mean(self.close/Delay(self.close,1),20)**2,20))**1.5)
        return p1/p2

    def alpha167(self):  
        ####SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)###
        cond = (self.close > Delay(self.close,1))
        part = self.close.copy(deep=True)
        part.loc[:, :] = None
        part[cond] = self.close-Delay(self.close,1)
        part[~cond] = 0
        return Sum(part,12)
    
    def alpha168(self):   #1657
        ####(-1*VOLUME/MEAN(VOLUME,20))###
        return (-1*self.volume/Mean(self.volume,20))
    
    def alpha169(self):   #1610
        ####SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),26),10,1)###
        return Sma(Mean(Delay(Sma(self.close-Delay(self.close,1),9,1),1),12)-Mean(Delay(Sma(self.close-Delay(self.close,1),9,1),1),26),10,1)
    
    def alpha170(self):   #1657
        ####((((RANK((1 / CLOSE)) * VOLUME) / MEAN(VOLUME,20)) * ((HIGH * RANK((HIGH - CLOSE))) / (SUM(HIGH, 5) /5))) - RANK((VWAP - DELAY(VWAP, 5))))###
        return ((((Rank((1 / self.close)) * self.volume) / Mean(self.volume,20)) * ((self.high * Rank((self.high - self.close))) / (Sum(self.high, 5) /5))) - Rank((self.vwap - Delay(self.vwap, 5))))
   
    def alpha171(self):   #1789
        ####((-1 * ((LOW - CLOSE) * (OPEN^5))) / ((CLOSE - HIGH) * (CLOSE^5)))###
        return ((-1 * ((self.low - self.close) * (self.open**5))) / ((self.close - self.high) * (self.close**5)))
    
    def alpha172(self):  
        ####MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 &HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 &HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)
        TR = Max(Max(self.high-self.low,Abs(self.high-Delay(self.close,1))),Abs(self.low-Delay(self.close,1)))
        HD = self.high-Delay(self.high,1)
        LD = Delay(self.low,1)-self.low
        cond1 = ((LD>0) & (LD>HD))
        cond2 = ((HD>0) & (HD>LD)) 
        part1 = self.close.copy(deep=True)
        part1.loc[:, :] = None
        part1[cond1] = LD
        part1[~cond1] = 0
        part2 = self.close.copy(deep=True)
        part2.loc[:, :] = None
        part2[cond2] = HD
        part2[~cond2] = 0
        return Mean(Abs(Sum(part1,14)*100/Sum(TR,14)-Sum(part2,14)*100/Sum(TR,14))/(Sum(part1,14)*100/Sum(TR,14)+Sum(part2,14)*100/Sum(TR,14))*100,6)
    
    def alpha173(self):   #1797
        ####3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)###
        return 3*Sma(self.close,13,2)-2*Sma(Sma(self.close,13,2),13,2)+Sma(Sma(Sma(Log(self.close),13,2),13,2),13,2)
    
    def alpha174(self):  
        ####SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)###
        cond = (self.close>Delay(self.close,1))
        part = self.close.copy(deep=True)
        part.loc[:, :] = None
        part[cond] = Std(self.close,20)
        part[~cond] = 0
        return Sma(part,20,1)
    
    def alpha175(self):   #1759
        ####MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)###
        return Mean(Max(Max((self.high-self.low),Abs(Delay(self.close,1)-self.high)),Abs(Delay(self.close,1)-self.low)),6)
    
    def alpha176(self):   #1678
        ####CORR(RANK(((CLOSE - TSMIN(LOW, 12)) / (TSMAX(HIGH, 12) - TSMIN(LOW,12)))), RANK(VOLUME), 6)###
        return Corr(Rank(((self.close - Tsmin(self.low, 12)) / (Tsmax(self.high, 12) - Tsmin(self.low,12)))), Rank(self.volume), 6)
    
    def alpha177(self):  
        ####((20-HIGHDAY(HIGH,20))/20)*100###
        return ((20-Highday(self.high,20))/20)*100
    
    def alpha178(self):   #1790
        ####(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME###
        return (self.close-Delay(self.close,1))/Delay(self.close,1)*self.volume
    
    def alpha179(self):   #1421   数据量较少
        ####(RANK(CORR(VWAP, VOLUME, 4)) *RANK(CORR(RANK(LOW), RANK(MEAN(VOLUME,50)), 12)))###
        return (Rank(Corr(self.vwap, self.volume, 4)) *Rank(Corr(Rank(self.low), Rank(Mean(self.volume,50)), 12)))
    
    def alpha180(self):  #指标有问题
        ####((MEAN(VOLUME,20) < VOLUME) ? ((-1 * TSRANK(ABS(DELTA(CLOSE, 7)), 60)) * SIGN(DELTA(CLOSE, 7)) : (-1 *VOLUME)))
        cond = (Mean(self.volume,20) < self.volume)
        part = self.close.copy(deep=True)
        part.loc[:, :] = None
        part[cond] = (-1 * Tsrank(Abs(Delta(self.close, 7)), 60)) * Sign(Delta(self.close, 7)) 
        part[~cond] = -1 * self.volume
        return part
    
    def alpha181(self):   #1532  公式有问题，假设后面的sum周期为20
        ####SUM(((CLOSE/DELAY(CLOSE,1)-1)-MEAN((CLOSE/DELAY(CLOSE,1)-1),20))-(BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^2,20)/SUM((BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^3)###
        return Sum(((self.close/Delay(self.close,1)-1)-Mean((self.close/Delay(self.close,1)-1),20))-(self.benchmark_close-Mean(self.benchmark_close,20))**2,20)/Sum(((self.benchmark_close-Mean(self.benchmark_close,20))**3),20)
    
    def alpha182(self):  
        ####COUNT((CLOSE>OPEN & BANCHMARKINDEXCLOSE>BANCHMARKINDEXOPEN)OR(CLOSE<OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN),20)/20###
        return Count((((self.close>self.open) & (self.benchmark_close>self.benchmark_open)) | ((self.close<self.open) & (self.benchmark_close<self.benchmark_open))),20)/20
    
    def alpha183(self):  
        ###MAX(SUMAC(CLOSE-MEAN(CLOSE,24)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,24)))/STD(CLOSE,24)###
        p1 = Rowmax(Sum(self.close-Mean(self.close,24), 24))
        p2 = Rowmin(Sum(self.close-Mean(self.close,24), 24))
        p3 = Std(self.close,24)
        return -1*(1/p3.div(p2, axis = 0)).sub(p1, axis=0)
    
    def alpha184(self):   #983   数据量较少
        ####(RANK(CORR(DELAY((OPEN - CLOSE), 1), CLOSE, 200)) + RANK((OPEN - CLOSE)))###
        return (Rank(Corr(Delay((self.open - self.close), 1), self.close, 200)) + Rank((self.open - self.close)))
    
    def alpha185(self):   #1797
        ####RANK((-1 * ((1 - (OPEN / CLOSE))^2)))###
        return Rank((-1 * ((1 - (self.open / self.close))**2)))
    
    def alpha186(self):  
        ####(MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)+DELAY(MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6),6))/2
        TR = Max(Max(self.high-self.low,Abs(self.high-Delay(self.close,1))),Abs(self.low-Delay(self.close,1)))
        HD = self.high-Delay(self.high,1)
        LD = Delay(self.low,1)-self.low
        cond1 = ((LD>0) & (LD>HD))
        cond2 = ((HD>0) & (HD>LD)) 
        part1 = self.close.copy(deep=True)
        part1.loc[:, :] = None
        part1[cond1] = LD
        part1[~cond1] = 0
        part2 = self.close.copy(deep=True)
        part2.loc[:, :] = None
        part2[cond2] = HD
        part2[~cond2] = 0
        return (Mean(Abs(Sum(part1,14)*100/Sum(TR,14)-Sum(part2,14)*100/Sum(TR,14))/(Sum(part1,14)*100/Sum(TR,14)+Sum(part2,14)*100/Sum(TR,14))*100,6)+Delay(Mean(Abs(Sum(part1,14)*100/Sum(TR,14)-Sum(part2,14)*100/Sum(TR,14))/(Sum(part1,14)*100/Sum(TR,14)+Sum(part2,14)*100/Sum(TR,14))*100,6),6))/2
    
    def alpha187(self):  
        ####SUM((OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20)###
        cond = (self.open<=Delay(self.open,1))
        part = self.close.copy(deep=True)
        part.loc[:, :] = None
        part[cond] = 0
        part[~cond] = Max((self.high-self.open),(self.open-Delay(self.open,1)))
        return Sum(part,20) 
    
    def alpha188(self):   #1797
        ####((HIGH-LOW–SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100###
        return ((self.high-self.low-Sma(self.high-self.low,11,2))/Sma(self.high-self.low,11,2))*100
    
    def alpha189(self):   #1721
        ####MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)###
        return Mean(Abs(self.close-Mean(self.close,6)),6)
    
    def alpha190(self):  
        ####LOG((COUNT( CLOSE/DELAY(CLOSE,1)>((CLOSE/DELAY(CLOSE,19))^(1/20)-1) ,20)-1)*(SUMIF((CLOSE/DELAY(CLOSE,1)-((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE,1)<(CLOSE/DELAY(CLOSE,19))^(1/20)-1))/((COUNT((CLOSE/DELAY(CLOSE,1)<(CLOSE/DELAY(CLOSE,19))^(1/20)-1),20))*(SUMIF((CLOSE/DELAY(CLOSE,1)-((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE,1)>(CLOSE/DELAY(CLOSE,19))^(1/20)-1))))
        return 0
    
    def alpha191(self):   #1721
        ####((CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE)###
        return ((Corr(Mean(self.volume,20), self.low, 5) + ((self.high + self.low) / 2)) - self.close)
    
if __name__ == '__main__':
    year = '2019'
    list_assets,df_asserts = get_hs300_stocks(f'{year}-01-01')

    ################ 计算所有 #################    
    Alphas191.generate_alphas(year, list_assets,"sh000300")

    ################ 计算单个 #################
    # ret = Alphas191.generate_alpha_single('alpha170', year, list_assets, "sh000300", True)
    # print(ret)