# alphas
## 文件说明

1.research.ipynb-----读股票和因子数据，并使用alphalens进行分析

2.backtrader.ipynb-----回测

3.factorycompare.ipynb-----计算历年所有因子的收益并进行百分比排名

alphas.py-----因子计算的基类，支持多进程

alphas101.py-----alphas101因子计算类

alphas191.py-----alphas191因子计算类

analy_alphas-----因子计算与分析

datas.py-----股票数据下载，支持多进程


## 使用步骤

1.运行datas.py下载股票数据

2.运行alphas101.py或alphas191.py计算因子

3.在1.research.ipynb里分析因子

4.在2.backtrader.ipynb回测

5.在factorycompare.ipynb计算历年所有因子的收益并进行百分比排名
