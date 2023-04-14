from alphalens.utils import get_clean_factor_and_forward_returns
import alphalens.performance as perf
import alphalens.utils as utils
from multiprocessing import Queue, Pool
import pandas as pd
import numpy as np
from datas import *

def analy_alphas(alphaPath, close, year):
    alpha = pd.read_csv(alphaPath)

    # 筛选出今年的数据，需与股票收盘日期区间一致
    alpha = alpha[(alpha['date'] >= f'{year}-01-01') & (alpha['date'] <= f'{year+1}-01-01')]

    # 因子矩阵转换为一维数据(alphalens需要的格式)
    alpha = alpha.melt(id_vars=['date'], var_name='asset', value_name='factor' )

    # date列转为日期格式
    alpha['date'] = pd.to_datetime(alpha['date'])
    alpha = alpha[['date', 'asset', 'factor']]

    # 设置二级索引
    alpha = alpha.set_index(['date', 'asset'], drop=True)
    alpha.sort_index(inplace=True)

    factor_data = get_clean_factor_and_forward_returns(alpha, close,quantiles=5)
    mean_quant_ret, std_quantile = perf.mean_return_by_quantile(
        factor_data,
        by_group=False
    )
    mean_quant_rateret = mean_quant_ret.apply(
        utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
    )

    ic = perf.factor_information_coefficient(factor_data)

    result = {
        'return_max': mean_quant_rateret['1D'].iloc[-1] * 10000,
        'return_min': mean_quant_rateret['1D'].iloc[0] * 10000,
        'ic_mean': ic['1D'].mean(),
        'ic_std': ic['1D'].std(),
        'ir': ic['1D'].mean() / ic['1D'].std()
        }
    print(result)
    return result

def generate_year_report(alpha_name, list_assets, year):
    # list_assets, df_assets = get_hs300_stocks(f'{year}-01-01')
    dfs= get_all_date_data(f'{year}-01-01', f'{year+1}-01-01', list_assets)

    df_all = dfs[['date', 'asset', "close"]]
    df_all['date'] = pd.to_datetime(df_all['date'])

    close = df_all.pivot(index='date', columns='asset', values='close')
    lst_a = os.listdir(f'alphas/{alpha_name}/{year}/')

    list_ret = []
    for a in lst_a:        
        try:
            ret = analy_alphas(f'alphas/{alpha_name}/{year}/{a}', close, year)
            ret['name'] = a.split('.')[0]
            list_ret.append(ret)
        except Exception as e:
            print(e)

    df = pd.DataFrame(list_ret)
    df = df.set_index(['name'], drop=True)
    print(df)
    path = 'analysis'
    if not os.path.isdir(path):
        os.makedirs(path)
    df.to_csv(f'{path}/{alpha_name}_{year}_result.csv')

def compare_factor(g_name, start, end):
    list_df = []
    for i in range(start,end):
        df = pd.read_csv(f'analysis/{g_name}_{i}_result.csv')
        df['year'] = i
        list_df.append(df)

    df_all = pd.concat(list_df)

    df_all = df_all[['year', 'name', "return_max"]]
    returns = df_all.pivot(index='name', columns='year', values='return_max')
    ranks = returns.rank(axis=0, method='min', pct=True)
    print(ranks)
    weights = np.array(range(1, 11))
    sum_weights = np.sum(weights)
    ranks['avg'] = (ranks*weights).sum(axis="columns")
    print(ranks)
    ranks['avg_rank'] = ranks['avg'].rank(axis=0, method='min', pct=True)
    print(ranks)
    return ranks

if __name__ == '__main__':
    year = 2013
    alpha_name = 'Alphas101'
    # list_assets, df_assets = get_hs300_stocks(f'{year}-01-01')
    # generate_year_report(alpha_name, list_assets, year)
    compare_factor('Alphas101', 2013, 2023)

    

    

    
    
