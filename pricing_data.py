from sqlite3 import DatabaseError
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def get_data():
    krx = './Data/factor/'
    res = []
    files = os.listdir(krx)
    for file in files:
        res.append(pd.read_csv(krx+file, header=None).values)
    res = np.concatenate(res)
    columns = pd.read_csv(krx+"factor_1.csv", nrows=1).columns  # columns names only appear to be in this file
    data = pd.DataFrame(res, columns=columns)
    data["date"] = pd.to_datetime(data.date, errors="coerce")
    data = data[~data.date.isna()]  # remove wrong dates
    data['ticker'] = data.ticker.apply(lambda x: '0'*(6-len(str(x)))+str(x))
    data = data.replace([np.inf, -np.inf],
                        np.nan).dropna(subset=['target', 'mom4']).fillna(0)
    cols = [i for i in data.columns.tolist() if 'hml_bin_prec' not in i]
    cols = [i for i in cols if 'size_bin_prec' not in i]
    data = data[cols]
    data.set_index('date', inplace=True)
    for col in data.columns[data.dtypes!=np.float64]:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.replace([np.inf, -np.inf],np.nan).dropna()
    return data


def train_test_preprocess(data, train_period, test_period):
    years = data.index.year.unique()
    test_years = years[train_period:len(years)-test_period+1]
    train = {}
    test = {}
    for y in tqdm(test_years, desc='Data Split'):
        temp = data[(data.index.year < y + test_period - 1)
                    & (data.index.year >= y - train_period)]
        temp_test = data[data.index.year == y]
        train[y] = [temp.target, temp.drop(['target'], axis=1)]
        test[y] = [temp_test.target, temp_test.drop(['target'], axis=1)]
    return train, test, test_years


def krx_data(train_period=5, test_period=1):
    data = get_data()
    return train_test_preprocess(data, train_period, test_period)


def data_split():
    filePath = './Data/'  # 폴더 주소 입력
    filePath_new = './Data/factor/'
    file = 'factor.csv'
    rowsize = sum(1 for row in (open(filePath + file, encoding='UTF-8')))
    newsize = 33700   # 쪼개고 싶은 행수 수준으로 입력. 이정도 행수는 200mb 이하임.
    times = 0
    for i in range(0, rowsize, newsize):
        times += 1   # 폴더 내 파일을 하나씩 점검하면서, 입력한 newsize보다 넘는 행을 쪼개줌
        df = pd.read_csv(filePath + file, nrows=newsize, skiprows=i)
        # 쪼갠 수만큼 _1, _2... _n으로 꼬리를 달아서 파일명이 저장됨
        csv_output = file[:-4] + '_' + str(times) + '.csv'
        df.to_csv(filePath_new + csv_output, index=False, chunksize=rowsize)
