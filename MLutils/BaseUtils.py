import numpy as np
from tensorflow import convert_to_tensor

def Xy(df, target, clss, tensor = False):
    if clss:
        temp = df.drop([target], axis=1), df[target] >0
    else:
        temp =  df.drop([target], axis=1), df[target]
    return [convert_to_tensor(i) for i in temp]