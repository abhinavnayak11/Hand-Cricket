import os
import re
import glob
import pandas as pd
import numpy as np


input_path = '/home/abhinavnayak11/workspace/projects/hand-cricket/input'
files = glob.glob(os.path.join(input_path, '*'))
print(len(files))
df = pd.DataFrame()
df['file_path'] = files


def target(x):
    file_name = x.split('/')[-1]
    # print(file_name)
    target = re.findall(r"(\d)", file_name)[0]
    return target
    
    
df['target'] = df['file_path'].apply(lambda x:target(x))
df = df.sample(frac=1, random_state = 42)
df.to_csv('folds/train.csv', index = False)
df = pd.read_csv('folds/train.csv')

print(df.head(20))
