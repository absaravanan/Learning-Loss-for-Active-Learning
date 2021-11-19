import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import ntpath
import os
import random

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

data_info = pd.read_csv(sys.argv[1])
# First column contains the image paths
X = np.asarray(data_info.iloc[:, 0])
# Second column is the labels
y = np.asarray(data_info.iloc[:, 1])

limit = sys.argv[3]
zipped = list(zip(X, y))
random.shuffle(zipped)
zipped = zipped[:int(limit)]
X, y = zip(*zipped)

if sys.argv[2] == "no_stratify":
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)





filename = path_leaf(sys.argv[1])

df_train = pd.DataFrame(list(zip(X_train, y_train)), columns =['url', 'label'])
df_test = pd.DataFrame(list(zip(X_test, y_test)), columns =['url', 'label'])

df_train.to_csv(filename.replace(".csv","_train.csv"))
df_test.to_csv(filename.replace(".csv","_test.csv"))

