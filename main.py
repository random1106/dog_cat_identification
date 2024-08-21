import os
import torch
import torch.nn as nn
from PIL import Image
# from ImageProcess import ImageProcess
from CNN import CNN
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
# cat: 0; dog: 1
# import the data

def output(X):
    classifier.eval()
    dt = pd.DataFrame(columns = ["id", "label"])
    for i in range(X.shape[0]):
        dt.append({"id": i+1, "label": classifier(X[i])})
    dt.to_csv("output")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier = CNN()
classifier.to(device)
# process = ImageProcess(224, 224)

# X, y = process.import_train("./train")

with open("./train.pkl", "rb") as file:
    X, y = pickle.load(file)

print("trainging_data has been imported")

X = torch.from_numpy(X).permute(0, 3, 1, 2).float()
y = torch.from_numpy(y).long()

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state= 42)

print(f"X_train, X_valid, y_train, y_valid = {X_train.shape, X_valid.shape, y_train.shape, y_valid.shape}")

X_train = X_train.to(device)
X_valid = X_valid.to(device)
y_train = y_train.to(device)
y_valid = y_valid.to(device)

classifier.fit(X_train, X_valid, y_train, y_valid, lr = 0.001)


# with open("./test", "rb") as file:
#     X_test = pickle.load("./test")
# output(X_test)



