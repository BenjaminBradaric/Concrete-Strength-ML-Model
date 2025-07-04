import pandas as pd
import os
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

## Fetching the csv file----------------------------------------------
file_path = r"MY_PATH"

if os.path.exists(file_path):
    concrete_data = pd.read_csv(file_path)
else:
    print("Cant access or find file")
##--------------------------------------------------------------------

# Dividing the label strength from the other data cloumns
concrete_data_summary = concrete_data.describe()

outlier_cols = ["Blast Furnace Slag", "Water", "Superplasticizer", "Fine Aggregate", "Age"]

def cap_outliers(df, col):
    q1 = concrete_data_summary[col].loc["25%"]
    q3 = concrete_data_summary[col].loc["75%"]
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    df.loc[df[col] < lower_bound, col] = concrete_data_summary[col].loc["mean"]
    df.loc[df[col] > upper_bound, col] = concrete_data_summary[col].loc["mean"]
    return df
    
    
for col in outlier_cols:
    concrete_data = cap_outliers(concrete_data, col)

X = concrete_data.drop(["Strength"], axis = 1)
columns_to_drop = [ c for c in concrete_data.columns if c != "Strength"]
Y = concrete_data.drop(columns_to_drop, axis = 1)

features_training_set, features_test_set, labels_train_set, labels_test_set = train_test_split(X, Y, test_size = .2, random_state=42)


#Scaling Data then converting to from pandas dataframe to numpy array because we want tensor then we must go from pandad -> numpy -> pytorch
feature_scaler = StandardScaler() #must have the same scaler for both to avvoid memory leak 
features_training_set = feature_scaler.fit_transform(features_training_set.to_numpy())
features_test_set = feature_scaler.fit_transform(features_test_set.to_numpy())

label_scaler = StandardScaler()
labels_train_set = label_scaler.fit_transform(labels_train_set.to_numpy())
labels_test_set = label_scaler.fit_transform(labels_test_set.to_numpy())


features_training_set = torch.tensor(features_training_set, dtype= torch.float32)
features_test_set = torch.tensor(features_test_set, dtype=torch.float32)
labels_train_set = torch.tensor(labels_train_set, dtype= torch.float32)
labels_test_set = torch.tensor(labels_test_set, dtype = torch.float32)

sigma = label_scaler.scale_[0]
