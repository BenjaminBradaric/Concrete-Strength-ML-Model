import torch
from torch import nn
from model import model 
from Processing import features_training_set, features_test_set, labels_train_set, labels_test_set, sigma 


model.load_state_dict(torch.load("saved_model.pth"))
eval_metric_list = []
mse = nn.MSELoss()
model.eval()

with torch.inference_mode():

    train_preds = model(features_training_set)
    train_rmse = torch.sqrt(mse(train_preds, labels_train_set)).item()

    test_predictions = model(features_test_set)
    eval_metric = torch.sqrt(mse(test_predictions, labels_test_set)).item()


    print(f'RMSE from training sett {train_rmse}')
    print(f'RMSE from test sett {eval_metric}')
    print("\n")
    print(f'RMSE times Scaling factor from training sett {train_rmse * sigma}')
    print(f'RMSE times Scaling factor from test sett {eval_metric * sigma}')