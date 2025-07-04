import torch 
from torch import nn
import torch.nn.init as init
from Processing import features_training_set

#The Model
class RegressionModel(nn.Module):
    def __init__(self, input_dim, output_1_dim, output_2_dim, output_3_dim, output_4_dim, output_dim):
        super().__init__()

        ##first layer
        self.weights1 = nn.Parameter(torch.empty( input_dim, output_1_dim, requires_grad= True))
        self.bias1 = nn.Parameter(torch.empty(output_1_dim, requires_grad= True))

        ##second layer
        self.weights2 = nn.Parameter(torch.empty( output_1_dim, output_2_dim, requires_grad= True))
        self.bias2 = nn.Parameter(torch.empty(output_2_dim, requires_grad= True))

        ##third layer
        self.weights3 = nn.Parameter(torch.empty(output_2_dim, output_3_dim, requires_grad= True))
        self.bias3 = nn.Parameter(torch.empty(output_3_dim, requires_grad= True))

        #Fourth layer
        self.weights4 = nn.Parameter(torch.empty(output_3_dim, output_4_dim, requires_grad= True))
        self.bias4 = nn.Parameter(torch.empty(output_4_dim, requires_grad=True))

        ##Output layer
        self.weights5 = nn.Parameter(torch.empty(output_4_dim, output_dim, requires_grad= True))
        self.bias5 = nn.Parameter(torch.empty(output_dim, requires_grad= True))

        #Weight initialization 
        init.kaiming_uniform_(self.weights1, a=0.01, mode='fan_in', nonlinearity='leaky_relu') #a need to be equal to alpha
        init.zeros_(self.bias1)

        init.kaiming_uniform_(self.weights2, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        init.zeros_(self.bias2)

        init.kaiming_uniform_(self.weights3, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        init.zeros_(self.bias3)

        init.kaiming_uniform_(self.weights4, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        init.zeros_(self.bias4)

        init.kaiming_uniform_(self.weights5, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        init.zeros_(self.bias5)

        #Dropout
        self.layer2_dropout = nn.Dropout(0.4)
        self.layer3_dropout = nn.Dropout(0.4)
        self.layer4_dropout = nn.Dropout(0.4)

        #Batch Normalization
        self.layer1_batch_norm = nn.BatchNorm1d(output_1_dim)
        self.layer2_batch_norm = nn.BatchNorm1d(output_2_dim)
        self.layer3_batch_norm = nn.BatchNorm1d(output_3_dim)
        self.layer4_batch_norm = nn.BatchNorm1d(output_4_dim)
    
    def LeakyReLu(self, y: torch.Tensor, alpha : float = 0.01) -> torch.Tensor:
        return torch.where(y > 0, y , y * alpha)
    
    def ReLu(self, y: torch.Tensor, alpha : float = 0) -> torch.Tensor:
        return torch.where(y > 0, y , y * alpha)
                

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = x @ self.weights1 + self.bias1
        y = self.layer1_batch_norm(y)
        y = self.LeakyReLu(y)
        
        y2 = y @ self.weights2 + self.bias2
        y2 = self.layer2_batch_norm(y2)
        y2 = self.LeakyReLu(y2)
        y2 = self.layer2_dropout(y2)

        y3 = y2 @ self.weights3 + self.bias3
        y3 = self.layer3_batch_norm(y3)
        y3 = self.LeakyReLu(y3)
        y3 = self.layer3_dropout(y3) 

        y4 = y3 @ self.weights4 + self.bias4
        y4 = self.layer4_batch_norm(y4)
        y4 = self.LeakyReLu(y4)
        y4 = self.layer4_dropout(y4) 

        y5 = y4 @ self.weights5 + self.bias5
        y5 = self.LeakyReLu(y5)

        return y5
    

model = RegressionModel(features_training_set.shape[1], 512, 256, 128, 32, 1)