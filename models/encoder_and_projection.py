from models.mlp_head import MLPHead
from torch import nn
import torch.nn.functional as F
from models.complexcnn import ComplexConv, ComplexConv_trans

class Encoder_and_projection(nn.Module):
    def __init__(self):
        super(Encoder_and_projection, self).__init__()
        self.encoder_and_projection = nn.Sequential(
            ComplexConv(in_channels=1, out_channels=128, kernel_size=4, padding =1, stride=2),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True),
          
            
            ComplexConv(in_channels=128, out_channels=128, kernel_size=4, padding =1, stride=2),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True),
           
            
            ComplexConv(in_channels=128, out_channels=128, kernel_size=4, padding =1, stride=2),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True),
           
            
            ComplexConv(in_channels=128, out_channels=128, kernel_size=4, padding =1, stride=2),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True),
        
            
            ComplexConv(in_channels=128, out_channels=128, kernel_size=4, padding =1, stride=2),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True)
         
            
        )
       
        
       



    def forward(self,x):
        x = self.encoder_and_projection(x)
        return x
