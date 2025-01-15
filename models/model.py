import torch
import numpy as np
import torch.nn as nn
from models.encoder_and_projection import Encoder_and_projection



class CDCK6(nn.Module):
    ''' CDCK2 with double decoder and a shared encoder '''
    def __init__(self, timestep, batch_size, seq_len):

        super(CDCK6, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep
        
        self.encoder = Encoder_and_projection()
        
        self.gru1 = nn.GRU(256, 256, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk1  = nn.ModuleList([nn.Linear(256, 256) for i in range(timestep)])
        
        self.gru2 = nn.GRU(256, 256, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk2  = nn.ModuleList([nn.Linear(256, 256) for i in range(timestep)])
        
        self.softmax  = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

       
        for layer_p in self.gru1._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru1.__getattr__(p), mode='fan_out', nonlinearity='relu')
        for layer_p in self.gru2._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru2.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden1(self, batch_size, use_gpu=True): 
        if use_gpu:
            return torch.zeros(1, batch_size, 256).cuda()
        else:
            return torch.zeros(1, batch_size, 256)
       

    def init_hidden2(self, batch_size, use_gpu=True):
        if use_gpu:
            return torch.zeros(1, batch_size, 256).cuda()
        else:
            return torch.zeros(1, batch_size, 256)

    def forward(self, x, x_reverse, hidden1, hidden2):
        batch = x.size()[0]
        nce = 0 
        t_samples = torch.randint(int(self.seq_len/32)-self.timestep, size=(1,)).long() 

       
        z = self.encoder(x)
        z = z.transpose(1,2)
        encode_samples = torch.empty((self.timestep,batch,256)).float() 
        
        for i in np.arange(1, self.timestep+1):
            encode_samples[i-1] = z[:,t_samples+i,:].view(batch,256)
            
        forward_seq = z[:,:t_samples+1,:] 
        output1, hidden1 = self.gru1(forward_seq, hidden1)
        
        c_t = output1[:,t_samples,:].view(batch, 256) 
        pred = torch.empty((self.timestep,batch,256)).float() 
        
        for i in np.arange(0, self.timestep):
            linear = self.Wk1[i]
            pred[i] = linear(c_t) 
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1)) 
            correct1 = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch))) 
            nce += torch.sum(torch.diag(self.lsoftmax(total))) 

       
        z = self.encoder(x_reverse)
      
        z = z.transpose(1,2)
        encode_samples = torch.empty((self.timestep,batch,256)).float() 
        for i in np.arange(1, self.timestep+1):
            encode_samples[i-1] = z[:,t_samples+i,:].view(batch,256) 
            
        forward_seq = z[:,:t_samples+1,:] 
        output2, hidden2 = self.gru2(forward_seq, hidden2) 
        c_t = output2[:,t_samples,:].view(batch, 256) 
        pred = torch.empty((self.timestep,batch,256)).float() 
        for i in np.arange(0, self.timestep):
            linear = self.Wk2[i]
            pred[i] = linear(c_t) 
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1)) 
            correct2 = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch))) 
            nce += torch.sum(torch.diag(self.lsoftmax(total)))

        nce /= -1.*batch*self.timestep
        nce /= 2. 
        accuracy = 1.*(correct1.item()+correct2.item())/(batch*2) 
      

        return accuracy, nce, hidden1, hidden2

    def predict(self, x, x_reverse, hidden1, hidden2):
        
        z1 = self.encoder(x)
        z1 = z1.transpose(1,2)
        output1, hidden1 = self.gru1(z1, hidden1) 
        z2 = self.encoder(x_reverse)
        z2 = z2.transpose(1,2)
        output2, hidden2 = self.gru2(z2, hidden2)

        return torch.cat((output1, output2), dim=2) [:,-1,:]
       

class CDCK5(nn.Module):
    ''' CDCK2 with a different decoder '''
    def __init__(self, timestep, batch_size, seq_len):

        super(CDCK5, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep
        self.encoder = Encoder_and_projection()

        self.gru = nn.GRU(256, 40, num_layers=2, bidirectional=False, batch_first=True)
        self.Wk  = nn.ModuleList([nn.Linear(40, 256) for i in range(timestep)])
        self.softmax  = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden(self, batch_size, use_gpu=True):
        if use_gpu:
            return torch.zeros(2*1, batch_size, 40).cuda()
        else:
            return torch.zeros(2*1, batch_size, 40)

    def forward(self, x, hidden):
        batch = x.size()[0]
        t_samples = torch.randint(int(self.seq_len/32)-self.timestep, size=(1,)).long() # randomly pick time stamps
        z = self.encoder(x)
        z = z.transpose(1,2)
        nce = 0 
        encode_samples = torch.empty((self.timestep,batch,256)).float() 
        
        for i in np.arange(1, self.timestep+1):
            encode_samples[i-1] = z[:,t_samples+i,:].view(batch,256) 
            
        forward_seq = z[:,:t_samples+1,:] 
        output, hidden = self.gru(forward_seq, hidden) 
        c_t = output[:,t_samples,:].view(batch, 40)
        pred = torch.empty((self.timestep,batch,256)).float() 
        
        for i in np.arange(0, self.timestep):
            decoder = self.Wk[i]
            pred[i] = decoder(c_t) 
            
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1)) 
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch))) 
            nce += torch.sum(torch.diag(self.lsoftmax(total))) 
            
        nce /= -1.*batch*self.timestep
        accuracy = 1.*correct.item()/batch

        return accuracy, nce, hidden

    def predict(self, x, hidden):
      
        z = self.encoder(x)
        z = z.transpose(1,2)
        output, hidden = self.gru(z, hidden) 

        return output[:,-1,:], hidden 
      
        
        
        
class CDCK2(nn.Module):
    def __init__(self, timestep, batch_size, seq_len):

        super(CDCK2, self).__init__()
        
        self.encoder = Encoder_and_projection()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep
        
        self.gru = nn.GRU(input_size = 256, hidden_size = 128, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk  = nn.ModuleList([nn.Linear(128, 256) for i in range(timestep)])
        self.softmax  = nn.Softmax(dim=1)
        self.lsoftmax = nn.LogSoftmax(dim=1)

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden(self, batch_size, use_gpu=True):
        if use_gpu:
            return torch.zeros(1, batch_size, 128).cuda()
        else:
            return torch.zeros(1, batch_size, 128)

    def forward(self, x, hidden):
        batch = x.size()[0]
        t_samples = torch.randint(int(self.seq_len/32) - self.timestep, size=(1,)).long() 
        z = self.encoder(x)
        z = z.transpose(1,2)
        nce = 0 
        encode_samples = torch.empty((self.timestep,batch,256)).float()
        
        for i in np.arange(1, self.timestep+1):
            encode_samples[i-1] = z[:,t_samples+i,:].view(batch,256) 
        
        
        
        forward_seq = z[:,:t_samples+1,:] 
        
        output, hidden = self.gru(forward_seq, hidden) 
        c_t = output[:,t_samples,:].view(batch, 128) 
        
        pred = torch.empty((self.timestep,batch,256)).float() 
        
        for i in np.arange(0, self.timestep):
            
            linear = self.Wk[i]
            pred[i] = linear(c_t) 
            
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1)) 
            
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch))) 
            nce += torch.sum(torch.diag(self.lsoftmax(total))) 
            
            
        nce /= -1.*batch*self.timestep
        accuracy = 1.*correct.item()/batch

        return accuracy, nce, hidden

    def predict(self, x, hidden):

        z = self.encoder(x)
        z = z.transpose(1,2)
        output, hidden = self.gru(z, hidden) 

        return output[:,-1,:] ,hidden
       
