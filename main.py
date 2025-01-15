import os
import sys
import torch
import yaml
import random
import numpy as np
from models.model import CDCK2,CDCK5,CDCK6
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import train_test_split
from trainer import CDCK2Trainer
from get_dataset import PreTrainDataset_prepared
from torch.utils.data import TensorDataset,DistributedSampler

torch.backends.cudnn.benchmark = True
print(torch.__version__)

RANDOM_SEED = 300 # any random number


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现
set_seed(RANDOM_SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"

def setup(rank, world_size):
    # 初始化分布式环境
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()
#学习率动态调整
class ScheduledOptim(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 128 
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0 
        self.delta = 1

    def state_dict(self):
        self.optimizer.state_dict()

    def step(self):
        """Step by the inner optimizer"""
        self.optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer"""
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2

    def update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr
    
def main(rank,world_size):
    setup(rank, world_size)
    current_directory = os.path.dirname(os.path.realpath(__file__))
    config = yaml.load(open(os.path.join(current_directory, "config/config.yaml"), "r"), Loader=yaml.FullLoader)

    device = rank
    print(f"Training with: {device}")

    X_train, Y_train = PreTrainDataset_prepared()

  
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.3, random_state=30)
    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))
    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
    
    val_dataset_sampler = DistributedSampler(val_dataset, num_replicas=world_size,rank=rank)
    train_dataset_sampler = DistributedSampler(train_dataset,num_replicas=world_size, rank=rank)
    
   
    
   
    myCDCK2 = CDCK6(timestep=config['trainer']["timestep"],seq_len=config['trainer']["seq_len"],batch_size=config['trainer']["batch_size"]).to(device)

 


    optimizer = ScheduledOptim(
        torch.optim.Adam(
            filter(lambda p: p.requires_grad, myCDCK2.parameters()), 
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),300
        )


                                    
                                        
                                                
    myCDCK2 = DDP(myCDCK2, device_ids=[device])
    
    trainer = CDCK2Trainer(model=myCDCK2,
                          device=device,
                          optimizer=optimizer,
                          config=config)
  
    

    trainer.train_and_val(train_dataset=train_dataset, val_dataset=val_dataset,train_dataset_sampler=train_dataset_sampler, val_dataset_sampler=val_dataset_sampler)

if __name__ == '__main__':
    world_size = torch.cuda.device_count()  # 获取可用的 GPU 数量
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    

