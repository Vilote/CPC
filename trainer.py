import os
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
import torch.distributed as dist
import torch.multiprocessing as mp

current_directory = os.path.dirname(os.path.realpath(__file__))
def create_model_training_folder(writer, files_to_same):
    model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        for file in files_to_same:
            copyfile(file, os.path.join(model_checkpoints_folder, os.path.basename(file)))

class CDCK2Trainer:
    def __init__(self, model, device, optimizer, config):
        self.model = model
        self.device = device
        if self.device==0:
            self.writer = SummaryWriter(f"runs-Lof/seed300_pretrain_class_in_{config['trainer']['class_start']}-{config['trainer']['class_end']}")
            create_model_training_folder(self.writer, files_to_same=[os.path.join(current_directory,"config/config.yaml",),"main.py", 'trainer.py'])
            
            
        self.optimizer = optimizer
        self.config = config
       


    def train(self, train_loader,epoch_counter ):
        self.model.train()
        train_loss_epoch = 0
        for data, _ in train_loader:
            data = data.to(self.device)
            data_reverse = torch.flip(data,dims=[2]).to(self.device)

            self.optimizer.zero_grad()
            hidden1 = self.model.module.init_hidden1(self.config['trainer']["batch_size"], use_gpu=True)
            hidden2 = self.model.module.init_hidden2(self.config['trainer']["batch_size"], use_gpu=True)
            
            acc, loss, hidden1, hidden2 = self.model.module(data, data_reverse, hidden1, hidden2)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), max_norm=1.0)
            self.optimizer.step()
            lr = self.optimizer.update_learning_rate()
            train_loss_epoch += loss.item()
        train_loss_epoch /= len(train_loader)
        if self.device==0:
            print(f"The loss on train dataset: {train_loss_epoch}")
            self.writer.add_scalar('train_loss_epoch', train_loss_epoch, global_step=epoch_counter)
    
    
    def eval(self, val_loader, epoch_counter):
        self.model.eval()
        eval_loss_epoch = 0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(self.device)
                data_reverse = torch.flip(data,dims=[2]).to(self.device)
                hidden1 = self.model.module.init_hidden1(self.config['trainer']["batch_size"], use_gpu=True)
                hidden2 = self.model.module.init_hidden2(self.config['trainer']["batch_size"], use_gpu=True)
                
                acc, loss, hidden1, hidden2 = self.model(data, data_reverse, hidden1, hidden2)
                eval_loss_epoch += loss.item()
        eval_loss_epoch /= len(val_loader)
        if self.device==0:
            print(f"The loss on eval dataset: {eval_loss_epoch}")
            self.writer.add_scalar('eval_loss_epoch', eval_loss_epoch, global_step=epoch_counter)
        return eval_loss_epoch 

            
    
    
    def train_and_val(self, train_dataset, val_dataset, train_dataset_sampler, val_dataset_sampler):
        train_loader = DataLoader(train_dataset, batch_size=self.config['trainer']["batch_size"],
                                  drop_last=True, sampler=train_dataset_sampler)
        val_loader = DataLoader(val_dataset, batch_size=self.config['trainer']["batch_size"],
                                  drop_last=True, sampler=val_dataset_sampler)
        if self.device==0:
            model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        
        loss_min = 1000000
        for epoch_counter in range(self.config['trainer']["max_epochs"]):
            if self.device==0:
                print(f'Epoch={epoch_counter}')
            self.train(train_loader, epoch_counter)
            eval_loss_epoch = self.eval(val_loader, epoch_counter)
            if eval_loss_epoch <= loss_min:
                if self.device==0:
                    torch.save(self.model.module, os.path.join(model_checkpoints_folder, 'model_best.pth'))
                loss_min = eval_loss_epoch
            if self.device==0:
                print("End of epoch {}".format(epoch_counter))
        if self.device==0:
            torch.save(self.model, os.path.join(model_checkpoints_folder, 'model.pth'))
