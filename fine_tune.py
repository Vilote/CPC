import os
import torch
import yaml
import pandas as pd
from torch import nn
import numpy as np
import random
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader,DistributedSampler
from models.model import CDCK2
from models.classifier import Classifier
from get_dataset import FineTuneDataset_prepared, PreTrainDataset_prepared
from sklearn.model_selection import train_test_split
from models.model import CDCK2,CDCK5,CDCK6
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


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
    
def train(model, classifier, loss_nll, train_dataloader, optimizer_classifier,optimizer_model, epoch, device, writer,config):
    model = model.eval()
    classifier = classifier.train()
    correct = 0
    nll_loss = 0
    for data, target in train_dataloader:
        target = target.long()
        if torch.cuda.is_available():
            data = data.to(device)
            data_reverse = torch.flip(data,dims=[2]).to(device)
            target = target.to(device)
        


        optimizer_classifier.zero_grad()
        with torch.no_grad():
            hidden1 = model.init_hidden1(len(data), use_gpu=True)
            hidden2 = model.init_hidden2(len(data), use_gpu=True)
            output1 = model.predict(data, data_reverse, hidden1, hidden2)
        output = classifier(output1)
 
        output = F.log_softmax(output, dim=1)
     
        
        loss = loss_nll(output, target)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer_classifier.step()


        
        lr = optimizer_classifier.update_learning_rate()
        nll_loss += loss.item()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()  

    nll_loss /= len(train_dataloader)
    if device==0:
        print('Train Epoch: {} \tClass_Loss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
            epoch,
            nll_loss,
            correct,
            len(train_dataloader.dataset),
            100.0 * correct / len(train_dataloader.dataset))
        )
        writer.add_scalar('Accuracy/train', 100.0 * correct / len(train_dataloader.dataset), epoch)
        writer.add_scalar('Loss/train', nll_loss, epoch)

def evaluate(model, classifier, loss_nll, val_dataloader, epoch, device,writer,config):
    model = model.eval()
    classifier = classifier.eval()
 
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
                data_reverse = torch.flip(data,dims=[2]).to(device)
                
            hidden1 = model.init_hidden1(len(data), use_gpu=True)
            hidden2 = model.init_hidden2(len(data), use_gpu=True)
            
            output = model.predict(data, data_reverse, hidden1, hidden2)
            output = F.log_softmax(classifier(output), dim=1)
            
            test_loss += loss_nll(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_dataloader)

    fmt = '\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            test_loss,
            correct,
            len(val_dataloader.dataset),
            100.0 * correct / len(val_dataloader.dataset),
        )
    )
    writer.add_scalar('Accuracy/val', 100.0 * correct / len(val_dataloader.dataset), epoch)
    writer.add_scalar('Loss/val', test_loss, epoch)
    return test_loss

def test(model, classifier, test_dataloader, device, config):
    model = model.eval()
    classifier = classifier.eval()
    test_loss = 0
    correct = 0
    loss = nn.NLLLoss()
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                data_reverse = torch.flip(data,dims=[2]).to(device)
                target = target.to(device)
                loss = loss.to(device)
                
            hidden1 = model.init_hidden1(len(data), use_gpu=True)
            hidden2 = model.init_hidden2(len(data), use_gpu=True)
            
            output = model.predict(data, data_reverse, hidden1, hidden2)
            output = F.log_softmax(classifier(output), dim=1)
            
            test_loss += loss(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
     
    test_loss /= len(test_dataloader)
    fmt = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
                test_loss,
                correct,
                len(test_dataloader.dataset),
                100.0 * correct / len(test_dataloader.dataset),
        )
    )
    return 100.0 * correct / len(test_dataloader.dataset)

def train_and_test(model, classifier, loss_nll, train_dataloader, val_dataloader, optimizer_classifier,optimizer_model, epochs, save_path_online_network, save_path_classifier, device, writer, config):
    
    current_min_test_loss = 10000000000
    for epoch in range(1, epochs + 1):
        train(model, classifier, loss_nll, train_dataloader, optimizer_classifier, optimizer_model, epoch, device, writer,config)
        validation_loss = evaluate(model, classifier, loss_nll, val_dataloader, epoch, device,writer, config)
        if validation_loss < current_min_test_loss:
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(current_min_test_loss, validation_loss))
            torch.save(model, save_path_online_network)
            torch.save(classifier, save_path_classifier)
            current_min_test_loss = validation_loss
    
        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")

def run(train_dataloader, val_dataloader, test_dataloader, epochs, save_path_online_network, save_path_classifier, device, writer,config):
    device = device
    checkpoints_folder = os.path.join('runs-Lof',
                                      f"seed300_pretrain_class_in_{config['trainer']['class_start']}-{config['trainer']['class_end']}",
                                      'checkpoints')
    
    model = torch.load(os.path.join(checkpoints_folder, 'model_best.pth'))
    
    classifier = Classifier()
    loss_nll = nn.NLLLoss()
    
    model = model.to(device)
    classifier = classifier.to(device)
    
   
    optim_classifier = ScheduledOptim(
        torch.optim.Adam(
            filter(lambda p: p.requires_grad, classifier.parameters()), 
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-3, amsgrad=True),80
        )
    
    optimizer = ScheduledOptim(
        torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),200
        )

    
    train_and_test(model, 
                   classifier, 
                   loss_nll,
                   train_dataloader=train_dataloader,
                   val_dataloader=val_dataloader, 
                   optimizer_model= optimizer,
                   optimizer_classifier=optim_classifier,  
                   epochs=epochs, 
                   save_path_online_network=save_path_online_network, 
                   save_path_classifier=save_path_classifier,
                   writer=writer,
                   device=device, 
                   config=config)
 
    print("Test_result:")
    model = torch.load(save_path_online_network)
    classifier = torch.load(save_path_classifier)
    test_acc = test(model, classifier, test_dataloader, device, config=config)
    return test_acc

def main():
    current_directory = os.path.dirname(os.path.realpath(__file__))
    config = yaml.load(open(os.path.join(current_directory, "config/config.yaml"), "r"), Loader=yaml.FullLoader)
    config_ft = config['finetune']
    device = torch.device("cuda:0")
    test_acc_all = []

    for i in range(config['iteration']):
        print(f"iteration: {i}--------------------------------------------------------")
        set_seed(i)
        current_directory = os.path.dirname(os.path.realpath(__file__))
        writer = SummaryWriter(os.path.join(current_directory,f"log_finetuneLof/PT_{config['trainer']['class_start']}-{config['trainer']['class_end']}_FT_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot"))

        save_path_classifier = os.path.join(current_directory,f"model_weightLof/classifier_PT_{config['trainer']['class_start']}-{config['trainer']['class_end']}_FT_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot_{i}.pth")
        save_path_online_network = os.path.join(current_directory,f"model_weightLof/model_PT_{config['trainer']['class_start']}-{config['trainer']['class_end']}_FT_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot_{i}.pth")

        X_train, X_test, Y_train, Y_test = FineTuneDataset_prepared()
       
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=30)

        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
        train_dataloader = DataLoader(train_dataset, batch_size=config_ft['batch_size'],shuffle=True)

        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))
        val_dataloader = DataLoader(val_dataset, batch_size=config_ft['batch_size'], shuffle=True)

        test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
        test_dataloader = DataLoader(test_dataset, batch_size=config_ft['test_batch_size'],shuffle=True)

        # train
        test_acc = run(train_dataloader, val_dataloader, test_dataloader, epochs=config_ft['epochs'], save_path_online_network=save_path_online_network, save_path_classifier=save_path_classifier, device=device, writer=writer,config=config)
        test_acc_all.append(test_acc)
        writer.close()

    df = pd.DataFrame(test_acc_all)
    df.to_excel(f"./test_resultLof/PT_{config['trainer']['class_start']}-{config['trainer']['class_end']}_FT_{config_ft['class_start']}-{config_ft['class_end']}_{config_ft['k_shot']}shot(CPC_CDK6).xlsx")

    

if __name__ == '__main__':
    main()
  
