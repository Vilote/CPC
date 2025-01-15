import numpy as np
import random
import yaml
import os


current_directory = os.path.dirname(os.path.realpath(__file__))



'''

def PreTrainDataset_prepared():
    x = np.load(os.path.join(current_directory, "Dataset/ADS-B_4800_without_icao/X_train_90Class.npy"))
    y = np.load(os.path.join(current_directory, "Dataset/ADS-B_4800_without_icao/Y_train_90Class.npy"))
    
    
    x = x.transpose(0, 2, 1)
    train_index_shot = []
    for i in range(90):
        index_classi = [index for index, value in enumerate(y) if value == i]
        train_index_shot += index_classi[0:200]

    X_train = x[train_index_shot]
    mean_I = np.mean(X_train[:,0,:],axis=0,keepdims=True)
    mean_Q = np.mean(X_train[:,1,:],axis=0,keepdims=True)
    std_I = np.std(X_train[:,0,:],axis=0,keepdims=True)
    std_Q = np.std(X_train[:,1,:],axis=0,keepdims=True)
    
    X_train[:,0,:] = (X_train[:,0,:] -  mean_I) / std_I
    X_train[:,1,:] = (X_train[:,1,:] -  mean_Q) / std_Q
    
    Y_train = y[train_index_shot].astype(np.uint8)
   
    return X_train, Y_train

def FineTuneDataset_prepared():
    
    config = yaml.load(open(os.path.join(current_directory,"config/config.yaml"), "r"), Loader=yaml.FullLoader)
    params = config['finetune']
    k = params['k_shot']
   
    x = np.load(os.path.join(current_directory,"Dataset/ADS-B_4800_without_icao/X_train_10Class.npy"))
    y = np.load(os.path.join(current_directory,"Dataset/ADS-B_4800_without_icao/Y_train_10Class.npy"))
    x = x.transpose(0, 2, 1)
    X_test = np.load(os.path.join(current_directory,"Dataset/ADS-B_4800_without_icao/X_test_10Class.npy"))
    Y_test = np.load(os.path.join(current_directory,"Dataset/ADS-B_4800_without_icao/Y_test_10Class.npy")).reshape(-1,)
    X_test = X_test.transpose(0, 2, 1)
   
    finetune_index_shot = []
    for i in range(10):
        index_classi = [index for index, value in enumerate(y) if value == i]
        finetune_index_shot += random.sample(index_classi, k)
    X_train = x[finetune_index_shot]
    Y_train = y[finetune_index_shot].reshape(-1,)

    mean_I = np.mean(X_train[:,0,:],axis=0,keepdims=True)
    mean_Q = np.mean(X_train[:,1,:],axis=0,keepdims=True)
    std_I = np.std(X_train[:,0,:],axis=0,keepdims=True)
    std_Q = np.std(X_train[:,1,:],axis=0,keepdims=True)
    X_train[:,0,:] = (X_train[:,0,:] -  mean_I) / (std_I+0.0001)
    X_train[:,1,:] = (X_train[:,1,:] -  mean_Q) / (std_Q+0.0001)

    testmean_I = np.mean(X_test[:,0,:],axis=0,keepdims=True)
    testmean_Q = np.mean(X_test[:,1,:],axis=0,keepdims=True)
    teststd_I = np.std(X_test[:,0,:],axis=0,keepdims=True)
    teststd_Q = np.std(X_test[:,1,:],axis=0,keepdims=True)
   
   
    X_test[:,0,:] = (X_test[:,0,:] -  testmean_I) / (teststd_I+0.0001)
    X_test[:,1,:] = (X_test[:,1,:] -  testmean_Q) / (teststd_Q+0.0001)
    return X_train, X_test, Y_train, Y_test


'''




def FineTuneDataset_prepared():
    
    config = yaml.load(open(os.path.join(current_directory,"config/config.yaml"), "r"), Loader=yaml.FullLoader)
    params = config['finetune']
    k = params['k_shot']
   
    x = np.load(os.path.join(current_directory,"DatasetLof/train-Fine.npy"))
    y = np.load(os.path.join(current_directory,"DatasetLof/train-FineY.npy"))
    
    X_test = np.load(os.path.join(current_directory,"DatasetLof/testsplit.npy"))
    Y_test = (np.load(os.path.join(current_directory,"DatasetLof/testsplit-Y.npy"))-31).reshape(-1,)
    
   
    finetune_index_shot = []
    for i in range(30,40):
        index_classi = [index for index, value in enumerate(y) if value == i+1]
        finetune_index_shot += random.sample(index_classi, k)
    X_train = x[finetune_index_shot]
    Y_train = (y[finetune_index_shot]-31).reshape(-1,)

    mean_I = np.mean(X_train[:,0,:],axis=0,keepdims=True)
    mean_Q = np.mean(X_train[:,1,:],axis=0,keepdims=True)
    std_I = np.std(X_train[:,0,:],axis=0,keepdims=True)
    std_Q = np.std(X_train[:,1,:],axis=0,keepdims=True)
    X_train[:,0,:] = (X_train[:,0,:] -  mean_I) / (std_I+0.0001)
    X_train[:,1,:] = (X_train[:,1,:] -  mean_Q) / (std_Q+0.0001)

    testmean_I = np.mean(X_test[:,0,:],axis=0,keepdims=True)
    testmean_Q = np.mean(X_test[:,1,:],axis=0,keepdims=True)
    teststd_I = np.std(X_test[:,0,:],axis=0,keepdims=True)
    teststd_Q = np.std(X_test[:,1,:],axis=0,keepdims=True)
   
   
    X_test[:,0,:] = (X_test[:,0,:] -  testmean_I) / (teststd_I+0.0001)
    X_test[:,1,:] = (X_test[:,1,:] -  testmean_Q) / (teststd_Q+0.0001)
    return X_train, X_test, Y_train, Y_test





def PreTrainDataset_prepared():
    x = np.load('/home/zhoujunhui/CPC/DatasetLof/Train.npy')
    y = np.load('/home/zhoujunhui/CPC/DatasetLof/Train-Y.npy').reshape((-1,1))
    train_index_shot = []
    for i in range(0,10):
        index_classi = [index for index, value in enumerate(y) if value == i]
        train_index_shot += index_classi[0:2000]
        
    X_train = x[train_index_shot]
    Y_train = y[train_index_shot].astype(np.uint8)
    
    mean_I = np.mean(X_train[:,0,:],axis=0,keepdims=True)
    mean_Q = np.mean(X_train[:,1,:],axis=0,keepdims=True)
    std_I = np.std(X_train[:,0,:],axis=0,keepdims=True)
    std_Q = np.std(X_train[:,1,:],axis=0,keepdims=True)
    X_train[:,0,:] = (X_train[:,0,:]-mean_I) / std_I
    X_train[:,1,:] = (X_train[:,1,:]-mean_Q) / std_Q
   

    return X_train, Y_train



'''
def FineTuneDataset_prepared():
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
    params = config['finetune']
    k = params['k_shot']
   
    x = np.load("/home/zhoujunhui/ADS-B/Dataset/X_test_6class.npy")
    y = np.load("/home/zhoujunhui/ADS-B/Dataset/Y_test_6class.npy")
    
    test_index_shot = []
    finetune_index_shot = []
  
    for i in range(10,16):
        index_class = [index for index, value in enumerate(y) if value == i]
        finetune_index_shot += random.sample(index_class[2000:3000], k)
        test_index_shot += index_class[3000:4000]
    
    X_train = x[finetune_index_shot]
    Y_train = y[finetune_index_shot]-10
    X_test = x[test_index_shot]
    Y_test = y[test_index_shot]-10
    
    

    mean_I = np.mean(X_train[:,0,:],axis=0,keepdims=True)
    mean_Q = np.mean(X_train[:,1,:],axis=0,keepdims=True)
    std_I = np.std(X_train[:,0,:],axis=0,keepdims=True)
    std_Q = np.std(X_train[:,1,:],axis=0,keepdims=True)
    X_train[:,0,:] = (X_train[:,0,:] -  mean_I) / (std_I+0.0001)
    X_train[:,1,:] = (X_train[:,1,:] -  mean_Q) / (std_Q+0.0001)

    testmean_I = np.mean(X_test[:,0,:],axis=0,keepdims=True)
    testmean_Q = np.mean(X_test[:,1,:],axis=0,keepdims=True)
    teststd_I = np.std(X_test[:,0,:],axis=0,keepdims=True)
    teststd_Q = np.std(X_test[:,1,:],axis=0,keepdims=True)
   
   
    X_test[:,0,:] = (X_test[:,0,:] -  testmean_I) / (teststd_I+0.0001)
    X_test[:,1,:] = (X_test[:,1,:] -  testmean_Q) / (teststd_Q+0.0001)
    return X_train, X_test, Y_train, Y_test
'''