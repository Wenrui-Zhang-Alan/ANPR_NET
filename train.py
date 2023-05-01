import torch
import cv2
import xml.etree.ElementTree as ET
import torchvision.models as models
import numpy as np
import torch.utils.data as Data
import torch.optim as optim
import torchvision.transforms as transforms
import os

from torch.utils.data import DataLoader, Dataset
from ANPR_NET import *
from torch.utils.tensorboard import SummaryWriter






def preprocessing():

    all_img = [] # preprocessed image
    annotations = [] # preprocessed annotations

    for i in range(347):
        img = cv2.imread("./images/Cars"+ str(i) +".png")
        o_x,o_y = img.shape[1],img.shape[0] 
        r_x,r_y = 224/o_x, 224/o_y
        # Parse the annotation file
        tree = ET.parse("annotations/Cars"+ str(i) +".xml").getroot()
        new_xmin,new_ymin,new_xmax,new_ymax = float(tree[4][5][0].text)*r_x, float(tree[4][5][1].text)*r_y, float(tree[4][5][2].text) * r_x, float(tree[4][5][3].text)*r_y
        annotations.append([new_xmin,new_ymin,new_xmax,new_ymax]) # store annotation

        img = cv2.resize(img,(224,224)) 
        img = torch.tensor(img,dtype=torch.float32)
        img = img.permute(2,0,1)

        # Image Normalization, as done in resnet
        img = img / 255
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
        img = normalize(img)

        img = img.unsqueeze(0)
        all_img.append(img) # store image

    batched_img = all_img[0]
    for i in range(1,347):
        batched_img = torch.cat((batched_img,all_img[i]),dim=0)
        
    

    return batched_img,torch.tensor(annotations,dtype=torch.float32)


def train(lr,episode_len,save_dir,saving_freq):
    
    # initialization
    criterion = nn.MSELoss()
    model = ANPR_NET()  
    optimizer = optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter()
    

    # loading data
    data, target = preprocessing()
    data_set = Data.TensorDataset(data,target)
    loader = DataLoader(dataset=data_set,
                        batch_size=32,
                        shuffle=True)
    

    # training
    global_steps = 0
    for i in range(episode_len):

        step = 0
        total_loss = 0

        for batch_img, batch_y in loader:

            optimizer.zero_grad()
            out = model(batch_img)
            loss = criterion(out,batch_y)
            if global_steps % 5 == 0:   
                print("Training step {}, loss {}".format(global_steps,loss))
            loss.backward() # back prop
            optimizer.step()

            writer.add_scalar('training_loss', loss, global_steps)
            step += 1
            global_steps += 1
            total_loss += loss

            # saving model
            if global_steps % saving_freq == 0:
                state_to_save = model.state_dict()
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(state_to_save, save_dir+"/No_pre_saved_model_{}(0.001,500)".format(global_steps))

            

        print("Episode {} finished, average loss {}......".format(i,total_loss/step,2))
        



    


train(0.001,500,"saved_models",300)

    









