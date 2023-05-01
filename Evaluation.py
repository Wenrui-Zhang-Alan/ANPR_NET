import torch
import cv2
import xml.etree.ElementTree as ET
import torchvision.models as models
import numpy as np
import torch.utils.data as Data
import torch.optim as optim
import torchvision.transforms as transforms
import os
import imutils
import pytesseract
from pytesseract import Output
import easyocr
from ANPR_NET import *
from GPS import *
import torchvision
import random
from matplotlib import pyplot as plt
import re
import warnings

warnings.filterwarnings("ignore")

def preprocessing(path):

    img = cv2.imread(path)

    img = cv2.resize(img,(224,224)) 
    img = torch.tensor(img,dtype=torch.float32)
    img = img.permute(2,0,1)

    img = img / 255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    img = normalize(img)

    img = img.unsqueeze(0)
    

    return img



def fram_preprocessing(img):

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

    return img

    
def convert_back(num,orin):

    return int((num/224)*orin)


def str_check(str):

    for char in str:
        if ord(char) not in range(48,60) and ord(char) not in range(65,90) and ord(char) not in range(97,122):
            str = str.replace(char,"")

    return str


def visualization(image,output):

    coco_names = ['unlabeled', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in coco_names]

    result_image = np.array(image.copy())
    for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
        if score > 0.5:
            color = random.choice(colors)
            
            # draw box
            tl = round(0.002 * max(result_image.shape[0:2])) + 1  # line thickness
            c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(result_image, c1, c2, color, thickness=tl)
            # draw text
            display_txt = "%s: %.1f%%" % (coco_names[label], 100*score)
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(display_txt, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(result_image, c1, c2, color, -1)  # filled
            cv2.putText(result_image, display_txt, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    

    

    mask = output['masks'].tolist()
    
    seg = image.copy()
    for i in range(len(mask)):
        cur = mask[i][0]
        color = random.choice(colors)

        for col in range(len(cur)):
            for row in range(len(cur[col])):
                if cur[col][row] >= 0.2 and output['scores'].tolist()[i] > 0.5:
                    seg[col][row][0] = color[0]
                    seg[col][row][1] = color[1]
                    seg[col][row][2] = color[2]
                    


    
    result_image = cv2.addWeighted(seg,0.5,result_image,1-0.5,0)

    plt.figure(figsize=(20, 15))

    return result_image

def filter(output): # return the car with highest confidence score
    
    labels = output["labels"].tolist()
    confi = output["scores"].tolist()
    rectangles = output["boxes"].tolist()

    highest = -99.0
    index = -1

    for i in range(len(labels)):
        if labels[i] == 3 and confi[i] > highest:
            highest = confi[i]
            index = i

    if index == -1:
        return None
    
    return rectangles[index]



def v_view(path,addtional):

    print("Initializing result file...")

    with open('Voutput.txt', 'a') as file:
        file.write('_____ New Detection _____ \n')
    
    # MaskRcnn
    print("Initializing MaskRcnn...")
    MaskRcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    MaskRcnn_model = MaskRcnn_model.eval()

    # ANPR_NET
    print("Initializing ANPR_NET...")

    model = ANPR_NET()
    model.load_state_dict(torch.load("saved_models/saved_model_5100(0.001,500)"))
    model.eval()
    cap = cv2.VideoCapture(path)



    print("Reading video data...")
    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # maskrcnn
        image_tensor = torchvision.transforms.functional.to_tensor(frame)
        outer_out = MaskRcnn_model([image_tensor])[0]

        if addtional == 1:
            rr = visualization(frame,outer_out)
            cv2.imshow("MaskRcnn Result",rr)
            cv2.waitKey(0)

        # r = visualization(frame,outer_out)
        index = filter(outer_out)

        # No car detetcted in current frame
        if index == None:
            continue 

        for item in range(len(index)):
            index[item] = int(index[item])
        
        # segment
        seg_frame = frame[index[1]:index[3],index[0]:index[2],:]
        
        cv2.imshow("Real-time Video",frame)
        cv2.imshow("Car",seg_frame)
        cv2.waitKey(0)

        plate,v_result = Vp_view(seg_frame)

        if plate == None:
            continue
        
        # Filtering invalid results
        pattern = r'^[A-Z0-9 ]+$'

        if re.match(pattern,plate):
            with open('Voutput.txt', 'a') as file:
                file.write(plate + "\n")

        # visualisation

        
        cv2.imshow("Detection",v_result)
        
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

    print("\n ___FINISHED___ View result in Voutput.text")
    cap.release()
    cv2.destroyAllWindows()

    


def Vp_view(o_img):

    img = o_img.copy()

    # Preprocessing
    img = cv2.resize(img,(224,224)) 
    img = torch.tensor(img,dtype=torch.float32)
    img = img.permute(2,0,1)

    img = img / 255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    img = normalize(img)
    img = img.unsqueeze(0)
   

    ANPR_model = ANPR_NET()
    ANPR_model.load_state_dict(torch.load("saved_models/saved_model_5100(0.001,500)"))
    ANPR_model.eval()
    out = ANPR_model(img)

    # drwaing on original image
    out = np.array(out.detach())
    
    o_x,o_y = o_img.shape[1],o_img.shape[0] 
    x_min_n, y_min_n, x_max_n, y_max_n = convert_back(out[0][0],o_x),convert_back(out[0][1],o_y),convert_back(out[0][2],o_x),convert_back(out[0][3],o_y)
    

    # ocr
    ocr_part = o_img[y_min_n:y_max_n,x_min_n:x_max_n,:]
    gray = cv2.cvtColor(ocr_part, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 13, 15, 15)


    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  

    
    
   
    reader = easyocr.Reader(['en']) 

    result = reader.readtext(gray)

    if len(result) != 0:
        return result[0][1],gray
    else:
        return None,gray

    
    
    
    



def p_view(path):

    print("Reading data...")

    data = preprocessing(path)

    print("Initializing ANPR_model...")
    ANPR_model = ANPR_NET()
    ANPR_model.load_state_dict(torch.load("saved_models/saved_model_5100(0.001,500)"))
    ANPR_model.eval()
    out = ANPR_model(data)

    

    # drwaing on original image
    print("Drawing...")
    data = data.permute(0,3,2,1)
    out = np.array(out.detach())
    img = cv2.imread(path)
    o_x,o_y = img.shape[1],img.shape[0] 
    x_min_n, y_min_n, x_max_n, y_max_n = convert_back(out[0][0],o_x),convert_back(out[0][1],o_y),convert_back(out[0][2],o_x),convert_back(out[0][3],o_y)
    

    # ocr
    print("Ocr model applied...")
    ocr_part = img[y_min_n:y_max_n,x_min_n:x_max_n,:]
    gray = cv2.cvtColor(ocr_part, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 13, 15, 15)


    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  

    print("Showing detetcted result...")
    cv2.imshow("test",gray)
    cv2.waitKey(0)
   
    reader = easyocr.Reader(['en']) 
    print("Ocr reading...")
    result = reader.readtext(gray)

    print("\n ___ RESULT ___")
    if len(result) != 0:
        word = result[0][1]
        print("Detected : {}".format(word))
    else:
        print("Detected Nothing.")


    
    
    
    # GPS

    gps_meta = locate(path)
    print("GPS : {}".format(gps_meta))
    cv2.rectangle(img, (x_min_n, y_min_n), (x_max_n, y_max_n), (0, 255, 0), 2)
    cv2.imshow("result",img)
    cv2.waitKey(0)



