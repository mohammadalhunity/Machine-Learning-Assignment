



import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
import os

import torch
from torchvision import datasets,transforms,models
from torch.utils.data import Dataset,DataLoader
from PIL import Image

import sys
import torch.optim as optim

img_names=[] 
xml_names=[] 
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if os.path.join(dirname, filename)[-3:]!="xml":
            img_names.append(filename)
        else:
            xml_names.append(filename)
           
options={"with_mask":0,"without_mask":1,"mask_weared_incorrect":2}

#xmltodict.parse() : This is used to parse the given XML input and convert it into a dictionary.
#Generak syntax of xmltodict.parse() is xmltodict.parse(xml_input, encoding='utf-8', expat=expat, process_namespaces=False, namespace_separator=':', **kwargs)

# xml_input : It can be a either be a string or a file-like object. Here we are using read() method, which reads at most n bytes from file desciptor and return a string containing the bytes read. If the end of file referred to by fd has been reached, an empty string is returned.
# torchvision.transforms.Compose() : torchvision.transforms is used for common image transformations and when Compose is chained with it to Compose several transforms together.
# transforms.ToTensor(): This just converts input image to PyTorch tensor.
# torch.tensor(): It infers the dtype automatically. It always copies the data and torch.tensor(l) is equivalent to l.clone().detach().
# transforms.Resize() : The default interpolation is InterpolationMode.BILINEAR. It resize the input image as per the height and width provided.
# transforms.functional.crop() : Crop the given image at specified location and output size and it returs torch.Tensor.
#With preprocessed Images dataset is created and then we can split the dataset to training and test set.

def dataset_creation(image_list): 
    image_tensor=[]
    label_tensor=[]
    for i,j in enumerate(image_list):
        with open(path_annotations+j[:-4]+".xml") as fd:
            doc=xmltodict.parse(fd.read())
        if type(doc["annotation"]["object"])!=list:
            temp=doc["annotation"]["object"]
            x,y,w,h=list(map(int,temp["bndbox"].values()))
            label=options[temp["name"]]
            image=transforms.functional.crop(Image.open(path_image+j).convert("RGB"), y,x,h-y,w-x)
            image_tensor.append(my_transform(image))
            label_tensor.append(torch.tensor(label))
        else:
            temp=doc["annotation"]["object"]
            for k in range(len(temp)):
                x,y,w,h=list(map(int,temp[k]["bndbox"].values()))
                label=options[temp[k]["name"]]
                image=transforms.functional.crop(Image.open(path_image+j).convert("RGB"),y,x,h-y,w-x)
                image_tensor.append(my_transform(image))
                label_tensor.append(torch.tensor(label))
                
    final_dataset=[[k,l] for k,l in zip(image_tensor,label_tensor)]
    return tuple(final_dataset)


my_transform=transforms.Compose([transforms.Resize((226,226)),
                                 transforms.ToTensor()])

mydataset=dataset_creation(img_names)

mydataset[0]

#DataLoader() is an iterable that abstracts this complexity for us in an easy API.
#When shuffle is made true data is shuffled, after the iteration is over for all batches.

train_dataloader =DataLoader(dataset=trainset,batch_size=32,shuffle=True,num_workers=4)
test_dataloader =DataLoader(dataset=testset,batch_size=32,shuffle=True,num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available()
                               else "cpu")
import sys
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
    
train_features_np=train_features.numpy()

fig=plt.figure(figsize=(25,4))
for idx in np.arange(3):
    ax=fig.add_subplot(2,20/2,idx+1,xticks=[],yticks=[])
    plt.imshow(np.transpose(train_features_np[idx],(1,2,0)))
    
    

