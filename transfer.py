import json
import numpy as np
import os
import random
from PIL import Image
  
import glob

def crop(img,x,y):
    if (x + 16) > img.shape[1] or (y + 14) > img.shape[0]:
        return None
    return img[y:y + 16, x:x + 16,:]

def crop_random(img):
    x=int((img.shape[1]-16)*random.random())
    y=int((img.shape[0]-16)*random.random())
    return crop(img,x,y)


def merge(blank,img,x,y):
    blank[x:x+img.shape[1],y:y+img.shape[0],:]=img[:img.shape[1],:img.shape[0],:]
        
    

def fill_image(blank,img):
    if blank.shape[0]!=blank.shape[1]:
        exit("error: it should be square!")
    
    co=int(blank.shape[0]/16)
    for i in range(co):
        for j in range(co):
            merge(blank,crop_random(img),i*16,j*16)
    return blank

def result_image(img,h,w):
    d=img.shape[2]
    blank=np.zeros(h*w*d).astype(np.float32).reshape((h,w,d))
    fill_image(blank,img)
    return blank


root="datasets/style"
file=sorted(glob.glob(os.path.join(root) + "/*.*"))
img=Image.open(file[0])

img = np.asarray(img)
for i in range(6288):
    result=result_image(img,256,256)
    result = Image.fromarray(result.astype('uint8')).convert('RGB')
    result.save("datasets/styletransfer/trainA/"+str(i)+".jpg")
    

