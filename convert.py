# -*- coding:utf-8  -*-
from PIL import Image
import os
import string
from matplotlib import pyplot as plt


path=''
filelist=os.listdir(path)

for file in filelist:
    whole_path = os.path.join(path, file)
    print(whole_path)
    img = Image.open(whole_path)  
    img = img.resize((512,512)).convert("RGB")
    save_path = ''
    #img.save(save_path + img1)
    img.save(os.path.join(save_path,file))
