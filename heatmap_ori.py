# -*- coding: utf-8 -*-


import torch, sys
import numpy as np
probs = torch.load('probs-50.pth')
test = torch.load('test-50.lib')
grids = test['grid']
slides = test['slides']
targets = test['targets']

temp = []

for i in range(len(probs)//1849):
    #print(i)
    temp.append(probs[i*1849:(i+1)*1849])

heatmaps = []
for i in temp:
    t = []
    for j in range(43):
        t.append(i[43*j:43*(j+1)])
    heatmaps.append(t)
heatmaps = np.array(heatmaps)

import cv2
for i in range(len(slides)):
    print(slides[i],len(temp[i]),len(heatmaps[i]))
    sys.stdout.write('Processing: [{}/{}]\r'.format(i+1, len(slides)))
    sys.stdout.flush()
    img = cv2.imread('RGB/%s'%slides[i])
    heatmap = heatmaps[i]
    print(heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_mask=np.uint8(255 * heatmap)
    
    gray1 = np.copy(heatmap)
    rows, cols = gray1.shape[:2]
    for row in range(rows):
        for col in range(cols):
            if gray1[row, col] >= 0.5:
                gray1[row, col] = 1
            else:
                gray1[row, col] = 0
    gray1 = np.uint8(255 * gray1)
    

    cv2.imwrite('./test/%s_mymask.jpg' % (slides[i]), gray1)    

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)   
    superimposed_img = img * 0.5 + heatmap * 0.5
    cv2.imwrite('./test/%s.jpg'%(slides[i]), superimposed_img)      
    cv2.imwrite('./test/%s_heatmapmask.jpg'%(slides[i]), heatmap_mask)   

    
    img1 = cv2.imread(r"./test/%s.jpg"%(slides[i]))
    img2 = cv2.imread(r"./test/%s_mymask.jpg"%(slides[i]))
    dst = cv2.addWeighted(img1, 0.4, img2, 0.6, 0)
    cv2.imwrite("./test/%s_all.jpg"%(slides[i]), dst)    # The final result of storing your own binarized map and heat map superimposed on each other
    # cv2.imwrite('/home/amaxv1004/Data/LXY/DN-NET/dataset/ori/mil/internal/heatmap/positives/%s.jpg' % (slides[i]),superimposed_img)
    # cv2.imwrite('/home/amaxv1004/Data/LXY/DN-NET/dataset/ori/mil/internal/mask/positives/%s' % (slides[i]),heatmap_mask)

