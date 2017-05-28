import pandas as pd
import re
import os
import numpy as np
from calcIoU import box_iou

probs = {}
idlist = []
pattern = re.compile(r'\d{4}')
truths = {}
path = '/home/dj/data/CarPlane/label2'

with open("/home/dj/data/CarPlane/test.txt") as f:
    for line in f:
        #print (line)
        num = re.findall(pattern, line)
        num  = num[0]
        #print (num)
        if (int(num) > 510):
            idlist.append(num)

            if (num not in truths):
                truths[num] = []
            name = 'P' + num + '.txt'
            with open(path + '/' + name) as f5:
                for line in f5:
                    line = line.strip()
                    #print('line', line)
                    linelist = line.split('\t')

                    #print('linelist', linelist)
                    truth = linelist[9: 13]
                    #print('truth', truth)
                    for index, item in enumerate(truth):
                        truth[index] = float(item)
                    #print('truth2', truth)
                    xmin, ymin, w, h = truth[0], truth[1], truth[2], truth[3]
                    xmax = xmin + w
                    ymax = ymin + h
                    truth[2], truth[3] = xmax, ymax
                    truths[num].append(truth)
            f5.close()


#print(truths)     
# with open("comp4_det_test_car.txt") as f2:
#     for line in f2:
#         #print(line)
#         line = line.strip()
#         linelist = line.split(' ')
#         num = re.findall(pattern, linelist[0])
#         num  = num[0]
#         #print('num', num)
#         if num not in probs:
#             probs[num] = []
#         prob = linelist[1:6]
#         #print ('prob', prob)
#         for index, item in enumerate(prob):
#             prob[index] = float(item)
#         #print ('prob', prob)
#         probs[num].append(prob)
with open("comp4_det_test_plane.txt") as f3:
    for line in f3:
        line = line.strip()
        linelist = line.split(' ')
        num = re.findall(pattern, linelist[0])
        num  = num[0]
        if num not in probs:
            probs[num] = []
        prob = linelist[1:6]
        for index, item in enumerate(prob):
            prob[index] = float(item)
        probs[num].append(prob)


# while True:
#      pass
def calcu_pr(thresh):
    proposals = 0.0
    #thresh = 0.25
    iouthresh = 0.5
    total = 0.0
    avg_iou = 0
    correct = 0.0
    for id in idlist:
        #print(item)
        #print(probs[item]
        itruths = truths[id]
        iprobs = probs[id]
        print('id:', id)
        for item in iprobs:
            if item[0] > thresh:
                proposals = proposals + 1
        for index, itruth in enumerate(itruths):
            total = total + 1
            best_iou = 0
            #print ('itruth', itruth)
            for iprob in iprobs:
                ibox = iprob[1:5]
                #print ('itruth', itruth)
                #print ('iprob', iprob)
                #print ('ibox', ibox)
                iou = box_iou(ibox, itruth)
                if ((iprob[0] > thresh) and (iou > best_iou)):
                    best_iou = iou
                #print('iou:', iou)
            avg_iou = avg_iou + best_iou
            if (best_iou >= iouthresh):
                correct = correct + 1
    #print('correct', correct)
    #print('proposals:', proposals)
    #print('total:', total)
    if (proposals == 0):
        return [0, 0]
    precision = float(correct/proposals)
    recall = float(correct/total)
    #print('precision:', precision)
    #print('recall', recall)
    return [precision, recall]
PRC = []
for i in range(21):
    thresh = float(i)/21
    point = calcu_pr(thresh)
    if (point[0] == 0):
        break
    PRC.append(point)
PRC = np.array(PRC)
np.savetxt("plane_pr.txt", PRC)

    
