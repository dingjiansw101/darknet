import numpy as np
def add(a, b):
    
    return np.maximum(a + b, 0)


def box_iou(a, b):
    #xmin1, ymin1, xmax1, ymax1 = a[0], a[1], a[2], a[3]
    #xmin2, ymin2, xmax2, ymax2 = b[0], b[1], b[2], b[3]
    lu = np.maximum(a[:2], b[:2])
    rd = np.minimum(a[2:], b[2:])
    intersection = np.maximum(0.0, rd - lu)
    inter_square = intersection[0]*intersection[1]
    
    square1 = (a[2] - a[0])*(a[3] - a[1])
    square2 = (b[2] - b[0])*(b[3] - b[1])
    print(inter_square)
    union_square = np.maximum(square1 + square2 - inter_square, 1e-10)
    print(union_square)
    iou = inter_square/union_square
    if (iou < 0):
        iou = 0
    elif(iou > 1):
        iou = 1
    return iou
    