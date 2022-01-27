
import os
import cv2
import math
from lsqFitting import *
import numpy as np

red = [50, 0, 200]
x1, y1, z1 = red
red_mod = math.sqrt(x1*x1+y1*y1+z1*z1)
angle_th = 30

dir = 'C:/FILES/downloads/balls/'

names = os.listdir(dir)
names = list(filter(lambda x: '.jpg' in x, names))

print(math.cos(np.pi*angle_th/180))

# This is for getting contours of all the objects
def getContours(data, th):
    #uplimit of the number of edge points
    edgeUplim = data.shape[0]*2+data.shape[1]*4
    def within(data, coor):     #verdict if a point is within the image domain
        return coor[0] >= 0 and coor[0] < data.shape[1] and\
                coor[1] >= 0 and coor[1] < data.shape[0]
    #return all points that belong to the zone that starts from point coor
    def getZone(data, mark, th, coor):
        cir = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]
        #First step, extract the contour of the zone
        edge = [coor]
        lastDir = 2
        isolated = False
        while len(edge) < edgeUplim:
            for k in range(-2, 6):
                dir = (lastDir + k)&0x07
                coor = [edge[-1][0] + cir[dir][0], edge[-1][1] + cir[dir][1]]
                if within(data, coor) and data[coor[1], coor[0]] > th:
                    edge.append(coor)
                    lastDir = dir
                    break
                if k == 5:
                    isolated = True
            if isolated or (len(edge) > 7 and edge[-1] == edge[1] and edge[-2] == edge[0]):
                break
        for x, y in edge:
            mark[y,x] = 255

        return edge, mark

    contours = []
    mark = np.zeros(data.shape)
    for y in range(0, data.shape[0], 10):
        for x in range(1, data.shape[1]):
            if data[y,x-1] <= th and data[y,x] > th and mark[y][x] == 0:
                edge, mark = getZone(data, mark, th, [x, y])
                contours.append(edge)
    return contours

hs = 25 # half stride
n = hs*2 + 1
rad = 270
for img_idx, name in enumerate(names):
    path = os.path.join(dir, name)
    img = cv2.imread(path)
    h, w = img.shape[:2]
    mask = np.zeros(img.shape[:2]).astype(np.uint8)
    cont_img = np.zeros(img.shape[:2]).astype(np.uint8)
    cross_img = np.zeros((h*2, w*2)).astype(np.uint8)

    # matrices processing is much faster than pixel level processing
    imgr, imgg, imgb = img[:,:,0].astype(np.int32), img[:,:,1].astype(np.int32), img[:,:,2].astype(np.int32)
    img_mod = np.sqrt(imgr*imgr+imgg*imgg+imgb*imgb)
    v1_dot_v2 = x1*imgr+y1*imgg+z1*imgb
    cos_img = v1_dot_v2/(red_mod*img_mod+1)
    mask = ((cos_img>math.cos(np.pi*angle_th/180))*255).astype(np.uint8)

    contours = getContours(mask, 128)
    objects = [contour for contour in contours if len(contour) > 35*4]

    for object in objects:
        x, y = object[0]
        accx, accy, accxx, accyy, accxy = [x], [y], [x*x], [y*y], [x*y]
        llen = len(object)
        for x, y in object[1:]:
            accx.append(accx[-1]+x)
            accy.append(accy[-1]+y)
            accxx.append(accxx[-1]+x*x)
            accyy.append(accyy[-1]+y*y)
            accxy.append(accxy[-1]+x*y)
        ks, kks, angles = [], [], []
        for i in range(len(object)):
            sumx = accx[(i+hs)%llen] - accx[i-hs-1]
            sumy = accy[(i+hs)%llen] - accy[i-hs-1]
            sumxx = accxx[(i+hs)%llen] - accxx[i-hs-1]
            sumyy = accyy[(i+hs)%llen] - accyy[i-hs-1]
            sumxy = accxy[(i+hs)%llen] - accxy[i-hs-1]
            sumx = sumx if sumx >= 0 else sumx+accx[-1]
            sumy = sumy if sumy >= 0 else sumy+accy[-1]
            sumxx = sumxx if sumxx >= 0 else sumxx+accxx[-1]
            sumyy = sumyy if sumyy >= 0 else sumyy+accyy[-1]
            sumxy = sumxy if sumxy >= 0 else sumxy+accxy[-1]
            denox, denoy = n*sumxx - sumx*sumx, n*sumyy - sumy*sumy
            if denox > denoy:
                k = (n * sumxy - sumx * sumy) / denox if denox != 0 else 1000
            else:
                k = denoy / (n * sumxy - sumx * sumy) if (n * sumxy - sumx * sumy) != 0 else 1000
            kk = -1/k if abs(k) > 0.001 else 1000
            if abs(kk) < 1:
                b = sumy/n - kk*sumx/n
                ex = int(sumx/n+0.5)
                '''
                for x in range(ex-rad, ex+rad):
                    y = int(kk*x + b + 0.5)
                    xx, yy = x+w//2, y+h//2
                    if xx >= 0 and xx < w*2 and yy>=0 and yy < h*2:
                        None
                '''
            else:
                kk = 1/kk
                b = sumx/n - kk*sumy/n
                ey = int(sumy/n+0.5)
                '''
                for y in range(ey-rad, ey+rad):
                    x = int(kk*y + b+0.5)
                    xx, yy = x+w//2, y+h//2
                    if xx >= 0 and xx < w*2 and yy>=0 and yy < h*2:
                        None
                '''
            angle = math.atan(k)*180/np.pi
            ks.append(k)
            kks.append(kk)
            angles.append(angle)
        arc_pixs = np.zeros(len(object))
        for i in range(len(object)):
            pre_i, rear_i = i-7, (i+7)%llen
            delta_angle = angles[rear_i] - angles[pre_i]
            delta_angle = min(abs(delta_angle), abs(delta_angle+180), abs(delta_angle-180))
            x, y = object[i]
            #cont_img[y,x] = 255 if delta_angle < 25 else 80
            arc_pixs[i] = 1 if delta_angle < 25 else 0
        arcs = []
        for i in range(len(object)):
            if arc_pixs[i] == 1 and arc_pixs[i-1] == 0:
                for j in range(i+1, 2*len(object)):
                    if arc_pixs[j%llen] == 0:
                        arc = object[i:j] if j <= llen else object[i:] + object[:j%llen]
                        arcs.append(arc)
                        i = j+1
                        break
        arcs = [arc for arc in arcs if len(arc)>80]
        for arc in arcs:
            for x, y in arc:
                cont_img[y, x] = 80

        ellipses = []
        for arc in arcs:
            params = lsqEllipse(arc)
            if params is None:
                continue
            xc, yc, a, b, theta = params
            if a < 30 or b < 30 or a > 700 or b > 700:
                continue
            if a/b > 1.5 or b/a > 1.5:
                continue
            ellipses.append(params+[len(arc)])
        ellipses = sorted(ellipses, key=lambda x:x[-1])
        if len(ellipses) == 0:
            continue
        ellipse = ellipses[-1]
        xc, yc, a, b, theta, _ = ellipse

        coors = ellipseCoordits(xc, yc, a, b, theta, 1000)
        for x, y in coors:
            x, y = int(x + 0.5), int(y + 0.5)
            if x >= 0 and x < w and y >= 0 and y < h:
                cont_img[y, x] = 255
                img[y,x] = np.array(np.zeros(3).astype(np.uint8))

    print(img_idx, ': ', name)
    cv2.imshow('org', img)
    cv2.imshow('mask', mask)
    cv2.imshow('cont', cont_img)
    #cv2.imshow('cross', cross_img)
    #cv2.imshow('center', cent_img)
    cv2.waitKey(800)
