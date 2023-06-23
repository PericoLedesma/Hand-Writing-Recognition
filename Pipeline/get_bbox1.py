import cv2 as cv
import numpy as np
import os
import drop_fall as acid
from PIL import Image

def pt_ln_dist(point,ln_pt1,ln_pt2):
    vec1 = ln_pt1 - point
    vec2 = ln_pt2 - point
    distance = np.abs(np.cross(vec1,vec2)) / np.linalg.norm(ln_pt1-ln_pt2)
    return distance

def split(char, img):
    """
    This function takes as input the information of the multi character bounding boxes
    and outputs the split characters by using the drop fall algorithm on convex points
    in the images.

    char: information of the character in the line image.
    img: the line image
    return characters: an array containing all the seperated characters from the multi character parts
    """
    _, img = cv.threshold(img, 50, 255, cv.THRESH_BINARY_INV)

    cropped = img[char[1]:char[1]+char[3], char[0]:char[0]+char[2]]
    cropped = np.dstack((cropped, cropped, cropped))
    myCaptcha = acid.SJTUCaptcha(cropped)
    height, width, kernels = cropped.shape
    characters = []

    if char[5] == [0]:
        s = myCaptcha.mine()
        characters.append(np.array(s[0]))
        characters.append(np.array(s[1]))
    else:
        paths = []
        img=Image.fromarray(cropped)
        for convex_point in char[5]:
            s = myCaptcha.mine_convex(convex_point[0] - char[0])
            split_points = myCaptcha.filter_end_route
            paths.append(split_points)

        start_route = []
        end_route = []
        for y in range(height):
            start_route.append((0, y))

        for y in range(height):
            end_route.append((width - 1, y))

        for path in paths:
            characters.append(myCaptcha.do_split(img, start_route, path))
            start_route = path
        characters.append(myCaptcha.do_split(img, start_route, end_route))


    return characters


def bbox(img):
    """
    img_path: the path of img
    return rects: [x, y, w, h, Type, convex]
    x, y, w, h: the bbox of the characters
    type: 0,single chara; 1, multi-chars with convex points; 2, multi-chars without convex points
    convex: the convex points' coordinate

    """
    img_d=img.copy()
    img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(img_grey, 50, 255, cv.THRESH_BINARY)
    img_w=binary.shape[1]
    img_h=binary.shape[0]

    k = np.ones((3, 3), np.uint8)
    bi_ero = cv.erode(binary, k, iterations=1)

    contours, hierarchy = cv.findContours(bi_ero, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours==[]:
        print("no contours")
        return ['#']

    minx=img.shape[1]-2
    maxx=0
    rects=[]
    for i ,c in enumerate(contours):
        defects=[]
        convex=[]
        x, y, w, h = cv.boundingRect(c)
        if x<minx:
            minx=x
        if (x+w)>maxx:
            maxx=x+w
        if maxx>img.shape[1]-1:
            maxx=img.shape[1]-2
        area_rect=w*h
        if area_rect>170:
            cv.drawContours(img_d,contours,i,(0,0,255),1)
            cv.rectangle(img_d, (x, y), (x + w, y + h), (0, 255, 0), 1)
            thr_up=y + h//3
            hull = cv.convexHull(c, returnPoints=False)
            try:
                defects = cv.convexityDefects(c, hull)
            except:
                continue
            Type=0
            if defects is not None:
                for d in range(defects.shape[0]):
                    s, e, f, d = defects[d, 0]
                    start = np.array(c[s][0])
                    end = np.array(c[e][0])
                    far = np.array(c[f][0])
                    dt=pt_ln_dist(far,start,end)
                    cv.circle(img_d, (x,(y + h//3)), 2, (255,0, 0), -1)
                    if (area_rect>1200):
                        if ((start[1]<thr_up)&(end[1]<thr_up)&(dt>15)) \
                            or ((area_rect>2500)&((start[1]<thr_up)|(end[1]<thr_up))&(abs(start[0]-end[0])>w/2))&((w/h)>0.8):

                            Type=1
                            convex.append(list(far))
                            start = (start[0], start[1])
                            end = (end[0], end[1])
                            far = (far[0], far[1])
                            cv.putText(img_d, '%d'%(w*h),(x,y+h+3),cv.FONT_HERSHEY_TRIPLEX,0.5, (0,0,255), 1)
                            cv.line(img_d, start,end, (0, 0, 255), 2)
                            cv.circle(img_d, start, 3, (255, 0, 0), -1)
                            cv.circle(img_d, end, 3, (255, 0,0 ), -1)
                            cv.circle(img_d, far, 3, (0, 0, 255), -1)
                            #img_d = split(x, y, x + w, y + h, img_d, far)
                            cv.rectangle(img_d, (x, y), (x + w, y + h), (200, 0, 200), 1)

            if ((w/h)>1.5)& (area_rect>1100)&(Type==0):
                Type=2
                convex=[0]
                cv.rectangle(img_d, (x, y), (x + w, y + h), (200, 0, 200), 1)
            if Type==0:
                convex=[0]
            rects.append([x, y, w, h, Type, convex])

    img_d=img_d[:,minx:maxx]
    #cv.imshow("line", img_d)
    return rects

def takeSecond(elem):
        return elem[0]
