#coding:utf-8

#The basic implementation of the acid drop fall algorithm in this file is
#taken from the github page: https://github.com/lan2720/fuck-captcha.
#We added functionality such as a loss function and the ability to have multiple
#drop locations from which the best path is picked.

from PIL import Image
import numpy as np
from itertools import groupby

#coding: utf-8
import cv2 as cv
COLOR_RGB_BLACK = (0, 0, 0)
COLOR_RGB_WHITE = (255, 255, 255)

class SJTUCaptcha(object):
    def __init__(self, image):
        self._image = image

    def mine(self):
        img=Image.fromarray(self._image)
        split_images = self._drop_fall(img)
        return split_images

    def mine_convex(self, convex_point):
        img=Image.fromarray(self._image)
        split_images = self._drop_fall(img, convex_point)
        return split_images

    def _is_black(self, rgb):
        return True if rgb == COLOR_RGB_BLACK else False

    def _drop_fall(self, image, convex_point = None):
        width, height = image.size
        hist_width = [0]*width
        for x in range(width):
            for y in range(height):
                if self._is_black(image.getpixel((x, y))):
                    hist_width[x] += 1

        start_x = self._get_start_x(hist_width, convex_point)
        losses = []
        paths = []
        for start in start_x:

            start_route = []
            for y in range(height):
                start_route.append((0, y))

            end_route, loss = self._get_end_route(image, start, height)
            paths.append(end_route)
            losses.append(loss)

        self.end_route = paths[np.argmin(losses)]
        self.filter_end_route = [max(list(k)) for _, k in groupby(end_route, lambda x: x[1])]
        img1 = self.do_split(image, start_route, self.filter_end_route)

        start_route = list(map(lambda x: (x[0] + 1, x[1]), self.filter_end_route))

        end_route = []
        for y in range(height):
            end_route.append((width - 1, y))
        img2 = self.do_split(image, start_route, end_route)

        return [img1, img2]

    def _get_start_x(self, hist_width, convex_point):
        if convex_point == None:
            mid = int(len(hist_width)/2)
        else:
            mid = convex_point

        return np.arange(start=mid - 2, stop=mid + 2, step=1)

    def _get_end_route(self, image, start_x, height):
        left_limit = 0
        right_limit = image.size[0] - 1

        end_route = []
        cur_p = (start_x, 0)
        last_p = cur_p
        end_route.append(cur_p)
        loss = 0
        while cur_p[1] < (height - 1):
            sum_n = 0
            maxW = 0 # max Z_j*W_j
            nextX = cur_p[0]
            nextY = cur_p[1]
            for i in range(1, 6):
                curW = self._get_nearby_pixel_val(image, cur_p[0], cur_p[1], i, right_limit) * (6 - i)
                sum_n += curW
                if maxW < curW:
                    maxW = curW

            if sum_n == 0:
                maxW = 4

            if sum_n == 15:
                maxW = 6

            if maxW == 1:
                nextX = cur_p[0] - 1
                nextY = cur_p[1]
            elif maxW == 2:
                nextX = cur_p[0] + 1
                nextY = cur_p[1]
            elif maxW == 3:
                nextX = cur_p[0] + 1
                nextY = cur_p[1] + 1
            elif maxW == 5:
                nextX = cur_p[0] - 1
                nextY = cur_p[1] + 1
            elif maxW == 6:
                nextX = cur_p[0]
                nextY = cur_p[1] + 1
            elif maxW == 4:
                if nextX > cur_p[0]:
                    nextX = cur_p[0] + 1
                    nextY = cur_p[1] + 1

                if nextX < cur_p[0]:
                    nextX = cur_p[0]
                    nextY = cur_p[1] + 1

                if sum_n == 0:
                    loss += 10
                    nextX = cur_p[0]
                    nextY = cur_p[1] + 1
            else:
                raise Exception("get a wrong maxW, pls check")

            if last_p[0] == nextX and last_p[1] == nextY:
                if nextX < cur_p[0]:
                    maxW = 5
                    nextX = cur_p[0] + 1
                    nextY = cur_p[1] + 1
                else:
                    maxW = 3
                    nextX = cur_p[0] - 1
                    nextY = cur_p[1] + 1

            last_p = cur_p

            if nextX > right_limit:
                nextX = right_limit
                nextY = cur_p[1] + 1

            if nextX < left_limit:
                nextX = left_limit
                nextY = cur_p[1] + 1

            cur_p = (nextX, nextY)
            if sum_n != 0:
                loss += 1

            end_route.append(cur_p)
        return end_route, loss

    def _get_nearby_pixel_val(self, image, cx, cy, j, right):
        cx = int(cx)
        cy = int(cy)
        if cx >= right - 1:
            cx = right - 1
        if j == 1:
            return 0 if self._is_black(image.getpixel((cx - 1, cy + 1))) else 1
        elif j == 2:
            return 0 if self._is_black(image.getpixel((cx, cy + 1))) else 1
        elif j == 3:
            return 0 if self._is_black(image.getpixel((cx + 1, cy + 1))) else 1
        elif j == 4:
            return 0 if self._is_black(image.getpixel((cx + 1, cy))) else 1
        elif j == 5:
            return 0 if self._is_black(image.getpixel((cx - 1, cy))) else 1
        else:
            raise Exception("what you request is out of nearby range")

    def do_split(self, source_image, starts, filter_ends):
        left = starts[0][0]
        top = starts[0][1]
        right = filter_ends[0][0]
        bottom = filter_ends[0][1]

        for i in range(len(starts)):
            left = min(starts[i][0], left)
            top = min(starts[i][1], top)
            right = max(filter_ends[i][0], right)
            bottom = max(filter_ends[i][1], bottom)

        width = right - left + 1
        height = bottom - top + 1

        image = Image.new('RGB', (width, height), COLOR_RGB_WHITE)

        for i in range(height):
            start = starts[i]
            end = filter_ends[i]
            for x in range(start[0], end[0]+1):
                if self._is_black(source_image.getpixel((x, start[1]))):
                    image.putpixel((x - left, start[1] - top), COLOR_RGB_BLACK)

        return image
