from __future__ import print_function
from numba import jit
import numpy as np
import cv2 as cv
import random as r
alpha = 0.5
try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3
print(''' Simple Linear Blender
-----------------------
* Enter alpha [0.0-1.0]: ''')
input_alpha = float(raw_input().strip())
if 0 <= alpha <= 1:
    alpha = input_alpha
# [load]
#im_gray = cv.imread('mask.jpg', cv.IMREAD_GRAYSCALE)
#(thresh, im_bw) = cv.threshold(im_gray, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
#cv.imwrite('./2.png', im_bw)

#im_gray2 = cv.imread('preview16.jpeg', cv.IMREAD_GRAYSCALE)
#(thresh1, im_bw2) = cv.threshold(im_gray2, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
#cv.imwrite('./bw_image2.png', im_bw2)
@jit(parallel=True)
def func():
    for i in range(3862):
        for j in range(2):
            add = r.randrange(1,19)
            src1 = cv.imread(cv.samples.findFile('./imgs/Test ('+str(i + 1)+').jpg'))
            src2 = cv.imread(cv.samples.findFile('mask'+str(add)+'.jpg'))

            src2_resized = cv.resize(src2, (src1.shape[1], src1.shape[0]))

            # [load]
            if src1 is None:
                print("Error loading src1")
                exit(-1)
            elif src2 is None:
                print("Error loading src2")
                exit(-1)
            # [blend_images]
            beta = (1.0 - alpha)
            dst = cv.addWeighted(src1, alpha, src2_resized, beta, 0.0)
            #dst = np.uint8(alpha*(src1)+beta*(src2))
            # [blend_images]
            # [display]
            #cv.imshow('dst', dst)
            #cv.waitKey(0)
            cv.imwrite("./Set2/out"+str(i)+".jpg", dst);
        print(str(i)+' Completed.')
# [display]
#cv.destroyAllWindows()
func()
numba.cuda.profile_stop()