import numpy as np
import os
import cv2

img_dir = '../sku'
for img in os.listdir(img_dir):
    image = cv2.imread(os.path.join(img_dir,img))
    res_image = cv2.resize(image,None,None,fx=0.5, fy=0.5)
    res_img_path =os.path.join(img_dir,img.replace('.','_cps.'))
    cv2.imwrite(res_img_path, res_image)