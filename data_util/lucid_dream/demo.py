from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import cv2
from PIL import Image
from PIL import ImagePalette
import numpy as np
from patchPaint import paint
from lucidDream import dreamData

Iorg=cv2.imread('example2/img.jpg')
Morg=Image.open('example2/gt.png')
palette=Morg.getpalette()
if palette is None:
    palette = [0, 0, 0, 150, 0, 0, 0, 150, 0, 0, 0, 150]

bg=paint(Iorg,np.array(Morg),False)
cv2.imwrite('example2/bg.jpg',bg)

im_1,gt_1,im_2,gt_2=dreamData(Iorg,np.array(Morg),bg,True)
print('gt1 list: {}'.format(np.unique(gt_1)))
print('gt2 list: {}'.format(np.unique(gt_2)))

# Image 1 in this pair.
cv2.imwrite('example2/gen1.jpg',im_1)

# Mask for image 1.
gtim1=Image.fromarray(gt_1,'P')
gtim1.putpalette(palette)
gtim1.save('example2/gen1.png')

# Image 2 in this pair.
cv2.imwrite('example2/gen2.jpg',im_2)

# Mask for image 2.
gtim2=Image.fromarray(gt_2,'P')
gtim2.putpalette(palette)
gtim2.save('example2/gen2.png')
