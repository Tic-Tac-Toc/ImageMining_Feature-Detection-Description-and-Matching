import numpy as np
import cv2

from matplotlib import pyplot as plt

#Read grayscale image and conversion to float64
img=np.float64(cv2.imread('../Image_Pairs/Nuts1.png',0))

(h,w) = img.shape
print("Image dimension:",h,"rows x",w,"columns")

#Direct method
t1 = cv2.getTickCount()
img2 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
for y in range(1,h-1):
  for x in range(1,w-1):
    val = img[y+1, x]  - img[y-1, x]
    img2[y,x] = val
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Direct method:",time,"s")

plt.subplot(221)
plt.imshow(img2,cmap = 'gray')
plt.title('Convolution - Direct method')

#Method filter2D
t1 = cv2.getTickCount()
kernel = np.array([[-1], [0], [1]])
img3 = cv2.filter2D(img,-1,kernel)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Method filter2D :",time,"s")

plt.subplot(222)
plt.imshow(img3,cmap = 'gray')
plt.title('Convolution - filter2D')

rho = 5

#Box sum image - direct method
t1 = cv2.getTickCount()
img4 = cv2.copyMakeBorder(img3,0,0,0,0,cv2.BORDER_REPLICATE)
for y in range(rho,h-rho):
  for x in range(rho,w-rho):
    sum = 0
    for i in range(x - rho, x + rho):
      for j in range(y - rho, y + rho):
        sum += img3[j][i]
    sum = sum / rho**2
    img4[y,x] = min(max(sum,0),255)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Box sum image:",time,"s")

plt.subplot(223)
plt.imshow(img4,cmap = 'gray')
plt.title('Box sum image - direct method')

#Box sum image - filter2D method
t1 = cv2.getTickCount()
kernel = np.ones((rho * 2 + 1, rho * 2 + 1)) / (rho * 2 + 1)**2
img3 = cv2.filter2D(img,-1,kernel)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Box sum image - filter2D method :",time,"s")

plt.subplot(224)
plt.imshow(img4,cmap = 'gray')
plt.title('Box sum image - filter2D method')

plt.show()
