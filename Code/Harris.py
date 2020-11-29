import numpy as np
import cv2

from matplotlib import pyplot as plt

ALPHA = 0.06
SIGMA = 2

def gaussien_blur(sig):
    size = int(3 * sig)
    if size % 2 == 0:
        size += 1
    T = np.zeros((size,size))
    for x in range(size):
        for y in range(size):
            X = ((x-(size-1)/2))
            Y = ((y-(size-1)/2))
            T[x][y] = ((1)/(2 * np.pi * sig**2)) * np.exp(-   (X**2 + Y**2)  / (2*sig**2))
    sum = T.sum()
    T /= sum
    return(T)

def gaussian_kernel(IsX, size, sig):
    size = int(size)
    if size % 2 == 0:
        size += 1
    T = np.zeros((size,1))
    for x in range(size):
        X = ((x-(size-1)/2))
        T[x] = ((1)/(np.sqrt(2 * np.pi) * sig)) * (-2*X) / (2*sig**2) * ( np.exp(-   (X**2) / (2*sig**2)))
    
    if IsX:
        return (np.transpose(T))
    else: 
        return (T)

def first_order_derivate(I, X, sig):
    kernel = gaussian_kernel(X, 3*sig, sig)
    return cv2.filter2D(I,-1,kernel)

#W : list of points that contains W
def compute_autocorrelationmatrix(SX, SY, SXY, x, y):
    M = np.zeros((2,2))
    M[0][0] = SX[y][x]
    M[1][1] = SY[y][x]
    M[0][1] = SXY[y][x]
    M[1][0] = SXY[y][x]
    return M

#Reading grayscale image and conversion to float64
img=np.float64(cv2.imread('../Image_Pairs/Graffiti0.png',cv2.IMREAD_GRAYSCALE))
(h,w) = img.shape
print("Dimension of image:",h,"rows x",w,"columns")
print("Type of image:",img.dtype)

#Beginning of calculus
t1 = cv2.getTickCount()
Theta = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)

# Put here Harris interest function calculation

I_derivateX = first_order_derivate(img, True, SIGMA)
I_derivateY = first_order_derivate(img, False, SIGMA)
I_XX = I_derivateX**2
I_YY = I_derivateY**2
I_XY = I_derivateX * I_derivateY

SX = cv2.filter2D(I_XX, -1, gaussien_blur(2*SIGMA))
SY = cv2.filter2D(I_YY, -1, gaussien_blur(2*SIGMA))
SXY = cv2.filter2D(I_XY, -1, gaussien_blur(2*SIGMA))

for y in range(h):
    for x in range(w):
        M = compute_autocorrelationmatrix(SX, SY, SXY, x, y)
        det, trace = np.linalg.det(M), np.trace(M)
        Theta[y][x] = det - ALPHA * trace**2

# Computing local maxima and thresholding
Theta_maxloc = cv2.copyMakeBorder(Theta,0,0,0,0,cv2.BORDER_REPLICATE)
d_maxloc = 3
seuil_relatif = 0.01
se = np.ones((d_maxloc,d_maxloc),np.uint8)
Theta_dil = cv2.dilate(Theta,se)
#Suppression of non-local-maxima
Theta_maxloc[Theta < Theta_dil] = 0.0
#Values to small are also removed
Theta_maxloc[Theta < seuil_relatif*Theta.max()] = 0.0
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("My computation of Harris points:",time,"s")
print("Number of cycles per pixel:",(t2 - t1)/(h*w),"cpp")

plt.subplot(231)
plt.imshow(img,cmap = 'gray')
plt.title('Original image')

plt.subplot(232)
plt.imshow(Theta,cmap = 'gray')
plt.title('Harris function')

se_croix = np.uint8([[1, 0, 0, 0, 1],
[0, 1, 0, 1, 0],[0, 0, 1, 0, 0],
[0, 1, 0, 1, 0],[1, 0, 0, 0, 1]])
Theta_ml_dil = cv2.dilate(Theta_maxloc,se_croix)
#Re-read image for colour display
Img_pts=cv2.imread('../Image_Pairs/Graffiti0.png',cv2.IMREAD_COLOR)
(h,w,c) = Img_pts.shape
print("Dimension of image:",h,"rows x",w,"columns x",c,"channels")
print("Type of image:",Img_pts.dtype)
#Points are displayed as red crosses
Img_pts[Theta_ml_dil > 0] = [255,0,0]
plt.subplot(233)
plt.imshow(Img_pts)
plt.title('Harris points')

plt.subplot(234)
plt.imshow(I_derivateX,cmap = 'gray')
plt.title('X derivate')

plt.subplot(235)
plt.imshow(I_derivateY,cmap = 'gray')
plt.title('Y derivate')

Img_pts=cv2.imread('../Image_Pairs/Graffiti0.png',cv2.IMREAD_COLOR)
Img_pts=cv2.filter2D(Img_pts, -1, gaussien_blur(SIGMA))
#Points are displayed as red crosses
Img_pts[Theta_ml_dil > 0] = [255,0,0]
plt.subplot(236)
plt.imshow(Img_pts)
plt.title('Gaussian kernel - SIGMA = ' + str(SIGMA))

plt.show()
