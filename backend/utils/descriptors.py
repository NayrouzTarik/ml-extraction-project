import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog

def extract_hu_moments(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()
    return hu

def extract_orientation_histogram(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
    gy = cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    mag, angle = cv2.cartToPolar(gx,gy)
    hist, _ = np.histogram(angle, bins=16, range=(0,2*np.pi))
    return hist

def extract_hog(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray, pixels_per_cell=(16,16), cells_per_block=(2,2), orientations=9, visualize=True)
    return features

def extract_lbp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=np.arange(0, 10), density=True)
    return hist

def extract_color_hist(img):
    hist = []
    for i in range(3):
        h = cv2.calcHist([img],[i],None,[256],[0,256]).flatten()
        hist.append(h)
    return np.concatenate(hist)
