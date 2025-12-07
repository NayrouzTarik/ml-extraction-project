import cv2
import numpy as np
import io
import base64
import matplotlib.pyplot as plt

def extract_orientation_histogram(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, angle = cv2.cartToPolar(gx, gy)
    hist, _ = np.histogram(angle, bins=16, range=(0, 2*np.pi))
    return hist

def visualize_orientation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, _ = cv2.cartToPolar(gx, gy)
    
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    mag = np.uint8(mag)
    
    plt.figure(figsize=(3, 3))
    plt.imshow(mag, cmap='inferno')
    plt.title("Gradient Magnitude")
    plt.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')
