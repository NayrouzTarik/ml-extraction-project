import cv2
import numpy as np
import io
import base64
import matplotlib.pyplot as plt

def extract_color_hist(img):
    hist = []
    for i in range(3):
        h = cv2.calcHist([img],[i],None,[256],[0,256]).flatten()
        hist.append(h)
    return np.concatenate(hist)

def visualize_color_hist(img):
    colors = ('b', 'g', 'r')
    plt.figure(figsize=(4, 3))
    plt.title("Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    
    for i, col in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
        
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64
