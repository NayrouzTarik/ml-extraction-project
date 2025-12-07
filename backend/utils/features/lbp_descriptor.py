import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import io
import base64
import matplotlib.pyplot as plt

def extract_lbp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=np.arange(0, 10), density=True)
    return hist

def visualize_lbp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    
    plt.figure(figsize=(3, 3))
    plt.imshow(lbp, cmap='gray')
    plt.title("LBP Texture")
    plt.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')
