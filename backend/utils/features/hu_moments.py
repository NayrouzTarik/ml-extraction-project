import cv2
import numpy as np
import io
import base64
import matplotlib.pyplot as plt

def extract_hu_moments(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(thresh)
    hu = cv2.HuMoments(moments).flatten()
    return hu

def visualize_hu_moments(img):
    """Visualize the contour used for Hu Moments"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    vis_img = img.copy()
    cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 2)
    
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(3, 3))
    plt.imshow(vis_img)
    plt.title("Contour for Moments")
    plt.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')
