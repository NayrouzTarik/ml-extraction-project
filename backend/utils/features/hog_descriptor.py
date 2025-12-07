import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure
import io
import base64
import matplotlib.pyplot as plt

def extract_hog(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = hog(gray, pixels_per_cell=(16,16), cells_per_block=(2,2), orientations=9)
    return features

def visualize_hog(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, hog_image = hog(gray, pixels_per_cell=(16,16), cells_per_block=(2,2), orientations=9, visualize=True)
    
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
    plt.figure(figsize=(3, 3))
    plt.imshow(hog_image_rescaled, cmap='gray')
    plt.title("HOG Features")
    plt.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')
