import xml.etree.ElementTree as ET
import xml.dom.minidom
import cv2
from PIL import Image
import os
import pytesseract
import importlib
import json
import clip
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
from torchvision.transforms.functional import crop
from scipy.spatial import Delaunay, ConvexHull, Voronoi
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import gudhi as gd
# Use birth_death_pairs to create an embedding, for example, using t-SNE
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec
from sklearn.manifold import spectral_embedding
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from transformers import BertModel, BertTokenizer
import warnings
from PIL import ImageEnhance

from PIL import Image, ImageEnhance, ImageOps, ImageFilter

foobar = importlib.import_module("detectors.Visual.UIED-master.run_single")
pre = importlib.import_module("detectors.Visual.UIED-master.detect_compo.lib_ip.ip_preprocessing")
draw = importlib.import_module("detectors.Visual.UIED-master.detect_compo.lib_ip.ip_draw")
det = importlib.import_module("detectors.Visual.UIED-master.detect_compo.lib_ip.ip_detection")
file = importlib.import_module("detectors.Visual.UIED-master.detect_compo.lib_ip.file_utils")
Compo = importlib.import_module("detectors.Visual.UIED-master.detect_compo.lib_ip.Component")
ip = importlib.import_module("detectors.Visual.UIED-master.detect_compo.ip_region_proposal")
Congfig = importlib.import_module("detectors.Visual.UIED-master.config.CONFIG_UIED")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def bigImageEmbedding(image):
    bigimage = Image.open(image)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(bigimage)
    enhanced_image = enhancer.enhance(2.625)
    
    # Convert to grayscale
    grayscale_image = ImageOps.grayscale(enhanced_image)
    #inverted_img = ImageOps.invert(grayscale_image)
    
    # hist = grayscale_image.histogram()
    # most_common_color_index = np.argmax(hist)
    
    # # Get the most common color value
    # most_common_color_value = most_common_color_index // 256
    
    # # Create a mask identifying pixels with the most common color
    # mask = np.array(grayscale_image) == most_common_color_value
    
    # # Apply the mask to the original image to delete those pixels
    # img_array = np.array(bigimage)
    # img_array[mask] = [57, 255, 20] 

    # Convert the modified array back to an image
    #modified_img = Image.fromarray(img_array)

    # Apply Gaussian blur

    #smoothed_image = grayscale_image.filter(ImageFilter.SMOOTH)

    rank_filtered_image = grayscale_image.filter(ImageFilter.RankFilter(size=3, rank=3))
    
    with torch.no_grad():
        big_image_features = model.encode_image(preprocess(rank_filtered_image).unsqueeze(0).to(device))
    
    big_image_embedding = big_image_features[0].cpu().numpy()
    
    return big_image_embedding
    

def smallImageEmbeddings(allBoxes, image):
    points = []
    midpoints = []
    for i in allBoxes:
        # Load and preprocess the image
        image = Image.open(image)

        cropped = image.crop(i)
        image_input = preprocess(cropped).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            #text_features = model.encode_text(text_inputs)

        image_embedding = image_features[0].cpu().numpy()
        points.append(tuple(image_embedding))
        midpoints.append(get_midpoint(i))

    if jsonPath != "" and imagePath != "":  
        #print("deleted images") 
        os.remove(jsonPath)
        os.remove(imagePath)
    print(len(points))
    print(len(midpoints))


def get_midpoint(bounding_box):
    x_min, y_min, x_max, y_max = bounding_box
    mid_x = (x_min + x_max) / 2
    mid_y = (y_min + y_max) / 2
    return (mid_x, mid_y)





















