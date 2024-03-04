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

foobar = importlib.import_module("detectors.Visual.UIED-master.run_single")
pre = importlib.import_module("detectors.Visual.UIED-master.detect_compo.lib_ip.ip_preprocessing")
draw = importlib.import_module("detectors.Visual.UIED-master.detect_compo.lib_ip.ip_draw")
det = importlib.import_module("detectors.Visual.UIED-master.detect_compo.lib_ip.ip_detection")
file = importlib.import_module("detectors.Visual.UIED-master.detect_compo.lib_ip.file_utils")
Compo = importlib.import_module("detectors.Visual.UIED-master.detect_compo.lib_ip.Component")
ip = importlib.import_module("detectors.Visual.UIED-master.detect_compo.ip_region_proposal")
Congfig = importlib.import_module("detectors.Visual.UIED-master.config.CONFIG_UIED")

# Suppress all warnings
warnings.filterwarnings("ignore")
# os.chdir("/Users/arunkrishnavajjala/Documents/GMU/PhD/P3/UIEmbedding")
uiedDir = "./detectors/Visual/UIED-master/data/output/ip"

def runUIED(image):
    foobar.runSingle(image)
    files = []
    
    imagePath = ""
    jsonPath = ""
    for root, dirs, files_in_dir in os.walk(uiedDir):
        print(files_in_dir)
        for file_name in files_in_dir:

            if 'Store' not in file_name:
                if '.json' in file_name:
                    jsonPath = os.path.join(uiedDir, file_name)
                if '.jpg' in file_name:
                    imagePath = os.path.join(uiedDir, file_name)
    allBoxes = getBoxes(jsonPath)
    return [allBoxes, imagePath, jsonPath]


def getBoxes(json_path):
    # Read the JSON file
    with open(json_path, 'r') as file:
        json_data = file.read()

    # Parse the JSON data
    data = json.loads(json_data)

    # Extract the bounding box information
    bounding_boxes = []
    compos = data['compos']
    largestArea = 0
    allBoxes = []

    allBoxDict = {}

    for compo in compos:
        column_min = compo["column_min"]
        row_min = compo["row_min"]
        column_max = compo["column_max"]
        row_max = compo["row_max"]
        # Extract the region of interest from the image
        box = (column_min, row_min, column_max, row_max)
        boxArea = calculate_box_area(box)
        # if boxArea > 500:
        #     allBoxes.append(box)
        allBoxDict[tuple(box)] = boxArea

    # Sort the dictionary items by values in descending order
    sorted_items = sorted(allBoxDict.items(), key=lambda x: x[1], reverse=True)

    # Create a new dictionary with the sorted items
    sorted_dict = dict(sorted_items)
    print("Number of extracted Elements: " + str(len(sorted_dict)))
    counter= 0
    for i in sorted_dict.keys():
        if counter < 25:
            allBoxes.append(i)
            counter += 1
        else:
            break


    return(allBoxes)


def calculate_box_area(box):
    column_min, row_min, column_max, row_max = box
    width = column_max - column_min
    height = row_max - row_min
    area = width * height
    return area







