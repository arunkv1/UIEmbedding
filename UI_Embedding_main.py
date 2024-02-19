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

from UIEDComp import runUIED
from imageEmbedding import bigImageEmbedding, smallImageEmbeddings, get_midpoint
from textEmbedding import makeTextEmbedding
from graphCreation import makeGraph, getConnections

warnings.filterwarnings("ignore")
os.chdir("/Users/arunkrishnavajjala/Documents/GMU/PhD/P3/UIEmbedding")
uiedDir = "/Users/arunkrishnavajjala/Documents/GMU/PhD/P3/UIEmbedding/detectors/Visual/UIED-master/data/output/ip/"

def makeEmbedding(image, embeddingType):
    big_image = image
    image = image
    textImage = image
    uiedResult = runUIED(image)
    allBoxes = uiedResult[0]

    big_image_embedding = bigImageEmbedding(big_image)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    imagePath = uiedResult[2]
    jsonPath = uiedResult[1]
    points = []
    
    midpoints = []
    textEmbeddings = []
    image = Image.open(big_image)
    for i in allBoxes:
        # Load and preprocess the image
        cropped = image.crop(i)
        cropTextEmbed = makeTextEmbedding(textImage)
        image_input = preprocess(cropped).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            #text_features = model.encode_text(text_inputs)
        image_embedding = image_features[0].cpu().numpy()
        points.append(tuple(image_embedding))
        midpoints.append(get_midpoint(i))
        textEmbeddings.append(cropTextEmbed)
    if jsonPath != "" and imagePath != "":  
        #print("deleted images") 
        os.remove(jsonPath)
        os.remove(imagePath)

    big_textEmbedding = makeTextEmbedding(textImage)
    GRAPH = makeGraph(points, midpoints, textEmbeddings)
    nodes, images, texts, src, tgt, weights = getConnections(GRAPH)

    return nodes, images, texts, src, tgt, weights

if __name__ == '__main__':
#   # data_folder = './Data'
#   # print("===== Getting Data Files =====")
#   # files = []
#   # for root, dirs, files_in_dir in os.walk(data_folder):
#   #   for file_name in files_in_dir:
#   #       files.append(os.path.join(root, file_name[:-4]))


#   # for i in range(0, len(files), 2):
#   #   if "DS_S" not in files[i]:
#   #       image = files[i] + ".png"
#   #       #print(image)
#   #       makeEmbedding(image, 'regular') 
    print(makeEmbedding("/Users/arunkrishnavajjala/Documents/GMU/PhD/P3/Data/com.iven.iconify_Top_Down_12.png", 'regular'))







