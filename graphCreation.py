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


def makeGraph(points, midpoints, textEmbeddings):
    # Your list of 2D points
    # Create a graph
    G = nx.Graph()

    # Add nodes with attributes (assigning 2D points)
    count = 0
    for i in range(len(points)):
        count += 1
        G.add_node(tuple(midpoints[i]), pos=tuple(midpoints[i]), imageembedding=points[i], textembedding=textEmbeddings[i])  # 'pos' attribute stores the 2D point
        #G.add_node("Point:" + str(count), pos=tuple(midpoints[i]))  # 'pos' attribute stores the 2D point

    # Define a connection criterion (for example, connecting all points within a certain distance)
    threshold_distance = 200  # Change this threshold as needed

    edgeLengths = []

    # Connect points based on distance (example criteria)
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 != node2:
                distance = np.linalg.norm(np.array(G.nodes[node1]['pos']) - np.array(G.nodes[node2]['pos']))
                edgeLengths.append(distance)
                if distance <= threshold_distance and distance > 0:
                    G.add_edge(node1, node2, weight = float(100/distance))
    return(G)



def getConnections(G):
    nodes = []
    ImageEmbeddings = []
    TextEmbeddings = []
    for node in G.nodes():
        nodes.append(node)
        image_embedding = G.nodes[node].get('imageembedding', None)
        ImageEmbeddings.append(image_embedding)
        text_embedding = G.nodes[node].get('textembedding', None)
        TextEmbeddings.append(text_embedding)

    src = []
    tgt = []
    weights = []
    seen_edges = set()
    for node in G.nodes():
        node_index = nodes.index(node)
        for neighbor in G.neighbors(node):
            edge = frozenset([node, neighbor])
            if edge not in seen_edges:
                seen_edges.add(edge)
                edge_weight = G[node][neighbor]['weight']
                src.append(node_index)
                tgt.append(nodes.index(neighbor))
                weights.append(edge_weight)

    return nodes, ImageEmbeddings, textEmbeddings, src, tgt, weights

if __name__ == "__main__":
    # Sample Graph
    points = [['p1'], ['p2'],['p3'],['p4'],['p5'],['p6'],['p7']]
    midpoints = [(1,0), (44,0), (500,1000), (30,76), (90,400), (200, 3), (90,100)]
    textEmbeddings = [['t1'], ['t2'],['t3'],['t4'],['t5'],['t6'],['t7']]

    G = makeGraph(points, midpoints, textEmbeddings)
    nodes, images, texts, src, tgt, weights = getConnections(G)

    print("STATS")
    print("Length Nodes: " + str(len(nodes)))
    print("Length SRC: " + str(len(src)))
    print("Length TGT: " + str(len(tgt)))
    print("Length Weights: " + str(len(weights)))
    print()
    print("Nodes are: " + str(nodes))
    print("SRC is: " + str(src))
    print("TGT is: " + str(tgt))
    print("Weights are: " + str(weights))































