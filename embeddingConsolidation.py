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
import zlib
def concatenate_embeddings(embedding1, embedding2):
    concatenated_embedding = list(embedding1) + list(embedding2)  # Concatenate the two embeddings
    #normEmbed = normalize_list(concatenated_embedding)
    #concatenated_embedding = np.array(embedding1) * np.array(embedding2)
    return list(concatenated_embedding)

def get_area(triangle_embs, points):
    
    emb1 = np.array(triangle_embs[0])
    emb2 = np.array(triangle_embs[1])
    emb3 = np.array(triangle_embs[2])
    # Compute the pairwise distances
    a = np.linalg.norm(emb1 - emb2)
    b = np.linalg.norm(emb1 - emb3)
    c = np.linalg.norm(emb2 - emb3)
    # Compute s, the semi-perimeter
    s = (a + b + c) / 2
    # Now compute the area using Heron's formula
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    return area

#print(list(simplex_tree.get_filtration())[:50])

def get_midpoint(bounding_box):
    x_min, y_min, x_max, y_max = bounding_box
    mid_x = (x_min + x_max) / 2
    mid_y = (y_min + y_max) / 2
    return (mid_x, mid_y)

def compute_centroid(points):
    # Initialize RipsComplex with the points, using a max edge length of 1
    rips_complex = gd.RipsComplex(points=points, max_edge_length=0.5)
    # Compute the simplex tree (this computes the Rips complex)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    # Print the number of simplices in the complex
    print(f'Number of simplices: {simplex_tree.num_simplices()}')
    if simplex_tree.num_simplices() > 2000000:
        return "too many"
    #treeEmbedding = get_treeEmbedding(simplex_tree)
    #print(treeEmbedding)
    #print("Size of the array:", treeEmbedding.size)
    total_area = 0
    areas = []
    centroids = []
    count = 0
    # Print the simplices
    singles = {}
    for filtered_value in simplex_tree.get_filtration():
        if len(list(filtered_value[0])) == 1:
            singles[filtered_value[0][0]] =  points[filtered_value[0][0]]

        if len(list(filtered_value[0])) == 3: #change back to 3
            triangle_embeddings = [points[i] for i in list(filtered_value[0])]
            centroid_embedding = np.mean(triangle_embeddings, axis=0)
            for i in filtered_value[0]:
                if i in singles.keys():
                    singles.pop(i)
            centroids.append(centroid_embedding)
            area = get_area(triangle_embeddings, points)
            total_area += area
            areas.append(area)
            count += 1

    Singleembeddings = []
    for i in singles:
        Singleembeddings.append(singles[i])
   

    areas_np = np.array(areas)
    centroids_np = np.array(centroids)
    normalized_areas = areas_np / total_area
    result = normalized_areas[:, None] * centroids_np
    final_centroid = np.mean(result, axis=0)
    final_centroid = np.mean(centroids_np, axis=0)
    # if len(Singleembeddings) > 0:
    #     meanSingle = np.mean(Singleembeddings, axis = 0)
    #     final_centroid = np.mean([meanSingle, final_centroid], axis = 0)
    #     print("SINGLEE")
    print("Num Triangles: " + str(count))
    if count == 0:
        return []
    final_centroid = compress_embedding(final_centroid)
    return final_centroid    

def makePoints(aug_images, aug_texts):
    points = []
    for i in range(0, len(aug_images)):
       weighted_aug_texts = [0.5 * value for value in aug_texts[i]]
       point = list(aug_images[i]) + list(weighted_aug_texts)
       points.append(point)
    return points

def compress_embedding(embedding):
    """
    Compresses the given embedding array using Lempel-Ziv-Welch (LZW) compression.

    Parameters:
        embedding (numpy.ndarray): The embedding array to compress.

    Returns:
        bytes: The compressed representation of the embedding array.
    """
    # Convert embedding array to bytes
    embedding_bytes = embedding.tobytes()

    # Compress using zlib (LZW compression)
    compressed_embedding = zlib.compress(embedding_bytes)

    return compressed_embedding
