import time
from PIL import Image
import os
import importlib
import clip
import numpy as np
import torch
from PIL import Image
# Use birth_death_pairs to create an embedding, for example, using t-SNE
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
from embeddingConsolidation import makePoints, compute_centroid, concatenate_embeddings, average_lists

warnings.filterwarnings("ignore")
# os.chdir("/Users/arunkrishnavajjala/Documents/GMU/PhD/P3/UIEmbedding")
uiedDir = "/Users/arunkrishnavajjala/Documents/GMU/PhD/P3/UIEmbedding/detectors/Visual/UIED-master/data/output/ip/"


def get_degree (adj):
    degree = np.sum (adj, axis = 1)
    for i in range(len(degree)):
        if degree[i]:
            degree[i] = 1 / np.sqrt(degree[i])  
    return degree

def get_noralimized_adjacency(src_nodes, trgt_nodes, corpus_size, weight):
    adj = np.zeros ([corpus_size, corpus_size])
    adj_sec = np.zeros ([corpus_size, corpus_size])
    for i in range(corpus_size):
        direct_nebr, second_nebr = set(), set()
        for j in range(len(src_nodes)):
            if i == src_nodes[j]:
                adj[i][trgt_nodes[j]] = 1
                direct_nebr.add(trgt_nodes[j])
            if i == trgt_nodes[j]:
                adj[i][src_nodes[j]] = 1  
                direct_nebr.add(src_nodes[j])                  
        for j in range(len(src_nodes)):
            for k in direct_nebr:
                if k == src_nodes[j] and not adj[i][trgt_nodes[j]]:
                    adj_sec[i][trgt_nodes[j]] = 1
                    second_nebr.add(trgt_nodes[j])
                if k == trgt_nodes[j] and not adj[i][src_nodes[j]]:
                    adj_sec[i][src_nodes[j]] = 1
                    second_nebr.add(src_nodes[j])
        adj[i][i] = 0
        adj_sec[i][i] = 0

    degree, degree_sec = get_degree(adj), get_degree(adj_sec)
    adj = np.matmul(np.matmul(np.diag(degree), adj * weight), np.diag(degree)) + np.identity(corpus_size)
    adj_sec = adj + np.matmul(np.matmul(np.diag(degree_sec), adj_sec * weight), np.diag(degree_sec))  

    return adj, adj_sec

def augment_embeddings(images, texts, src, tgt, weights, option):
    node_size = len(images)
    adj, adj_sec = get_noralimized_adjacency(src, tgt, node_size, weights)
    if option == 1:
        return np.matmul(adj, images), np.matmul(adj, texts)
    else:
        return np.matmul(adj_sec, images), np.matmul(adj_sec, texts)

def makeEmbedding(image, embeddingType):
    big_image = image
    image = image
    textImage = image
    big_image_embedding = bigImageEmbedding(big_image)
    uiedResult = runUIED(image)
    allBoxes = uiedResult[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
   
    imagePath = uiedResult[2]
    jsonPath = uiedResult[1]
    print(jsonPath)
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
    # print(texts[1])
    # print(len(images), len(texts), len(src), len(tgt))
    # print(type(images[0]), type(texts[0]), type(src), type(tgt))
    aug_images, aug_texts = augment_embeddings(images, texts, src, tgt, weights = 0.9, option = 1)
    #print('BIG TEXT: ', len(big_textEmbedding))
    points = makePoints(aug_images, aug_texts)
    finalCentroid = average_lists(points)
    withBigClip = concatenate_embeddings(big_image_embedding, finalCentroid)
    #withBigText = concatenate_embeddings(withBigClip, big_textEmbedding)
    #retList = [float(tensor) for tensor in withBigText]
    return(withBigClip)


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
    start_time = time.time()
    print(len(makeEmbedding("/Users/arunkrishnavajjala/Documents/GMU/PhD/LabeledDataset/LabeledRICO/Search screen/com.auntieannes.pretzelperks_trace_1_213.jpg", 'regular')))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time Elapsed: {elapsed_time} seconds")


    ### embedding propagation using constant weights
    # weights can be chosen from [0, 1] with increment of 0.1 (0.5 by default used by Athena)
    # option = 1 means only consider 1-hop neighbors; = 2 means consider neighbors within two hops (2 by default used by Athena)
    


    




