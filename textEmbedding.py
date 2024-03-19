import pytesseract
import importlib
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import gudhi as gd
# Use birth_death_pairs to create an embedding, for example, using t-SNE
from sklearn.preprocessing import normalize
from sklearn.manifold import spectral_embedding
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



def makeTextEmbedding(image):
    extractedText = pytesseract.image_to_string(image) # extractedText = Settings    Close 
    textEmbedding = ""
    if len(extractedText) < 5:
        extractedText = 'NO-TEXT'
        textEmbedding = textEmbed(extractedText[0:511])

    if len(extractedText) > 512:
        textEmbedding = textEmbed(extractedText[0:511])
    else:
        textEmbedding = textEmbed(extractedText)
    return textEmbedding


def textEmbed(text = 'NO-TEXT'):
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    if len(text) < 5:
        text = "NO-TEXT"
    tokens = tokenizer.encode(text, add_special_tokens=True)

    input_ids = torch.tensor(tokens).unsqueeze(0)  # Add batch dimension
    # Forward pass through BERT model
    with torch.no_grad():
        outputs = model(input_ids)

    # Extract embeddings from the last layer
    embeddings = outputs.last_hidden_state
    pooled_embedding = torch.mean(embeddings, dim=1)
    #print('len of text: ', len(pooled_embedding[0]))
    return(pooled_embedding[0])



























