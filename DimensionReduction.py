import numpy as np
from sklearn.decomposition import PCA
import csv
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

def makeList(inlst):
    strLst = str(inlst)
    strLst = strLst.strip('tensor(')
    strLst = strLst.strip(')')
    strLst = strLst.strip('(')
    strLst = strLst.split(' ')
    embed = []
    for i in strLst:
        if len(i)>2:
            number = i.strip('\n')
            number = number.strip(', ')
            number = number.strip('[')
            number = number.strip(']')
            if len(number) > 2:
                add = float(number)
                embed.append(add)
    return embed
def normalPCA(csvFile):
    # Example embeddings (original data)
    allEmbeddings = []
    with open(csvFile, 'r') as file:
            csv_reader = csv.reader(file)
            count = 0
            for row in csv_reader:
                if count > 0:
                    embed = makeList(row[3])
                    allEmbeddings.append(embed)
                count += 1
    # Convert the original embeddings into a NumPy array
    print(len(allEmbeddings))
    X_original = np.array(allEmbeddings)

    # Perform PCA on the original embeddings
    pca = PCA(n_components=140)  # Choose the number of components you want to reduce to
    pca.fit(X_original)

    # Get the transformation matrix (components_)
    transformation_matrix = pca.components_.tolist()
    print(transformation_matrix)

def lle(csvFile):
    from sklearn.manifold import LocallyLinearEmbedding
    # Example embeddings (original data)
    allEmbeddings = []
    with open(csvFile, 'r') as file:
            csv_reader = csv.reader(file)
            count = 0
            for row in csv_reader:
                if count > 0:
                    embed = makeList(row[3])
                    allEmbeddings.append(embed)
                count += 1
    # Convert the original embeddings into a NumPy array

    X_original = np.array(allEmbeddings)
    lle = LocallyLinearEmbedding(n_components=256, n_neighbors=15, method='standard')
    embedding_lle = lle.fit_transform(X_original)

    # Return transformation matrix
    print(len(lle.embedding_))
    print(len(lle.embedding_[0]))



csvFile = "/Users/arunkrishnavajjala/Documents/GMU/PhD/P3/UIEmbedding/rips_ricodata.csv"
lle(csvFile)