# UI Embedding

A brief description of your project goes here.

## Technical Specifications
Computer vision

- UIED library to extract icons on the screen
- Pytesseract to extract text on the screen
  
Embeddings (could change, see link)

- CLIP for images
- BERT for text
- N/A for graph Nodes
  
Graph

- Python NetworkX graph to build the graph
- Edge Lengths 200 (average distance)
- Edge weights: 100/distance
  
Nodes

- Represent each icon on the screen
- NetworkX Graph Nodes
- Contain CLIP embedding
- Contain BERT Embedding
- Contain positional coordinate


## Environment Setup
- Required environment versions:
- ```Python Version: 3.9.13```
- ```Pip Version: 23.3.2```
- The repository will have a requirements.txt file that lists all required packages for UI Embedding to run. In your command line, create a new python environment: ``` python3 -m venv venv``` Once your environment is created, activate it with this command: ```source venv/bin/activate```. Use this command to download all of the dependencies into your virtual environment:  ```pip install -r requirements.txt```.
- Once the dependencies are downloaded, go to line 103 in UI_Embedding_main.py and replace the existing path with a path to a screenshot image on your machine
- Run ```python3 UI_Embedding_main.py``` to run the code

## 

List the requirements and environment setup instructions here. Include dependencies and any prerequisites needed to run the project.

```bash
# Example:
pip install -r requirements.txt
