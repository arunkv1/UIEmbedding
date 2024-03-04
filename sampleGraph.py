import networkx as nx

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
    threshold_distance = 50  # Change this threshold as needed

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

points = [['p1'], ['p2'],['p3'],['p4'],['p5'],['p6'],['p7']]
midpoints = [(1,0), (44,0), (500,1000), (30,76), (90,400), (200, 3), (90,100)]
textEmbeddings = [['t1'], ['t2'],['t3'],['t4'],['t5'],['t6'],['t7']]


def getConnections(G):
    return G

G = makeGraph(points, midpoints, textEmbeddings)

getConnections(G)
















