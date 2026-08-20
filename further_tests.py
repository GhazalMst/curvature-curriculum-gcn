import numpy as np
from scipy import stats
import networkx as nx
from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from data_preprocess import get_curvature

def get_curvature_distribution(file):
    positive = np.loadtxt(file, dtype=np.int64) #array of paris of positive interactions/adges
    if positive[0].shape==1:
        unique_entity = positive[0]
    G_all = nx.Graph()
    G_all.add_edges_from(positive)  #generates nodes automatically
    
    #get number of nodes
    with open(file, 'r') as f:
        first_line = f.readline()
        if first_line.startswith("#"):
            parts = first_line.split(":")
            unique_entity = int(parts[-1].strip())
        else: unique_entity = len(G_all.nodes)
    
    edges, curvature = get_curvature(unique_entity, positive)
    return curvature

def main():
    data1 = get_curvature_distribution('data/HuRI-PPI.edgelist')
    #data2 = get_curvature_distribution('data/PP-Decagon.edgelist')
    data3 = get_curvature_distribution('data/simulated13.edgelist')
    d_stat, p_value = stats.kstest(data1, data3)
    print(f"KS statistics: {d_stat}")
    print(f"p value: {p_value}")
    
if __name__ == "__main__":
    main()