import torch
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from data_preprocess import get_curvature
from scipy import stats
from scipy.stats import wasserstein_distance
import numpy as np
import os
import random
    
def load_curvature(file):
    print('Loading dataset...')
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
        
    edge_index_o_total, edge_cuva_total = get_curvature(unique_entity, positive)
    return edge_cuva_total
    
def measure_distance(dist1, dist2):
    distance = wasserstein_distance(dist1, dist2)
    return distance

def generate_community_edgelist(filename, num_nodes=1000, num_communities=5, p=0.1, q=0.001): 
    if not os.path.exists('simulated_data'):
        os.makedirs(foldername)
    file_path = os.path.join('simulated_data', filename)
    
    nodes_per_comm = num_nodes // num_communities
    edges = []

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Determine if they are in the same community
            same_comm = (i // nodes_per_comm) == (j // nodes_per_comm)
            
            # Probability check
            prob = p if same_comm else q
            
            if random.random() < prob:
                edges.append(f"{i} {j}")
    
    # Write to file
    with open(file_path, "w") as f:
        f.write(f"# Nodes: {num_nodes}\n")
        f.write("\n".join(edges))

    return file_path

def main():
    curv_HuRI2 = load_curvature('data/HuRI2.edgelist')
    #for i in range(10):
    #    indices = torch.randint(0, len(curv_HuRI2), (len(curv_HuRI2),)) #replace implicite
    #    curv_HuRI2_bootstrap = curv_HuRI2[indices]
    #    print(i, 'wasserstein distance curvature HuRI2 bootstrap', measure_distance(curv_HuRI2, curv_HuRI2_bootstrap))

    curv_Ohmnet = load_curvature('data/PPT-Ohmnet.edgelist')
    curv_Pathways = load_curvature('data/PP-Pathways.edgelist')
    curv_Decagon = load_curvature('data/PP-Decagon.edgelist')
    
    axes = plt.subplots(2, 2, figsize=(10, 8))[1]
    sns.histplot(curv_HuRI2, ax=axes[0, 0])
    sns.histplot(curv_Ohmnet, ax=axes[0, 1])
    sns.histplot(curv_Pathways, ax=axes[1, 0])
    sns.histplot(curv_Decagon, ax=axes[1, 1])
    axes[0, 0].set_title('Curvature HuRI2')
    axes[0, 1].set_title('Curvature PPT-Ohmnet')
    axes[1, 0].set_title('Curvature PP-Pathways')
    axes[1, 1].set_title('Curvature PP-Decagon')
    plt.tight_layout()
    plt.savefig('new_curvature_plots.png', dpi=300)
    
    print('wasserstein distance curvature HuRI2 PPT-Ohmnet', measure_distance(curv_HuRI2, curv_Ohmnet))
    d_stat, p_value = stats.kstest(curv_HuRI2, curv_Ohmnet)
    print(f"KS statistics: {d_stat}")
    print(f"p value: {p_value}")
    print('wasserstein distance curvature HuRI2 PP-Pathways', measure_distance(curv_HuRI2, curv_Pathways))
    d_stat, p_value = stats.kstest(curv_HuRI2, curv_Pathways)
    print(f"KS statistics: {d_stat}")
    print(f"p value: {p_value}")
    print('wasserstein distance curvature HuRI2 PP-Decagon', measure_distance(curv_HuRI2, curv_Decagon))
    d_stat, p_value = stats.kstest(curv_HuRI2, curv_Decagon)
    print(f"KS statistics: {d_stat}")
    print(f"p value: {p_value}")
    print('wasserstein distance curvature PPT-Ohmnet PP-Pathways', measure_distance(curv_Ohmnet, curv_Pathways))
    d_stat, p_value = stats.kstest(curv_Ohmnet, curv_Pathways)
    print(f"KS statistics: {d_stat}")
    print(f"p value: {p_value}")
    print('wasserstein distance curvature PPT-Ohmnet PP-Decagon', measure_distance(curv_Ohmnet, curv_Decagon))
    d_stat, p_value = stats.kstest(curv_Ohmnet, curv_Decagon)
    print(f"KS statistics: {d_stat}")
    print(f"p value: {p_value}")
    print('wasserstein distance curvature PP-Pathways PP-Decagon', measure_distance(curv_Pathways, curv_Decagon))
    d_stat, p_value = stats.kstest(curv_Pathways, curv_Decagon)
    print(f"KS statistics: {d_stat}")
    print(f"p value: {p_value}")
    
    best_file_HuRI2 = 'data/simulated8.edgelist'
    best_file_Ohmnet = 'data/simulated8.edgelist'
    best_file_Pathways = 'data/simulated8.edgelist'
    best_file_Decagon = 'data/simulated8.edgelist'
    curv_simulated8 = load_curvature('data/simulated8.edgelist')
    
    best_dist_HuRI2 = measure_distance(curv_HuRI2, curv_simulated8)
    best_dist_Ohmnet = measure_distance(curv_Ohmnet, curv_simulated8)
    best_dist_Pathways = measure_distance(curv_Pathways, curv_simulated8)
    best_dist_Decagon = measure_distance(curv_Decagon, curv_simulated8)
    
    for num_nodes in [1000, 2000]:
        for num_communities in [5, 10, 20, 50]:
            for p in [0.005, 0.01, 0.1, 0.2]:
                for q in [0.0001, 0.01, 0.02]:
                    print(f"num_nodes={num_nodes}, num_communities={num_communities}, p={p}, q={q}")
                    #filename = f"simulated_{num_nodes}_{num_communities}_p{p}_q{q}.edgelist"
                    #file_path = generate_community_edgelist(filename, num_nodes, num_communities, p, q)
                    file_path = f"simulated_data/simulated_{num_nodes}_{num_communities}_p{p}_q{q}.edgelist"
                    curv_simulated = load_curvature(file_path)
                    
                    #HuRI2
                    current_dist_HuRI2 = measure_distance(curv_HuRI2, curv_simulated)
                    if current_dist_HuRI2 < best_dist_HuRI2:
                        best_dist_HuRI2 = current_dist_HuRI2
                        best_file_HuRI2 = file_path
                    #PPT-Ohmnet
                    current_dist_Ohmnet = measure_distance(curv_Ohmnet, curv_simulated)
                    if current_dist_Ohmnet < best_dist_Ohmnet:
                        best_dist_Ohmnet = current_dist_Ohmnet
                        best_file_Ohmnet = file_path
                    #PP-Pathways
                    current_dist_Pathways = measure_distance(curv_Pathways, curv_simulated)
                    if current_dist_Pathways < best_dist_Pathways:
                        best_dist_Pathways = current_dist_Pathways
                        best_file_Pathways = file_path
                    #PP-Decagon
                    current_dist_Decagon = measure_distance(curv_Decagon, curv_simulated)
                    if current_dist_Decagon < best_dist_Decagon:
                        best_dist_Decagon = current_dist_Decagon
                        best_file_Decagon = file_path
                    
    #HuRI2
    print(best_file_HuRI2, best_dist_HuRI2)
    curv_simulated = load_curvature(best_file_HuRI2)
    d_stat, p_value = stats.kstest(curv_HuRI2, curv_simulated)
    print(f"KS statistics: {d_stat}")
    print(f"p value: {p_value}")
    
    #PPT-Ohmnet
    print(best_file_Ohmnet, best_dist_Ohmnet)
    curv_simulated = load_curvature(best_file_Ohmnet)
    d_stat, p_value = stats.kstest(curv_Ohmnet, curv_simulated)
    print(f"KS statistics: {d_stat}")
    print(f"p value: {p_value}")
                    
    #PP-Pathways
    print(best_file_Pathways, best_dist_Pathways)
    curv_simulated = load_curvature(best_file_Pathways)
    d_stat, p_value = stats.kstest(curv_Pathways, curv_simulated)
    print(f"KS statistics: {d_stat}")
    print(f"p value: {p_value}")
    
    #PP-Decagon
    print(best_file_Decagon, best_dist_Decagon)
    curv_simulated = load_curvature(best_file_Decagon)
    d_stat, p_value = stats.kstest(curv_Decagon, curv_simulated)
    print(f"KS statistics: {d_stat}")
    print(f"p value: {p_value}")
    
    


if __name__ == "__main__":
    main()