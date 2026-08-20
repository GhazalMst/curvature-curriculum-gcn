import random
import os
from data_preprocess import get_curvature
import numpy as np
import networkx as nx

def generate_community_edgelist(filename, foldername='data', num_nodes=1000, num_communities=5, p=0.1, q=0.001):
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    file_path = os.path.join(foldername, filename)
    
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
    
    print(f"Edgelist created with {num_nodes} nodes, {num_communities} communites and {len(edges)} edges.")
    return file_path

def compute_graph_properties(file):
    edges = np.loadtxt(file, dtype=np.int64)
    G = nx.Graph()
    G.add_edges_from(edges)
    
    #get number of nodes
    with open(file, 'r') as f:
        first_line = f.readline()
        if first_line.startswith("#"):
            parts = first_line.split(":")
            unique_entity = int(parts[-1].strip())
        else: unique_entity = len(G_all.nodes)
    
    print('avg_degree', sum(dict(G.degree()).values()) / G.number_of_nodes())
    print('density', nx.density(G))
    edge_index_o_total, edge_cuva_total = get_curvature(unique_entity, edges)
    pos_cuva_num = (edge_cuva_total >= 0).sum().item()
    neg_cuva_num = (edge_cuva_total <= 0).sum().item()
    if neg_cuva_num > 0:
        print('ratio positive to negative curvature', pos_cuva_num/neg_cuva_num)
    
def main():
    #simulated1_path = generate_community_edgelist('simulated1.edgelist', num_nodes=2000, num_communities=5, p=0.1, q=0.001)
    #compute_graph_properties(2000, simulated1_path)
    #simulated2_path = generate_community_edgelist('simulated2.edgelist', num_nodes=2000, num_communities=5, p=0.1, q=0.0001)
    #compute_graph_properties(2000, simulated2_path)
    #simulated3_path = generate_community_edgelist('simulated3.edgelist', num_nodes=2000, num_communities=10, p=0.1, q=0.0001)
    #compute_graph_properties(2000, simulated3_path)
    #simulated4_path = generate_community_edgelist('simulated4.edgelist', num_nodes=1000, num_communities=5, p=0.1, q=0.001)
    #compute_graph_properties(1000, simulated4_path)
    #simulated5_path = generate_community_edgelist('simulated5.edgelist', num_nodes=1000, num_communities=5, p=0.1, q=0.0001)
    #compute_graph_properties(1000, simulated5_path)
    #simulated6_path = generate_community_edgelist('simulated6.edgelist', num_nodes=2000, num_communities=10, p=0.01, q=0.0001)
    #compute_graph_properties(simulated6_path)
    #simulated7_path = generate_community_edgelist('simulated7.edgelist', num_nodes=3000, num_communities=20, p=0.01, q=0.0001)
    #compute_graph_properties(simulated7_path)  
    #simulated8_path = generate_community_edgelist('simulated8.edgelist', num_nodes=5000, num_communities=50, p=0.1, q=0.0001)
    #compute_graph_properties(simulated8_path)
    #simulated9_path = generate_community_edgelist('simulated9.edgelist', num_nodes=5000, num_communities=100, p=0.1, q=0.001)
    #compute_graph_properties(simulated9_path)
    #simulated10_path = generate_community_edgelist('simulated10.edgelist', num_nodes=5000, num_communities=100, p=0.1, q=0.0001)
    #compute_graph_properties(simulated10_path)
    #simulated11_path = generate_community_edgelist('simulated11.edgelist', num_nodes=5000, num_communities=50, p=0.05, q=0.0001)
    #compute_graph_properties(simulated11_path)
    #simulated12_path = generate_community_edgelist('simulated12.edgelist', num_nodes=5000, num_communities=50, p=0.2, q=0.0001) #ganz weit weg
    #compute_graph_properties(simulated12_path)
    #simulated13_path = generate_community_edgelist('simulated13.edgelist', num_nodes=5000, num_communities=50, p=0.1, q=0.0002)
    #compute_graph_properties(simulated13_path)
    simulated14_path = generate_community_edgelist('simulated14.edgelist', num_nodes=5000, num_communities=50, p=0.1, q=0.00005)
    compute_graph_properties(simulated14_path)
    
if __name__ == "__main__":
    main()