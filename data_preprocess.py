import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import to_undirected
from utils import *
from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci
#from indices import compute_all_edge_metrics
import pandas as pd
import ot


class Data_class(Dataset):
 
    def __init__(self, triple, curvature):
        self.entity1 = triple[:, 0]
        self.entity2 = triple[:, 1]
        self.label = triple[:, 2]
        self.curvature = curvature

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):

        return self.label[index], (self.entity1[index], self.entity2[index]), self.curvature[index]
    
def get_curvature(unique_entity, positive):
    #build graph
    G = nx.Graph()
    G.add_edges_from(positive)
    
    #adds self-loops
    for node in G.nodes:
        G.add_edge(node,node)
        
    #compute curvature
    orc = OllivierRicci(G, alpha=0.5, verbose="INFO")
    orc.compute_ricci_curvature()
    adj = sp.coo_matrix((np.ones(positive.shape[0]), (positive[:, 0], positive[:, 1])),
                        shape=(unique_entity, unique_entity), dtype=np.float32)
    adj_o = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    edges_o = adj_o.nonzero()
    edge_index_o = torch.tensor(np.vstack((edges_o[0], edges_o[1])), dtype=torch.long)
    
    edge_cuva = []
    edge_to_curv = {}  # Add this
    
    for a in range(edge_index_o.shape[1]):
        #if self-loop assign value 0
        if edge_index_o[0,a] == edge_index_o[1,a]:
            edge_cuva.append(0)
        else:
            row = edge_index_o[0,a].numpy()
            col = edge_index_o[1,a].numpy()
            c = orc.G[int(row)][int(col)]["ricciCurvature"]
            edge_cuva.append(c)
            edge_to_curv[(int(row), int(col))] = c  # Add this
            
    edge_cuva = torch.tensor(edge_cuva, dtype=torch.float)
    
    return edge_index_o, edge_cuva, edge_to_curv  # Return 3 values

#get theoretical curvature values of negative edges
def get_negatives(unique_entity, negative, positive, args, batch_size=1000):
    #build graph
    G = nx.Graph()
    G.add_edges_from(positive)
    
    #adds self-loops
    for node in list(G.nodes):
        G.add_edge(node, node)
        
    edge_cuva_neg = []
    
    #add batches of negative edges
    for i in range(0, len(negative), batch_size):
        batch = negative[i : i + batch_size]
        new_edges = [(int(u), int(v)) for u, v in batch]
        G.add_edges_from(new_edges)
        orc = OllivierRicci(G, alpha=0.5, verbose="ERROR")
        orc.compute_ricci_curvature()
        
        for u, v in new_edges:
            if u == v:
                edge_cuva_neg.append(0.0)
            else:
                curvature = orc.G[u][v].get("ricciCurvature", 0.0)
                edge_cuva_neg.append(curvature)
        G.remove_edges_from(new_edges)

    edge_index_o_neg = torch.tensor(negative.T, dtype=torch.long)
    edge_cuva_neg = torch.tensor(edge_cuva_neg, dtype=torch.float)

    return edge_index_o_neg, edge_cuva_neg


def _node_mass_distribution(G, node, alpha=0.5):
    #1-hop neighbor mass distribution used by Ollivier-Ricci curvature: alpha
    #stays on the node itself, the remaining (1-alpha) is split uniformly
    #across the node's real neighbors -- no self-loop added here (unlike
    #get_curvature()'s positive-edge graph), so a node's degree is its true
    #training-graph degree
    neighbors = list(G.neighbors(node))
    if not neighbors:
        return {node: 1.0}
    masses = {}
    per_nbr = (1.0 - alpha) / len(neighbors)
    for nbr in neighbors:
        masses[nbr] = masses.get(nbr, 0.0) + per_nbr
    masses[node] = masses.get(node, 0.0) + alpha
    return masses


#proxy curvature for non-edge pairs: kappa~(u,v) = 1 - W1(m_u, m_v)/d(u,v),
#computed from each node's EXISTING neighborhood. Unlike get_negatives()
#above, the graph is never mutated to include (u, v), so neither node's
#neighbor distribution is contaminated by the pair being scored -- this also
#means we skip the cost of rerunning OllivierRicci.compute_ricci_curvature()
#over the whole graph per batch.
#
#The real Ollivier-Ricci formula this codebase uses for edges is actually
#1 - W1(m_u,m_v)/d(u,v), which collapses to 1 - W1(m_u,m_v) for an edge
#because d(u,v)=1 in this codebase's unweighted graphs -- that's the form
#the write-up gives. For a non-edge, d(u,v) is usually >1 (or infinite
#across components), and W1 grows with it, so dropping the division blows
#kappa~ up to values far outside the range real edge curvature ever takes.
#Dividing by the actual shortest-path distance keeps kappa~ on the same
#scale as real kappa, which is what makes the two comparable under one
#curriculum threshold.
#High proxy curvature (shared neighbors relative to distance) marks a
#"hard" negative -- the mirror image of low curvature marking a hard
#positive edge.
#
#Pairs with NO path between them at all (different connected components,
#within this positive-edge graph) are handled separately, not run through
#the formula above: their cost matrix would be one constant "very far"
#number in every cell, which makes W1/d(u,v) cancel out to exactly 1
#regardless of what either node's neighborhood looks like -- a
#coincidental, meaningless neutral 0, not a real signal.
#
#These pairs are marked with curvature = inf rather than a fixed low
#constant. "No path in THIS graph" is not reliable evidence of "genuinely
#unrelated": when `positive` is a single fold's thinned-out training
#edges (not the full graph), removing edges for the split/fold routinely
#disconnects nodes that are actually close together in the true graph --
#checking against the full graph to tell the two cases apart would mean
#leaking held-out edges into a training-time decision, so instead of
#guessing, inf marks these pairs as "no real curvature computed" rather
#than asserting a specific (and often wrong) hardness. The training loop
#already treats non-finite curvature as no-signal (see the
#torch.isfinite checks in train.py), so these pairs keep a neutral loss
#weight and still contribute their plain classification label, instead of
#being scored as a fake "confidently easy" negative.
#
#DISCONNECTED_CURVATURE (-1.0) is kept for other callers/diagnostics that
#still key off the old sentinel value; get_proxy_curvature itself no
#longer produces it.
DISCONNECTED_CURVATURE = -1.0

def get_proxy_curvature(unique_entity, positive, pairs, alpha=0.5):
    G = nx.Graph()
    # negatives are sampled from the full node universe, so a pair can touch
    # a node with zero training-split positive edges -- add every node up
    # front (not just edge endpoints) so it still gets a valid mass
    # distribution (see _node_mass_distribution's no-neighbors case) instead
    # of a KeyError. No self-loops here, unlike get_curvature()'s positive
    # edges -- a node's own neighbor set for the proxy computation is exactly
    # its real training-graph neighbors, nothing added.
    G.add_nodes_from(range(unique_entity))
    G.add_edges_from(positive)

    apsp = dict(nx.all_pairs_shortest_path_length(G))

    mass_cache = {}
    def masses(n):
        if n not in mass_cache:
            mass_cache[n] = _node_mass_distribution(G, n, alpha=alpha)
        return mass_cache[n]

    proxy_curv = []
    for u, v in pairs:
        u, v = int(u), int(v)
        if u == v:
            proxy_curv.append(0.0)
            continue
        if v not in apsp[u]:
            proxy_curv.append(float('inf'))
            continue
        mu, mv = masses(u), masses(v)
        u_support, u_mass = list(mu.keys()), np.array(list(mu.values()))
        v_support, v_mass = list(mv.keys()), np.array(list(mv.values()))
        # u and v share a connected component, so every neighbor of either
        # is also in it -- apsp[su][sv] is always defined here
        cost = np.array([
            [apsp[su][sv] for sv in v_support]
            for su in u_support
        ], dtype=np.float64)
        w1 = ot.emd2(u_mass, v_mass, cost)
        proxy_curv.append(1.0 - w1 / apsp[u][v])

    return torch.tensor(proxy_curv, dtype=torch.float)

#stratification via degrees (currently not used)
def sample_balanced_negatives(pos_degrees, cand_degrees, candidates, num_bins=50):
    bins = np.linspace(min(pos_degrees.min(), cand_degrees.min()), 
                       max(pos_degrees.max(), cand_degrees.max()), 
                       num_bins + 1)

    pos_bin_idx = np.digitize(pos_degrees, bins)
    cand_bin_idx = np.digitize(cand_degrees, bins)
    
    selected_indices = []
    
    for b in range(1, len(bins)):
        num_needed = np.sum(pos_bin_idx == b)
        available_indices = np.where(cand_bin_idx == b)[0]     
        if len(available_indices) >= num_needed:
            choice = np.random.choice(available_indices, num_needed, replace=False)
            selected_indices.extend(choice)
        elif len(available_indices) > 0:
            selected_indices.extend(available_indices)
            
    if len(selected_indices) < len(pos_degrees):
        remaining_needed = len(pos_degrees) - len(selected_indices)
        all_indices = set(range(len(candidates)))
        used_indices = set(selected_indices)
        leftover_indices = list(all_indices - used_indices)
        
        extra_choice = np.random.choice(leftover_indices, remaining_needed, replace=False)
        selected_indices.extend(extra_choice)
        
    return candidates[selected_indices], cand_degrees[selected_indices]

#`fast` is unused now: negative test-edge curvature used to be skipped
#(fast=True, an inf placeholder) or computed with the expensive
#insert-based get_negatives() (fast=False) -- it's now always computed via
#the cheap get_proxy_curvature, matching how train/val compute theirs, so
#there's no longer a slow path for this parameter to opt out of. Kept in
#the signature for backward compatibility with any existing callers.
def load_data(args, val_ratio=0.1, test_ratio=0.2, fast=True):
    """Read data from path, convert data into loader, return features and symmetric adjacency"""
    # read data
    print('Loading {0} seed{1} dataset...'.format(args.in_file, args.seed))
    positive = np.loadtxt(args.in_file, dtype=np.int64)
    if positive[0].shape==1:
        unique_entity = positive[0]
    G_all = nx.Graph()
    G_all.add_edges_from(positive)

    #get number of nodes in case there are isolated nodes and it's marked at the beginning of the document
    with open(args.in_file, 'r') as f:
        first_line = f.readline()
        if first_line.startswith("#"):
            parts = first_line.split(":")
            unique_entity = int(parts[-1].strip())
        else: unique_entity = len(G_all.nodes)


    np.random.seed(args.seed)
    np.random.shuffle(positive)
    link_size = int(positive.shape[0] * args.network_ratio)
    positive = positive[:link_size]

    # split data
    val_size = int(val_ratio * positive.shape[0])
    test_size = int(test_ratio * positive.shape[0])

    #curvature for test positive edges: base graph is train+val edges only
    #(positive[:-test_size]), test edges inserted/removed batch-wise via
    #get_negatives() -- mirrors how negative curvature is computed, so no
    #test edge's curvature depends on any other test edge being present
    edge_index_o_test_pos, edge_cuva_test_pos = get_negatives(
        unique_entity, positive[-test_size:], positive[:-test_size], args)

    # sample negative
    # adds negative edges for non interaction such that #neg=#pos
    negative_all = list(nx.non_edges(G_all))
    np.random.seed(args.seed)
    np.random.shuffle(negative_all)
    negative = np.asarray(negative_all[:positive.shape[0]])
    print("positve examples: %d, negative examples: %d." % (positive.shape[0], negative.shape[0]))

    #add ground-truth value
    positive = np.concatenate([positive, np.ones(positive.shape[0], dtype=np.int64).reshape(positive.shape[0], 1)], axis=1)
    negative = np.concatenate([negative, np.zeros(negative.shape[0], dtype=np.int64).reshape(negative.shape[0], 1)], axis=1)

    train_data = np.vstack((positive[: -(val_size + test_size)], negative[: -(val_size + test_size)]))
    val_data = np.vstack((positive[-(val_size + test_size): -test_size], negative[-(val_size + test_size): -test_size]))
    test_data = np.vstack((positive[-test_size:], negative[-test_size:]))        

    #get curvature of only train/val data to avoid data leakage
    train_positive = positive[: -(val_size + test_size)]
    val_positive = positive[: -test_size]
    edge_index_o, edge_cuva, _ = get_curvature(unique_entity, train_positive[:,:2])
    edge_index_o_val, edge_cuva_val,_ = get_curvature(unique_entity, val_positive[:,:2])
    
    #get curvature for data loader (so that it can be accessed in the test for analysis purposes)
    train_cuva = []
    edge_to_curv = {}
    for e in range(edge_index_o.shape[1]):
        u = edge_index_o[0, e].item()
        v = edge_index_o[1, e].item()
        key = tuple(sorted((int(u), int(v))))
        edge_to_curv[key] = edge_cuva[e].item()

    for e in range(positive[: -(val_size + test_size)].shape[0]):
        u, v = train_data[e][0].item(), train_data[e][1].item()
        key = tuple(sorted((u, v)))
        curv = edge_to_curv[key] 
        train_cuva.append(curv)
    train_cuva = torch.tensor(train_cuva, dtype=torch.float)
    train_negative = negative[: -(val_size + test_size), :2]
    train_cuva_neg = get_proxy_curvature(unique_entity, train_positive[:, :2], train_negative)
    train_cuva = torch.cat([train_cuva, train_cuva_neg], dim=0)
  
    val_cuva = []
    edge_to_curv = {}
    for e in range(edge_index_o_val.shape[1]):
        u = edge_index_o_val[0, e].item()
        v = edge_index_o_val[1, e].item()
        key = tuple(sorted((int(u), int(v))))
        edge_to_curv[key] = edge_cuva_val[e].item()

    for e in range(positive[-(val_size + test_size): -test_size].shape[0]):
        u, v = val_data[e][0].item(), val_data[e][1].item()
        key = tuple(sorted((u, v)))
        curv = edge_to_curv[key] 
        val_cuva.append(curv)
    val_cuva = torch.tensor(val_cuva, dtype=torch.float)
    val_negative = negative[-(val_size + test_size): -test_size, :2]
    val_cuva_neg = get_proxy_curvature(unique_entity, val_positive[:, :2], val_negative)
    val_cuva = torch.cat([val_cuva, val_cuva_neg], dim=0)
    
    #for the test data: positive edges use the curvature computed above via
    #get_negatives (real ORC, inserted into the train+val base graph).
    #Negative edges use get_proxy_curvature -- the same method train/val use
    #for their own negatives -- instead of the old fast/not-fast split, which
    #either skipped negative test curvature entirely (inf placeholder for
    #every pair) or computed it with the expensive insert-based method.
    #get_proxy_curvature is cheap enough to just always run, so test negatives
    #now get real, analyzable curvature values instead of a placeholder.
    test_cuva = []
    edge_to_curv = {}
    for e in range(edge_index_o_test_pos.shape[1]):
        u = edge_index_o_test_pos[0, e].item()
        v = edge_index_o_test_pos[1, e].item()
        key = tuple(sorted((int(u), int(v))))
        edge_to_curv[key] = edge_cuva_test_pos[e].item()

    for e in range(positive[-test_size:].shape[0]):
        u, v = test_data[e][0].item(), test_data[e][1].item()
        key = tuple(sorted((u, v)))
        curv = edge_to_curv[key]
        test_cuva.append(curv)
    test_cuva = torch.tensor(test_cuva, dtype=torch.float)

    # test_negative is exactly negative[-test_size:]'s (u, v) columns, in the
    # same row order they appear in test_data's second half, so the returned
    # curvature tensor can be concatenated directly without a lookup
    test_negative = negative[-test_size:, :2]
    test_cuva_neg = get_proxy_curvature(unique_entity, positive[:-test_size, :2], test_negative)
    test_cuva = torch.cat([test_cuva, test_cuva_neg], dim=0)


    # build data loader
    params = {'batch_size': args.batch, 'shuffle': True, 'num_workers': args.workers, 'drop_last': True}
    
    use_zero_curvature = False

    if use_zero_curvature:
        train_cuva = torch.zeros_like(train_cuva)
        val_cuva = torch.zeros_like(val_cuva)
        test_cuva = torch.zeros_like(test_cuva)
    print("WARNING: Using zero curvatures for testing!")

    #for safety reasons don't add curvature
    training_set = Data_class(train_data, train_cuva) #Data_class(train_data, train_cuva)
    train_loader = DataLoader(training_set, **params)

    validation_set = Data_class(val_data, val_cuva) #Data_class(val_data, torch.zeros(len(val_data)))
    val_loader = DataLoader(validation_set, **params)

    test_set = Data_class(test_data, test_cuva) #Data_class(test_data, test_cuva)
    test_loader = DataLoader(test_set, **params)
    
    # VERIFICATION: Print curvature statistics
    print("\n=== CURVATURE VERIFICATION ===")
    print(f"Train curvatures - Mean: {train_cuva.mean():.4f}, Std: {train_cuva.std():.4f}, Min: {train_cuva.min():.4f}, Max: {train_cuva.max():.4f}")
    print(f"Train curvatures - First 10 values: {train_cuva[:10]}")
    print(f"Val curvatures - Mean: {val_cuva.mean():.4f}, Std: {val_cuva.std():.4f}, Min: {val_cuva.min():.4f}, Max: {val_cuva.max():.4f}")
    print(f"Val curvatures - First 10 values: {val_cuva[:10]}")
    print(f"Test curvatures - Mean: {test_cuva.mean():.4f}, Std: {test_cuva.std():.4f}, Min: {test_cuva.min():.4f}, Max: {test_cuva.max():.4f}")
    print(f"Test curvatures - First 10 values: {test_cuva[:10]}")
    print(f"Number of zero curvatures in train: {(train_cuva == 0).sum()} / {len(train_cuva)}")
    print("=" * 40 + "\n")
    
    # extract features
    print('Extracting features...')
    if args.feature_type == 'one_hot':
        features = np.eye(unique_entity)

    elif args.feature_type == 'uniform':
        np.random.seed(args.seed)
        features = np.random.uniform(low=0, high=1, size=(unique_entity, args.dimensions))

    elif args.feature_type == 'normal':
        np.random.seed(args.seed)
        features = np.random.normal(loc=0, scale=1, size=(unique_entity, args.dimensions))

    elif args.feature_type == 'position':
        adj = sp.coo_matrix((np.ones(train_positive.shape[0]), (train_positive[:, 0], train_positive[:, 1])),
                        shape=(unique_entity, unique_entity), dtype=np.float32)
        adj_o = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        features = adj_o.todense()

    elif args.feature_type == 'curvature':
        adj = sp.coo_matrix((np.ones(train_positive.shape[0]), (train_positive[:, 0], train_positive[:, 1])),
                        shape=(unique_entity, unique_entity), dtype=np.float32)
        adj_o = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        features = adj_o.todense()
        for i in range(edge_index_o.shape[1]):
            u = int(edge_index_o[0, i])
            v = int(edge_index_o[1, i])
        features[u, v] = (1+torch.exp(-edge_cuva[i]))/2  

    features_o = normalize(features)

    args.dimensions = features_o.shape[1]

    x_o = torch.tensor(features_o, dtype=torch.float)
    
    adj = sp.coo_matrix(
        (np.ones(train_positive.shape[0]),
         (train_positive[:, 0], train_positive[:, 1])),
        shape=(unique_entity, unique_entity),
        dtype=np.float32
    )

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    edges = adj.nonzero()
    
    edge_index_o = torch.tensor(
        np.vstack((edges[0], edges[1])),
        dtype=torch.long
    )


    data_o = Data(x=x_o, edge_index=edge_index_o, curva=edge_cuva)

    print('Loading finished!')
    return data_o, train_loader, val_loader, test_loader



