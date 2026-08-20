from torch_geometric.data import DataLoader
metric_to_consider = [
    'avg_eigenvector_centrality', 
    'avg_degree', 
    'subgraph_assortativity',      
    'subgraph_density',              
    'is_local_bridge', 
    'subgraph_treewidth',            
    'avg_closeness_centrality',      
    'avg_neighbor_degree', 
    'avg_katz_centrality',          
    'group_degree_centrality',
    'ollivier_ricci_curvature'
]

metric_sort = {}
metric_dataloader = {}

#@profile
def sort_metric_dataset(df_train_metric,args):
    
    global metric_sort
    global metric_to_consider
    
    metric_sort.clear()
    
    for m in metric_to_consider:
        if "A" in args.metric_order:
            metric_sort[m + "_A"] = df_train_metric.sort_values(by=m,ascending=True).index.tolist()
        
        if "D" in args.metric_order:
            metric_sort[m + "_D"] = df_train_metric.sort_values(by=m,ascending=False).index.tolist()
        
    
    if args.add_random:
        if "A" in args.metric_order:
            metric_sort["random_A"] = df_train_metric.sort_values(by="random",ascending=True).index.tolist()
        if "D" in args.metric_order:
            
            metric_sort["random_D"] = df_train_metric.sort_values(by="random",ascending=False).index.tolist()
        

def trim_dataset_as_per_competency(train_set, metric, c, order):
    if "A" == order:
            key = metric + "_A"
    if "D" == order:       
            key = metric + "_D"
   
    D = metric_sort[key]
    nb_examples = int(c * len(train_set))
    
    return [train_set[i] for i in D[:nb_examples]]



def revise_metric_dataloader_combined(epoch, bs, train_set, c,metrics):
    combined_examples_idx = []
    for m in metrics:
        D = metric_sort[m]
        nb_examples = int(c * len(train_set))
        combined_examples_idx.extend(D[:nb_examples])

    combined_examples_idx = list(set(combined_examples_idx))
    combined_examples = [train_set[i] for i in combined_examples_idx]
    print("epoch = {} , number of combined examples = {}".format(epoch, len(combined_examples)))
    return DataLoader(combined_examples, batch_size=bs, shuffle=True, pin_memory=True, num_workers=0)

def revise_metric_dataloader(bs, train_set, c, args):
    global metric_dataloader
    global metric_to_consider

    
    for m in metric_to_consider:
        if "A" in args.metric_order:
            
            metric_dataloader[m + "_A"] = DataLoader(trim_dataset_as_per_competency(train_set, m, c, "A"), batch_size=bs, shuffle=True, pin_memory=True, num_workers=0)
            
        if "D" in args.metric_order:
            
            metric_dataloader[m + "_D"] = DataLoader(trim_dataset_as_per_competency(train_set, m, c, "D"), batch_size=bs, shuffle=True, pin_memory=True, num_workers=0)
            
    
    if args.add_random:
        metric_dataloader["random" + "_A"] = DataLoader(trim_dataset_as_per_competency(train_set, "random", c, "A"), batch_size=bs, shuffle=True, pin_memory=True, num_workers=0)
        metric_dataloader["random" + "_D"] = DataLoader(trim_dataset_as_per_competency(train_set, "random", c, "D"), batch_size=bs, shuffle=True, pin_memory=True, num_workers=0)

        
def revise_metric_dict(args):

        
    asc_dict = {}
    desc_dict = {}
    q_dict = {}
    metric_dict = {}
    if args.use_k_means:
        if args.dataset == "huri":
            selected_keys = ['avg_eigenvector_centrality', 'avg_degree', 'subgraph_assortativity', 'subgraph_density', 'is_local_bridge', 'subgraph_treewidth',        'avg_closeness_centrality', 'avg_neighbor_degree', 'avg_katz_centrality', 'group_degree_centrality', 'ollivier_ricci_curvature']
            
            for m in selected_keys:
                if "A" in args.metric_order:
                    asc_dict[m +"_A"] = metric_dataloader[m + "_A"]
                if "D" in args.metric_order:
                    asc_dict[m +"_D"] = metric_dataloader[m + "_D"]
        elif args.dataset == "pgr":
            pgr_k_means_metric = ["add_average_degree_connectivity", "add_eigenvector_centrality_numpy", "degree", "degree_assortativity_coefficient", "density",
                                  "len_local_bridges", "mean_degree_mixing_matrix", "node_connectivity", "treewidth_min_degree", "add_closeness_centrality"] 
            for m in pgr_k_means_metric:
                if "A" in args.metric_order:
                    asc_dict[m +"_A"] = metric_dataloader[m + "_A"]
                if "D" in args.metric_order:
                    desc_dict[m +"_D"] = metric_dataloader[m + "_D"]

                    
        elif args.dataset == "gdpr":
            omim_k_means_metric = ["add_avg_neighbor_deg", "add_average_degree_connectivity", "add_eigenvector_centrality_numpy", "large_clique_size", "degree_assortativity_coefficient",
                                  "density", "add_katz_centrality_numpy", "group_degree_centrality", "treewidth_min_degree", "add_closeness_centrality"]
            for m in omim_k_means_metric:
                if "A" in args.metric_order:
                    asc_dict[m +"_A"] = metric_dataloader[m + "_A"]
                if "D" in args.metric_order:
                    desc_dict[m +"_D"] = metric_dataloader[m + "_D"]

    else:
        
        for m in metric_to_consider:
            if "A" in args.metric_order:
                asc_dict[m +"_A"] = metric_dataloader[m + "_A"]
            if "D" in args.metric_order:
                desc_dict[m +"_D"] = metric_dataloader[m + "_D"]     
                
    if args.add_random:
        
            print("random added")
            if "A" in args.metric_order:
                asc_dict["random" +"_A"] = metric_dataloader["random" + "_A"]
            if "D" in args.metric_order:
                 desc_dict["random" +"_D"] = metric_dataloader["random" + "_D"] 

    metric_dict.update(asc_dict)
    metric_dict.update(desc_dict)
    metric_dict.update(q_dict)
                
    
    
    return metric_dict