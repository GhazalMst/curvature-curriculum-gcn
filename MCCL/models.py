from torch_geometric.nn import  SAGEConv
from torch_geometric.nn import *
import torch
import torch.nn as nn


class GTNN_outer(torch.nn.Module):
  
 
    def __init__(self, dim, device, gs_dim, additional_feature_dim, nb_classes, use_additional_features):

        super(GTNN_outer, self).__init__()
        print('Layers are on device = {}'.format(device))
        self.dim = dim
        self.device = device
        self.gs_dim = gs_dim
        self.use_additional_features = use_additional_features
        self.num_layers = 1
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(dim, gs_dim).to(device))
    
        if use_additional_features:
            self.hidden_layer_ir_scores = nn.Linear(additional_feature_dim, 2*self.gs_dim).to(device)
            self.hidden_layer_1 = nn.Linear(gs_dim*2 * 2*self.gs_dim, 200).to(device)
            self.hidden_layer_2 = nn.Linear(200, nb_classes).to(device)

        else:   
            self.hidden_layer_1 = nn.Linear(gs_dim*2, 200).to(device)
            self.hidden_layer_2 = nn.Linear(200, nb_classes).to(device)
         
        self.relu = nn.ReLU().to(device)
        self.sigmoid = nn.Sigmoid().to(device)
      
  
    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        for i, layer in enumerate(self.convs):   
            x = layer(x, edge_index)
            x = self.relu(x)            
        x = self.get_minibatch_embeddings(x, batch) # 3 x 400
        x = self.decode(x)
        return x

    def get_minibatch_embeddings(self, x, batch):
        device = x.device
        set_indices, batch_ = batch.set_indices, batch.batch
        num_graphs = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
        num_nodes = torch.eye(num_graphs, device=device)[batch_].sum(dim=0)
        zero = torch.tensor([0], dtype=torch.long).to(device)
        index_bases = torch.cat([
            zero.view(-1), 
            torch.cumsum(num_nodes, dim=0).view(-1)[:-1]
        ])
        #next to line to fix error, don't know if make sense
        set_graph_assignment = torch.zeros(set_indices.size(0), dtype=torch.long, device=device)
        index_bases = index_bases[set_graph_assignment]
        
        index_bases = index_bases.unsqueeze(1).expand(-1, set_indices.size(-1))
        #print(f"DEBUG: index_bases shape = {index_bases.shape}, set_indices shape = {set_indices.shape}")
        assert(index_bases.size(0) == set_indices.size(0))
        set_indices_batch = index_bases + set_indices
        set_indices_batch = set_indices_batch.long()
        x = x[set_indices_batch]  # shape [B, set_size, F]
        x = self.fusion(x, batch.ir_score) 
        return x

    def fusion(self,x, feature_vecs):
        if self.use_additional_features:
        
            x = x.reshape(-1,2*self.gs_dim)
            S = self.hidden_layer_ir_scores(feature_vecs.float())   # S dim = 4
            S = self.relu(S)  
    
            outer_product = torch.bmm(x.unsqueeze(2), S.unsqueeze(1)) 
            x = outer_product.reshape(outer_product.shape[0], -1)  
        
        else:
       
            x = x.reshape(-1,2*self.gs_dim)         
      
        return x  
      
    def decode(self, x):
        logits = self.hidden_layer_1(x)
        logits = self.relu(logits)
        logits = self.hidden_layer_2(logits)
        logits = logits.squeeze(dim=1)
     
        return logits



