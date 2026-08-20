import utils, models, g_metric
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
#from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as StandardDataLoader
from torch_geometric.data import Data
from torch.utils.data import Subset
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import evaluate
from sklearn.utils.class_weight import compute_class_weight
import sys
from sklearn.preprocessing import normalize
import numpy as np
import gc

class Data_class(Dataset):
    def __init__(self, triple):
        # Konvertiere das gesamte Array einmalig in Tensoren, 
        # das ist performanter als in jedem Schritt __getitem__
        self.entity1 = torch.tensor(triple[:, 0], dtype=torch.long)
        self.entity2 = torch.tensor(triple[:, 1], dtype=torch.long)
        self.label = torch.tensor(triple[:, 2], dtype=torch.float32)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # Gib die Daten als echte Tensoren zurück
        label = self.label[index]
        # Wir geben die Knoten als Tensor-Paar zurück, damit der DataLoader 
        # sie zu einem Batch der Form [2, Batch_Size] oder zwei separaten Batches stapeln kann
        nodes = torch.tensor([self.entity1[index], self.entity2[index]], dtype=torch.long)
        
        return label, nodes

bce_loss = nn.BCEWithLogitsLoss(reduction='none')
cat_loss = nn.CrossEntropyLoss(reduction='none')
optimizer = None
forward_pass_loss = {} 
forward_key2prediction = {}
forward_labels = {}
key2idx = {}
forward_proba = {}

node_feature_matrix = None

def init_model(args):
    global node_feature_matrix
    device = args.device
    node_feature_dim = 300
    gs_dim = 100
    use_additional_features = args.add_additional_feature
    
    if args.dataset == "huri":
        node_feature_dim = 8272
        gs_dim = 100
        add_additional_feature_dim = 768
        nb_classes = 1
        
    if args.dataset == "pgr":
        node_feature_dim = 300
        gs_dim = 100
        add_additional_feature_dim = 768
        nb_classes = 1

    if args.dataset == "gdpr":
        node_feature_dim = 300
        gs_dim = 100
        add_additional_feature_dim = 4
        nb_classes=1

    

    model = models.GTNN_outer(node_feature_dim, device, gs_dim, add_additional_feature_dim,nb_classes, use_additional_features)

    return model

def get_weighted_loss(loss, label, device):
    classes = [l.item() for l in label]

    if len(classes) > sum(classes) > 0 :
        cls_weight = compute_class_weight("balanced", [0,1], classes)
        cls_weight = cls_weight.tolist()
        weight = torch.tensor(cls_weight).to(device)
        weight_ = weight[label.data.view(-1).long()].view_as(label)
        loss = loss * weight_
    return loss

batch_loss_history = {}
global_batch_counter = 0

def get_key2idx(train_set):
    global key2idx
    for idx,t in enumerate(train_set):
        #t.key[-1] = str(t.key[-1])
        #t.key = [str(k) for k in t.key]
        #key2idx[" ".join(t.key)] = idx
        # Falls t.key eine Liste von Zahlen/Tensor-Werten ist:
        tuple_k = tuple(int(x) for x in t.key)
        key2idx[tuple_k] = idx

def seperate_seen_from_new_edges(data_loader,train_set):
    edge_to_be_processed = []
    pre_computed_loss_values = []
    predictions = []
    labels = []
    proba = []
    for batch in data_loader:
        keys = batch.key
        for k in keys:
            #k = [str(j) for j in k]
            #k[-1] = str(k[-1])
            #str_k = " ".join(k)
            
            # Konvertiere das Key-Array direkt in ein Integer-Tupel
            tuple_k = tuple(int(j) for j in k)
            
            if tuple_k not in forward_pass_loss:             #if forward_pass_loss.get(str_k) is None:
                edge = train_set[key2idx[tuple_k]]
                edge_to_be_processed.append(edge)
            else:
                pre_computed_loss_values.append(forward_pass_loss[tuple_k])
                predictions.append(forward_key2prediction[tuple_k])
                labels.append(forward_labels[tuple_k])
                proba.append(forward_proba[tuple_k])

    return pre_computed_loss_values, edge_to_be_processed, predictions, labels, proba


def calculate_error_metric(model, data_loader, device,  train_set, args):
    global forward_pass_loss
    global forward_key2prediction
    global forward_labels
    global forward_proba

    saved_computed_loss_values, edge_to_be_processed, saved_predictions, saved_labels, saved_proba = seperate_seen_from_new_edges(
        data__loader, train_set)
    # print(len(edge_to_be_processed))
    #updated_dataloader = DataLoader(edge_to_be_processed, batch_size=32, shuffle=False, pin_memory=True, num_workers=0)
    if len(edge_to_be_processed) > 0:
        subset_train_set = Subset(train_set, edge_to_be_processed)
        updated_dataloader = DataLoader(subset_train_set, batch_size=32, shuffle=False, pin_memory=True, num_workers=0)
    else:
        print('error')

    model.eval()
    batch_loss = []
    batch_predictions = []
    batch_labels = []
    batch_proba = []

    with torch.no_grad():
        counter = 0
        for batch in updated_dataloader:

            batch = batch.to(device)
            label = batch.y

            prediction = model(batch) # gdpr / pgr -- logits
            prediction_proba = F.sigmoid(prediction)
            batch_predictions.extend(prediction_proba.cpu().numpy().tolist())
            batch_proba.extend(prediction_proba.cpu().numpy().tolist())

            loss = bce_loss(prediction_proba.float(), label.float())
            
            #update_loss_history(batch, loss)
        
            loss = loss.cpu().detach()

            batch_loss.extend(loss.cpu().numpy().tolist())
            batch_labels.extend(label.cpu().numpy().tolist())

            for key, l, p, lb, pb in zip(batch.key, loss, prediction_proba, label, prediction_proba):
                #key[-1] = str(key[-1])
                #str_k = " ".join(key)
                tuple_k = tuple(int(x) for x in key)
                
                forward_pass_loss[str_k] = l.item() 
                forward_key2prediction[str_k] = p.detach().cpu().item() if p.numel() == 1 else p.detach().cpu().numpy()
                forward_labels[str_k] = lb.item()
                forward_proba[str_k] = pb.detach().cpu().item() if pb.numel() == 1 else pb.detach().cpu().numpy()

    if len(edge_to_be_processed) > 0:
        if len(saved_labels) > 0:
            #combined_loss = batch_loss + [i.item() for i in saved_computed_loss_values]

            #combined_predictions = batch_predictions + [i.item() for i in saved_predictions]

            #combined_proba = batch_proba + [i.item() for i in saved_proba]

            #combined_labels = batch_labels + [i.item() for i in saved_labels]
            combined_loss = batch_loss + saved_computed_loss_values
            combined_predictions = batch_predictions + saved_predictions
            combined_proba = batch_proba + saved_proba
            combined_labels = batch_labels + saved_labels
        else:
            combined_loss = batch_loss
            combined_predictions = batch_predictions
            combined_proba = batch_proba
            combined_labels = batch_labels
    else:
        #combined_loss = [i.item() for i in saved_computed_loss_values]
        #combined_predictions = [i.item() for i in saved_predictions]
        #combined_labels = [i.item() for i in saved_labels]
        #combined_proba = [i.item() for i in saved_proba]
        combined_loss = saved_computed_loss_values
        combined_predictions = saved_predictions
        combined_labels = saved_labels
        combined_proba = saved_proba
        
    combined_loss = np.array(combined_loss)
    combined_predictions = np.array(combined_predictions)
    combined_labels = np.array(combined_labels)
    combined_proba = np.array(combined_proba)


    
    performance = None

    return combined_loss.mean(), combined_loss, performance, combined_proba
    # return sum(batch_loss) / len(batch_loss), batch_loss, performance, batch_proba.cpu().numpy()



def update_competency(t, T, c_0, p):
    term = pow(((1 - pow(c_0,p))*(t/T)) + pow(c_0,p), (1/p))
    return min([1,term])


def calculate_error_metric_using_complexity_score(metric, c, train_set):
    if metric.endswith("_A"):
        metric = metric.replace("_A", "")
        D = df_train_metric.sort_values(by=metric,ascending=True).index.tolist()
    if metric.endswith("_D"):
        metric = metric.replace("_D", "")
        D = df_train_metric.sort_values(by=metric,ascending=False).index.tolist()
        
    nb_examples = int(c * len(train_set))
    D_trim = D[:nb_examples]
    return df_train_metric[metric][D_trim].mean(), None

df_train_metric = None


def fix_negative_value_if_any(values):
    tmp = np.array(values)
    min_value = tmp.min()
    if min_value < 0:
        print("fixing negative values")
        tmp = tmp + (-1*min_value)
    return tmp

def read_train_metric(args):
    
    if args.dataset == "huri":
        df_train_metric = pd.read_csv("../df/indices_small.csv")
        print("using l2 norm")
        for c in df_train_metric.columns.tolist()[4:]:
            
            tmp = fix_negative_value_if_any(df_train_metric[c])
            df_train_metric[c] = tmp 
            df_train_metric[c] = normalize(df_train_metric[c].values[:, np.newaxis], axis=0)
    
    elif args.dataset == "pgr":
        df_train_metric = pd.read_csv("../indices/pgr_col_indices.csv")
        print("using l2 norm")
        for c in df_train_metric.columns.tolist()[4:]:
            
            tmp = fix_negative_value_if_any(df_train_metric[c])
            df_train_metric[c] = tmp 
            df_train_metric[c] = normalize(df_train_metric[c].values[:, np.newaxis], axis=0)
            
    elif args.dataset == "gdpr":
        df_train_metric = pd.read_csv("../indices/omim_col_indices.csv")
        df_train_metric = df_train_metric.fillna(0)
        print("using l2 norm")
        for c in df_train_metric.columns.tolist()[2:]:
            tmp = fix_negative_value_if_any(df_train_metric[c])
            df_train_metric[c] = tmp 
            df_train_metric[c] = normalize(df_train_metric[c].values[:, np.newaxis], axis=0)
     
    if args.add_random:
        print("random added")
        df_train_metric['random'] = np.random.rand(len(df_train_metric)).tolist()
        df_train_metric['random'] = normalize(df_train_metric['random'].values[:, np.newaxis], axis=0)
    
    return df_train_metric


#@profile
def train_model_w_competence(train_set, eval_set,  args):
    global df_train_metric
    global forward_pass_loss
    global optimizer
    
    val_set = eval_set
    df_train_metric = read_train_metric(args)
    get_key2idx(train_set)
        
    approach = args.prioritizing_approach
    bs = args.batch_size
    c_0 = 0.1

        
    g_metric.sort_metric_dataset(df_train_metric, args)
    g_metric.revise_metric_dataloader(bs, train_set, c_0, args)
    metric_dict_train = g_metric.revise_metric_dict(args)

    lr = args.lr
    L2 = args.l2
    device = args.device
    model = init_model(args)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=L2)


    # to select 1st bucket for the training
    metric_dict = metric_dict_train

    split_set = train_set

    metric_loss_values = []
    metrics = []
    
    print('test before metrics')
    for metric in metric_dict: 
        print(metric)
        if approach == "loss_based":
            metric_loss,metric_loss_vec, s_e, _ = calculate_error_metric(model, metric_dict[metric], device, split_set, args)
               
        elif approach == "complexity_score_based":
            metric_loss, s_e = calculate_error_metric_using_complexity_score(metric, c_0, split_set)
                
        metric_loss_values.append(metric_loss)
        metrics.append(metric)
    print('test after metrics')
            


    # train model on the revise dataset based on competency

    T = args.curriculum_length
    best_pref = -1.0
    
    global_x = train_set[0].x.to(device)
    global_edge_index = train_set[0].edge_index.to(device)
    
    for t in range(1, T):
        
        ("*"*80)
        
        forward_pass_loss = {}
        forward_key2prediction = {}
        forward_labels = {}
        forward_proba = {}
  
        if args.loss_creteria == 'max':
            j = metrics[np.argmax(metric_loss_values)]
        else:
            j = metrics[np.argmin(metric_loss_values)]
        

        final_dataloader = metric_dict_train[j]
        model.train()
        losses = []
        for batch in final_dataloader:
                
            optimizer.zero_grad()
            #batch = batch.to(device)
            #label = batch.y

            #prediction = model(batch)
             
            #try to fix memory problem
            target_edges = batch.set_indices.squeeze(1).to(device) if hasattr(batch, 'set_indices') else batch.edge_index.to(device)
            labels = batch.y.to(device)
            model_input = Data(
                x=global_x,
                edge_index=global_edge_index,
                set_indices=target_edges,
                y=labels
            )
            if hasattr(batch, 'ir_score'):
                model_input.ir_score = batch.ir_score.to(device)
                
            prediction = model(model_input)
            label = labels
            
            loss = bce_loss(prediction.float(), label.float())
                
            if args.dataset == "gdpr":
                
                loss = get_weighted_loss(loss, label, device)
                
            
            #update_loss_history(batch, loss)
            loss = loss.mean()
            losses.append(loss.item())
           
            print(t, j, loss.item())
            loss.backward()
            optimizer.step()
 
            del model_input
            del batch
            del loss
            del prediction

        # save model on every competency
        
        eval_data_loader = DataLoader(eval_set, batch_size=16, shuffle=True, pin_memory=True, 
                                      num_workers=0) #batch_size=args.batch_size
        
       
        auroc, p, r, f1, predictions, predictions_proba = evaluate.eval_model(model, eval_data_loader, device)
        eval_ds_name = "test" if args.evaluate_test_per_epoch else "val"
        print(
                "eval {0} t = {1:d} loss = {2:.6f} p = {3:.4f} r = {4:.4f} f1 = {5:.4f} auroc = {6:.4f} ".format(eval_ds_name,t, sum(losses) / len(losses), p,
                                                                                        r, f1, auroc))
        current_pref = f1
            
        if current_pref > best_pref or t == 0:
            best_pref = current_pref
            utils.save_the_best_model(model, t, optimizer, {"p": p, "r": r, "f1": f1}, args)
        else:
            pass
        sys.stdout.flush()

        # update competency AND calculates the error for next bucket
        
        c = update_competency(t, T, c_0, 2)
        print(t, c)
        g_metric.revise_metric_dataloader(bs, train_set, c, args)
        metric_dict_train = g_metric.revise_metric_dict(args)
        metric_dict = metric_dict_train
        split_set = train_set
    
        print('test second metric')
        metric_loss_values = []
        metrics = []
        for metric in metric_dict:
                    
                if approach == "loss_based":
                    # 3. FIX typo check: Make sure this argument reads 'metric_dict[metric]' 
                    # and verify it uses the internal dataloader correctly
                    metric_loss, metric_loss_vec, s_e, proba_vec = calculate_error_metric(model, metric_dict[metric], device, split_set, args)
                elif approach == "complexity_score_based":
                    metric_loss, s_e = calculate_error_metric_using_complexity_score(metric, c, split_set)
                metric_loss_values.append(metric_loss)
                metrics.append(metric)
                
        #clean up
        final_dataloader = None
        metric_dict_train = None
        metric_dict = None
        forward_pass_loss.clear()
        forward_key2prediction.clear()
        forward_labels.clear()
        forward_proba.clear()
        g_metric.metric_dataloader.clear()
        del predictions, predictions_proba, eval_data_loader
        gc.collect()
        torch.cuda.empty_cache()
        
    return model


def train_model_gtnn(train_set, eval_set,  args):
    approach = args.prioritizing_approach
    bs = args.batch_size

    global optimizer

    lr = args.lr
    L2 = args.l2
    device = args.device
    model = init_model(args)


    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=L2)
    
    global_x = train_set[0].x.to(device)
    global_edge_index = train_set[0].edge_index.to(device)


    T = args.curriculum_length
    best_pref = -1.0
    #train_data = Data_class(train_set)
    #train_data_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True,
    #                                  num_workers=0)
    train_data_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)
        
    for t in range(1, T):
        
        model.train()
        losses = []
        for batch in train_data_loader:
        #for i, (label, inp) in enumerate(train_data_loader):
            #label = label.to(device)
    
            optimizer.zero_grad()
        
            target_edges = batch.set_indices.squeeze(1).to(device)
            labels = batch.y.to(device)
            model_input = Data(
                x=global_x,
                edge_index=global_edge_index,
                set_indices=target_edges,
                y=labels
            )
            if hasattr(batch, 'ir_score'):
                model_input.ir_score = batch.ir_score.to(device)
            
            #batch = batch.to(device)

            prediction = model(model_input)  #model(batch)
            
            loss = bce_loss(prediction.float(), labels.float())
            #update_loss_history(batch, loss)
            if args.dataset == "gdpr":
                loss = get_weighted_loss(loss, labels, device)

            loss = loss.mean()
            losses.append(loss.item())
            print(t,loss.item())
            loss.backward()
            optimizer.step()
            loss = loss.detach().cpu()
            loss = None
            prediction = None
            batch = None
            del model_input
            del batch
            del loss
            del prediction
            torch.cuda.empty_cache()


        # save model on every competency
        eval_data_loader = DataLoader(eval_set, batch_size=16, shuffle=True, num_workers=0) #batch_size=args.batch_size
        #DataLoader(eval_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)

        
        auroc, p, r, f1, _, _ = evaluate.eval_model(model, eval_data_loader, device)
        eval_ds_name = "test" if args.evaluate_test_per_epoch else "val"
        print(
                "eval {0} t = {1:d} loss = {2:.6f} p = {3:.4f} r = {4:.4f} f1 = {5:.4f} auroc = {6:.4f} ".format(eval_ds_name, t, sum(losses) / len(losses), p,
                                                                                        r, f1, auroc))
        current_pref = f1
            
        if current_pref > best_pref or t == 0:
            best_pref = current_pref
            utils.save_the_best_model(model, t, optimizer, {"p": p, "r": r, "f1": f1}, args)
        else:
            pass
        
        sys.stdout.flush()

        #del eval_graph_list, pyg_eval_loader
        gc.collect()
        torch.cuda.empty_cache()

    return model