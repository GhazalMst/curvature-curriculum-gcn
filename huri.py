import pandas as pd
import torch
import requests
import time
import io
import numpy as np
#from bio_embeddings.embed import ProtTransBertBFDEmbedder,SeqVecEmbedder
from transformers import AutoTokenizer, EsmModel
from tqdm import tqdm



def build_data():
     # load data
    df = pd.read_csv('data/HuRI.tsv', sep='\t', header=None, names=['ProteinA', 'ProteinB'])

    # find unique proteins
    unique_proteins = sorted(list(set(df['ProteinA']).union(set(df['ProteinB']))))
    print(f"Number of unique proteins (nodes): {len(unique_proteins)}")
    
    df_seq = pd.read_csv('data/idmapping.tsv', sep='\t')
    df_seq_clean = df_seq[['From','Entry']][df_seq['Reviewed']=='reviewed']
    df_seq_clean = df_seq_clean.drop_duplicates(subset=['From'], keep='first') #alternatively keep longes sequence

    
    #add 0 if no sequence
    df_final = pd.DataFrame({'Protein_ID': unique_proteins})
    df_final = pd.merge(df_final, df_seq_clean, left_on='Protein_ID', right_on='From', how='left')
    df_final = df_final.fillna(0)
    df_final = df_final.drop(columns=['From','Protein_ID'])
    
    # dict for protein
    node_mapping = {protein_id: index for index, protein_id in enumerate(unique_proteins)}

    # build edgelist
    #source_nodes = [node_mapping[p] for p in df['ProteinA']]
    #target_nodes = [node_mapping[p] for p in df['ProteinB']]
    #edges = list(zip(source_nodes, target_nodes))
    #np.savetxt('data/HuRI2.edgelist', edges, fmt='%d', delimiter=' ')
    #print("Datei 'data/HuRI2.edgelist' wurde erfolgreich gespeichert.")
    
    return df_final

def get_embedding(sequence, model, tokenizer, device):
    embedding_dim = model.config.hidden_size
    if sequence == 0 or not isinstance(sequence, str):
        return np.zeros(embedding_dim, dtype=np.float32)
    inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        
    #mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy().flatten()

def get_features():
    df_final = build_data()
    
    model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    #embedder = SeqVecEmbedder()
    
    tqdm.pandas()
    #get_embedding('Q9H2S6', model, tokenizer, device)
    df_final['Embedding'] = df_final['Entry'].progress_apply(
        lambda x: get_embedding(x, model, tokenizer, device)
    )
    
    feature_matrix = np.stack(df_final['Embedding'].values).astype(np.float32)
    return feature_matrix
    
def main():
    get_features()
    

if __name__ == "__main__":
    main()