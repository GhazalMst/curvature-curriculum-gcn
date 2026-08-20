from torch.optim import Adam
from layer import CGCN
import torch
import numpy as np
import random

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Für absolute Reproduzierbarkeit (kann die Performance leicht senken)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

def Create_model(args, num_layers=1):
    set_seed(args.seed)
    model = CGCN(aggregator=args.aggregator, feature=args.dimensions, hidden1=args.hidden1, hidden2=args.hidden2, decoder1=args.decoder1, dropout=args.dropout, num_layers=num_layers)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return model, optimizer