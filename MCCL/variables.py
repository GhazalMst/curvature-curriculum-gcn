import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''
Models are saved under "saved_models"
Logs are written in "logs" directory

'''
dir_model = "/raven/u/clajo/MCCL/saved_models"
dir_logs = "/raven/u/clajo/MCCL/logs"
dir_data = "/raven/u/clajo/data"