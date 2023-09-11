import os
import random as ra
import torch
import argparse
import numpy as np
from src.data_loader import sql_attached_dataloader

from src.model.factory import initialize_model
import matplotlib.pyplot as plt


def load_model(tensorboard_path: str):
    """
    Args:
    tensorboard_path: the path of the directory of tensorboard
    """
    arg_file_path = os.path.join(tensorboard_path, "args.txt")
    model_config = reload_argparse(arg_file_path)
    
    net = initialize_model(model_config)
    
    model_pth_path = os.path.join(tensorboard_path, "best_model.pth")
    saved_state_dict = torch.load(model_pth_path)

    net.load_state_dict(saved_state_dict)
    print("successfully load model")
    return net, model_config

def reload_argparse(file_path:str):
    
    d = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            key, value = line.strip('\n').split(',')
            # print(f"{key}, {value}\n")
            try:
                re = eval(value)
            except:
                re = value
            d[key] = re
    
    return argparse.Namespace(**d)   


def fig(x,y,path):
    plt.figure(figsize=(10, 6))  # Set the figure size
    colors = [plt.cm.viridis(ra.random()) for _ in range(len(y))]
    bar_width = 0.7
    plt.bar(x, y, width=bar_width, color=colors, alpha=0.7)
    plt.title('Beautiful Histogram')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(path, dpi=300, bbox_inches='tight')



if __name__ == '__main__':
    path = '/hdd1/sams/tensor_log/cvd/sparsemax_vertical_sams_balance_K16_alpha_17'
    net, config = load_model(path)
    print(config.workload)
    # load data
    _, _, workload = sql_attached_dataloader(config)
    
    net.eval()
   
    frequency = []
    expert_num_for_each_item = []
    with torch.no_grad():
        for i in range(len(workload)):
            _, sql = workload[i]
            sql = sql.unsqueeze(0) # [1, nfield]
            
            score = net.cal_gate_score(sql) # [1, n]
            score = score.squeeze(0)
            
            nonzero = score.numpy() != 0            
    	    # L = expert nnum
            frequency.append(nonzero)
            expert_num_for_each_item.append(np.sum(nonzero))
            
    frequency = np.array(frequency)
    frequency = np.sum(frequency, axis=0) / len(workload)
    frequency = frequency.tolist()


    print(config.K)
    print(frequency)
    fig(range(len(frequency)), frequency, "./Expert_K16_A17_B001.png")
    fig(range(len(expert_num_for_each_item)), expert_num_for_each_item, "./Item_K16_A17_B001.png")        
        
    
    
