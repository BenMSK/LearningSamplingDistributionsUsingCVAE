import numpy as np
import random
import os 
import torch
import torch.nn.functional as F
import pickle
from torch.utils.data import Dataset
import cv2

def img2mp4(image_set, pathOut , fps =10 ) :
    height, width, layers = image_set[0].shape
    size = (width, height)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'MP4V'), fps, size)
    for img in image_set:
        # writing to a image array
        out.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    out.release()
    
def GetWorkspace(random_seed, env_dir):    
    random.seed(random_seed)
    np.random.seed(random_seed)

    # List of data directories where raw data resides
    try:
        train_workspace = os.listdir(env_dir + "/train")
        val_workspace = os.listdir(env_dir + "/validation")
        test_workspace = os.listdir(env_dir + "/test")
        np.random.shuffle(train_workspace)# By random seed.
    except:
        print("Invalid workspace dir. Process is died")
        quit()
    return (train_workspace, val_workspace, test_workspace)

def PreprocessData(train_val_test_ratio, data_path):
    tr_ratio, val_ratio, te_ratio = train_val_test_ratio
    f = open(data_path, "rb")
    dataset = pickle.load(f)
    f.close()
    print("Load the dataset... (total#: {})".format(len(dataset)))
    train_fraction = tr_ratio + val_ratio
    print("Data fraction is (train: {}%, val: {}%, test: {}%)"\
            .format(tr_ratio*100, val_ratio*100, te_ratio*100))

    # List of data directories where raw data resides
    np.random.shuffle(dataset)
    dataset_cnt = len(dataset)

    # Divide the datasets to {train, val, test}
    train_dataset = dataset[: int(dataset_cnt * tr_ratio)]
    val_dataset = dataset[int(dataset_cnt * tr_ratio): int(dataset_cnt * train_fraction)]
    test_dataset = dataset[int(dataset_cnt * train_fraction) :]
    return train_dataset, val_dataset, test_dataset


class DataSet(Dataset):   
    def __init__(self, dataset, device):        
        self.dataset = dataset
        self.device = device
        self.total_data_num = len(self.dataset)

    def __len__(self):
        return self.total_data_num

    def __getitem__(self, idx):
        return self.dataset[idx]

    def rollout(self, samples):
        # start_goal_batch = []
        env_batch = []
        label_batch = []
        for (env, label) in samples:
            env_batch.append([env]) # image width x height
            label_batch.append([label[0], label[1]])
        env_batch = torch.from_numpy(np.array(env_batch)).float().to(self.device)
        label_batch = torch.from_numpy(np.array(label_batch)).float().to(self.device)
        return env_batch, label_batch


def dfs(visited, edge_sequence, graph, node, parent, x_lims, y_lims):  #function for dfs 
    if node not in visited:
        visited.add(node)
        if parent is not None:
            norm_parent = state_normalize(parent, x_lims, y_lims)
            norm_child = state_normalize(node, x_lims, y_lims)
            edge_sequence.append([norm_parent[0], norm_parent[1], norm_child[0], norm_child[1]])
        for neighbour in graph[node]:
            dfs(visited, edge_sequence, graph, neighbour, node, x_lims, y_lims)


def build_edge_sequence(tree, x_lims, y_lims):
    visited = set()
    graph = tree._graph
    root = tree._root
    norm_root = state_normalize(root, x_lims, y_lims)
    edge_sequence = [[norm_root[0], norm_root[1], norm_root[0], norm_root[1]]]
    dfs(visited, edge_sequence, graph, root, None, x_lims, y_lims)
    return np.array(edge_sequence) # shape: N x 2

def state_normalize(s, x_lims=(0, 100), y_lims=(0, 100)):
    s = ( (s[0] - x_lims[0]) / (x_lims[1] - x_lims[0]), (s[1] - y_lims[0]) / (y_lims[1] - y_lims[0]) )
    return s

def state_upscaling(s, x_lims=(0, 100), y_lims=(0, 100)):
    s = ( s[0] * (x_lims[1] - x_lims[0]) + x_lims[0], s[1] * (y_lims[1] - y_lims[0]) + y_lims[0])
    return s

def _to_numpy(tensor):
    return tensor.data.cpu().numpy()
