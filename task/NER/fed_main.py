import pandas as pd
import os
import torch
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "False"

from models.BERT import BertModel
from datasets.dataset import DataSequence
from trainer.trainer import train_loop, NER_FedAvg
import random
import numpy as np
from collections import defaultdict



import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=str, help="WORKSPACE folder", default="site-1")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    args = vars(parse_args())
    saved_dir = args['workspace']


    dls = defaultdict(lambda: {})
    model_name='bert-base-uncased'
    root_dir = "./data/2018_Track_2_ADE_and_medication_extraction_challenge/"
    for idx, dataset_name in enumerate(["site-1", "site-2"]):
        df_train = pd.read_csv(os.path.join(root_dir, dataset_name+"_train.csv"))
        df_val = pd.read_csv(os.path.join(root_dir, dataset_name+"_val.csv"))
        train_dataset = DataSequence(df_train, model_name=model_name)
        val_dataset = DataSequence(df_val, model_name=model_name)
        dls[idx]['train'] = DataLoader(train_dataset, num_workers=4, batch_size=32, shuffle=True)
        dls[idx]['validation'] = DataLoader(val_dataset, num_workers=4, batch_size=32)

    device = torch.device("cuda")

    print("get model ready")

    fed_model = NER_FedAvg(
                dls=dls,
                client_weights = [0.5, 0.5], 
                lrs = [5e-5, 5e-5], 
                max_epoches=30, 
                aggregation_freq=1,
                device=device, 
                saved_dir = saved_dir,
                model_name=model_name
                )
    fed_model.fit()