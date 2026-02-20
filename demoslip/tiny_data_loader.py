import pickle
from torch.utils.data import DataLoader
import torch
import numpy as np


from .dataset_poper import SlipNoise_Welch_CustomDataset


def load_demo_segments(pkl_path: str):
    with open(pkl_path, "rb") as f:
        demo_segments = pickle.load(f)
    # expects: list[dict]
    return demo_segments


# ->> entry point
def make_demo_loader(cfg, pkl_path, batch_size=1):
    demo_segments = load_demo_segments(pkl_path)

    clip_len = None if cfg.dataset.training.test_clip_mode == "full_len" else cfg.dataset.training.clip_len


    dataset = SlipNoise_Welch_CustomDataset(
        demo_segments,
        set_mode="test",
        clip_len=clip_len,
        cfg=cfg,
    )

    dataloader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda samples: custom_info_collate(samples, cfg),
        num_workers=0,
    )

    return dataloader






def custom_info_collate( samples, cfg): 

    idx = 0
    data = [ torch.tensor(s[idx]) for s in samples] 
    data = torch.stack(data)
    data_vanilla = data 
    

    
    idx +=1
    dense_slip_label = [ torch.tensor(np.array(s[idx])) for s in samples]
    dense_slip_label = torch.stack(dense_slip_label)


    idx +=1
    dense_noise_label = [ torch.tensor(np.array(s[idx])) for s in samples]
    dense_noise_label = torch.stack(dense_noise_label)

    
    idx +=1
    dense_torques = [ torch.tensor(s[idx]) for s in samples]
    dense_torques = torch.stack(dense_torques) # stacks on B -> B, 2, L 

    

    idx +=1
    pze_col_name_embedding = [ torch.tensor(s[idx]) for s in samples]
    pze_col_name_embedding = torch.stack(pze_col_name_embedding)


    idx +=1
    sup = [ s[idx] for s in samples]



    pze_col_name_embedding = pze_col_name_embedding.unsqueeze(2)  # Shape: (batch, channel, 1)
    pze_col_name_embedding = pze_col_name_embedding.repeat(1, 1, data.size(2))  # Shape: (batch, channel, length)



    conscious_data = [data]
    
    if cfg.net_extras.conscious_torques == "True":         
        dense_torques_diff = dense_torques.diff(dim=2, prepend=torch.zeros(dense_torques.size(0),dense_torques.size(1),1))
        conscious_data.append(dense_torques)
        conscious_data.append(dense_torques_diff)

    if cfg.net_extras.conscious_sensor  == "True": 
        conscious_data.append(pze_col_name_embedding)


    conscious_data = torch.concatenate( conscious_data, dim = 1) # BCL , concat on C ...
    

    
    
    return  data_vanilla, dense_slip_label, dense_noise_label, conscious_data, sup

