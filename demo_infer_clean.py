"""
demo_infer.py

Minimal demo inference script.
- Config-driven (yaml)
- Reconstructs exact training config from Optuna study (temporary)
- Runs inference on one tiny example
"""

import yaml
import numpy as np
import torch

from pprint import pformat
import pickle


from types import SimpleNamespace


from pathlib import Path
ROOT = Path(__file__).resolve().parent



from demoslip.net import Net

from demoslip.tiny_data_loader  import make_demo_loader
from demoslip.visualizer import visu_demo



#  to remove OmegaConf dependency, handles config dictionary as namespace
def to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [to_ns(x) for x in d]
    return d




# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------


WEIGHTS_PATH = ROOT / "weights" / "slip_net_optuna_carousel_split_seed_999_best_ep_trial_59.pth"

CONFIG_PATH =  ROOT / "demo_config_59.yaml"
DATA_PATH = ROOT / "data" / "demo_thayral_nanoslip_dataset.pkl" # dataset for inference
# DATA_PATH = "enriched_dataset_demo.pkl"

DATA_VISU_PATH = ROOT / "data" / "demo_thayral_visu_dataset.pkl" # fft and  pze signals for visualization



DEVICE = "cpu"


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

@torch.no_grad()
def main():



    print()
    print("---  spectro-temporal slip detection ---")


    # --- load base config
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
        
    # print("Config:")
    # print(pformat(cfg, sort_dicts=True, width=100))
    # print()


    cfg = to_ns(cfg) # namespace format


    

    BAND_TO_FREQ = {
        330: 8 +1, 
        650: 16 +1, 
        1300: 32 +1, 
        2500: 64 +1, 
        5000: 128+1 #end
    }


    assert cfg.net_extras.minfreq == 1
    # cfg.net_extras.minfreq = minf
    cfg.net_extras.maxfreq = BAND_TO_FREQ[cfg.net_extras.max_freq_Hz]




    loader = make_demo_loader(cfg, DATA_PATH, batch_size=1)



    # --- build model
    model = Net(
        dropout_temporal_proba=cfg.net_base.dropout_temporal_proba,
        rnn_dropout_proba=cfg.net_base.rnn_dropout_proba,
        batch_norm=cfg.net_base.batch_norm,
        mid_feature_dim=cfg.net_base.mid_feature_dim,
        mid_depth=cfg.net_base.mid_depth,
        cfg=cfg,
    )

    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval().to(DEVICE)


    test_results = []


    print(f"Running inference on {len(loader)} samples." )
    print()

    ###  DEMO  LOADER
    for batch in loader:


        inputs, labels_dense, _, conscious_data, info_seg_dict = batch

        # print(info_seg_dict[0]["seg_type"])

        conscious_data = conscious_data.float() 

        labels_dense = labels_dense.float()    
         
 
        outputs = model(conscious_data)




        outputs = outputs.squeeze(dim=1)
         
        probs = torch.sigmoid(outputs.data)  # Convert to probabilities
        predicted_dense = (probs >= 0.5).float()  # Convert to class labels (0 or 1)


        # POST PROCESS SMOOTHING PREDICTION, remove outlier slip  detection  (lonely ones)
        for b in range(predicted_dense.shape[0]):
            for t in range(1, predicted_dense.shape[1] - 1):
                if predicted_dense[b, t] == 1 and predicted_dense[b, t - 1] == 0 and predicted_dense[b, t + 1] == 0:
                    predicted_dense[b, t] = 0 

         


        b = 0


        res_dict = {}
        res_dict['slip_run_id'] = info_seg_dict[b]['slip_run_id'] 


        res_dict['labels_dense'] = labels_dense.cpu().numpy()[b]
        res_dict['predicted_dense'] = predicted_dense.cpu().numpy()[b]


        res_dict["clip_wlech_limits"] = info_seg_dict[b]['clip_wlech_limits'] 


        res_dict['seg_type'] = info_seg_dict[b]['seg_type']
        res_dict['pze_col_name'] = info_seg_dict[b]['pze_col_name']




        res_dict['inputs'] = inputs.cpu().numpy()[b]


        # res_dict['acc_mask'] = None # = info_seg_dict[b]['clear_selection_formaskacc']  
        res_dict['outputs'] = outputs.cpu().numpy()[b]


        test_results.append(res_dict)
 

    print("-- Analysing results.")

    metrics = compute_metrics_from_results(test_results, threshold=0.5)
    print_metrics_report(metrics)

    # print("SEGMENT:", metrics["segment"])
    # print("DENSE:", metrics["dense"])
    # print("Segment confusion:\n", metrics["segment"]["confusion"])
    # print("Dense confusion:\n", metrics["dense"]["confusion"])
        


    # additional signals for visualization (raw pze signals)

    def load_visu_dataset(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    visu_list = load_visu_dataset(DATA_VISU_PATH)




    # lookup  map on visualization dataset
    visu_map_tmp = {d["slip_run_id"]: d for d in visu_list}
    visu_map = {  visu_map_tmp[d]["slip_run_id"]: visu_map_tmp[d] for d in visu_map_tmp.keys()  } 




    cfg = dict(
        visu_maxfreq=1200, # hz
        num_welch_pad = cfg.dataset.base.num_welch_pad,  # if input is  padded,  synchronize  in the  visualization
    )



    for i, res_dict in enumerate(test_results, 1):
        print(f"[{i:02d}/{len(test_results):02d}] Plot: {res_dict['slip_run_id']}")
        
        visu_sample = visu_map[res_dict["slip_run_id"]]

        visu_demo(res_dict, visu_sample, cfg=cfg)







import numpy as np

def compute_metrics_from_results(test_results, threshold=0.5):
    """
    test_results: list of dicts, each dict must contain:
      - 'outputs': array-like (L,) logits
      - 'labels_dense': array-like (L,) 0/1

    Returns a dict with dense-level and segment-level metrics, including
    per-class accuracy and balanced accuracy.
    """

    # --- helpers
    def sigmoid(x):
        x = np.asarray(x, dtype=np.float32)
        return 1.0 / (1.0 + np.exp(-x))

    def safe_mean(x):
        x = np.asarray(x, dtype=np.float32)
        return float(np.mean(x)) if x.size else float("nan")

    # --- accumulate confusion counts
    # Segment-level confusion: rows = true label, cols = pred label
    seg_conf = np.zeros((2, 2), dtype=np.int64)

    # Dense-level confusion: rows = true label, cols = pred label
    dense_conf = np.zeros((2, 2), dtype=np.int64)

    # Also track per-class accuracies directly
    dense_correct_mask_per_class = {0: [], 1: []}
    seg_correct_mask_per_class = {0: [], 1: []}

    for r in test_results:
        logits = np.asarray(r["outputs"]).reshape(-1)          # (L,)
        y = np.asarray(r["labels_dense"]).astype(np.uint8).reshape(-1)  # (L,)

        # Guard: align lengths if something weird happens
        L = min(len(logits), len(y))
        logits = logits[:L]
        y = y[:L]

        p = sigmoid(logits)
        yhat_dense = (p >= threshold).astype(np.uint8)

        # --- segment labels/preds from dense
        y_seg = 1 if np.any(y == 1) else 0
        yhat_seg = 1 if np.any(yhat_dense == 1) else 0

        # --- segment confusion
        seg_conf[y_seg, yhat_seg] += 1
        seg_correct_mask_per_class[y_seg].append(1 if (y_seg == yhat_seg) else 0)

        # --- dense confusion
        # Count per-timestep (TP/TN/FP/FN)
        # rows=true, cols=pred
        # We'll do it by counting occurrences of each pair
        for true_val in (0, 1):
            for pred_val in (0, 1):
                dense_conf[true_val, pred_val] += int(np.sum((y == true_val) & (yhat_dense == pred_val)))

        # per-class dense accuracy: accuracy restricted to timesteps of that class
        for cls in (0, 1):
            mask = (y == cls)
            if np.any(mask):
                dense_correct_mask_per_class[cls].append(float(np.mean(yhat_dense[mask] == y[mask])))

    # --- compute accuracies from confusion matrices
    def metrics_from_conf(conf):
        # conf rows=true [0,1], cols=pred [0,1]
        TN, FP = conf[0, 0], conf[0, 1]
        FN, TP = conf[1, 0], conf[1, 1]
        total = TN + FP + FN + TP

        acc = (TP + TN) / total if total else float("nan")

        # per-class recall == accuracy on that class:
        # class 0: TN / (TN+FP)
        # class 1: TP / (TP+FN)
        acc0 = TN / (TN + FP) if (TN + FP) else float("nan")
        acc1 = TP / (TP + FN) if (TP + FN) else float("nan")
        bal_acc = 0.5 * (acc0 + acc1) if (np.isfinite(acc0) and np.isfinite(acc1)) else float("nan")

        # positive-class precision/recall/F1 (class 1 treated as "positive")
        precision = TP / (TP + FP) if (TP + FP) else float("nan")
        recall = acc1  # TP / (TP + FN)
        f1 = (2 * precision * recall / (precision + recall)) if (np.isfinite(precision) and np.isfinite(recall) and (precision + recall) and (precision + recall) !=0 ) else float("nan")
        # alternatively: f1 = (2*TP / (2*TP + FP + FN)) if (2*TP + FP + FN) else float("nan")

        return {
            "confusion": conf,
            "accuracy": float(acc),
            "acc_class0": float(acc0),
            "acc_class1": float(acc1),
            "balanced_accuracy": float(bal_acc),
            "precision_pos": float(precision),
            "recall_pos": float(recall),
            "f1_score": float(f1),
            "TN": int(TN), "FP": int(FP), "FN": int(FN), "TP": int(TP),
        }

    dense_metrics = metrics_from_conf(dense_conf)
    seg_metrics = metrics_from_conf(seg_conf)

    # mean per-class accuracy computed directly (useful sanity check)
    dense_metrics["mean_acc_class0_segments_avg"] = safe_mean(dense_correct_mask_per_class[0])
    dense_metrics["mean_acc_class1_segments_avg"] = safe_mean(dense_correct_mask_per_class[1])

    seg_metrics["mean_acc_class0"] = safe_mean(seg_correct_mask_per_class[0])
    seg_metrics["mean_acc_class1"] = safe_mean(seg_correct_mask_per_class[1])

    return {
        "threshold": float(threshold),
        "dense": dense_metrics,
        "segment": seg_metrics,
    }

def format_confusion(conf):
    # conf = [[TN FP],[FN TP]]
    TN, FP = int(conf[0,0]), int(conf[0,1])
    FN, TP = int(conf[1,0]), int(conf[1,1])
    return (
        "          pred0   pred1\n"
        f"true0     {TN:5d}  {FP:5d}\n"
        f"true1     {FN:5d}  {TP:5d}"
    )

def print_metrics_report(metrics):
    def line(title, m):
        return (
            f"{title}\n"
            f"  acc={m['accuracy']:.4f}  bal_acc={m['balanced_accuracy']:.4f}  "
            f"prec={m['precision_pos']:.4f}  rec={m['recall_pos']:.4f}  f1={m['f1_score']:.4f}\n"
            f"  confusion:\n{format_confusion(m['confusion'])}\n"
        )

    print(line("SEGMENT-LEVEL METRICS", metrics["segment"]))
    print(line("100Hz DETECTION METRICS", metrics["dense"]))




if __name__ == "__main__":
    main()
