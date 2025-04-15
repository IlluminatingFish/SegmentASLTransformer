import torch
import os
import re
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset
from model import TransformerSegmenter
import ast

# 数据集定义
class SegmentDataset(Dataset):
    def __init__(self, skel, label_idx, label_probs, preseg, targets):
        self.skel = skel
        self.label_idx = label_idx
        self.label_probs = label_probs
        self.preseg = preseg
        self.targets = targets

    def __len__(self):
        return len(self.skel)

    def __getitem__(self, idx):
        return self.skel[idx], self.label_idx[idx], self.label_probs[idx], self.preseg[idx], self.targets[idx]

# 提取边界段落 (1 -> 2) 和 (2 -> 1)
def extract_segments_transition(seq):
    segments = []
    start = None
    for i in range(1, len(seq)):
        if seq[i - 1] == 1 and seq[i] == 2:
            start = i
        elif seq[i - 1] == 2 and seq[i] == 1:
            if start is not None:
                segments.append((start, i - 1))
                start = None
    if start is not None and seq[-1] == 2:
        segments.append((start, len(seq) - 1))
    return segments

# 计算容忍匹配的 F1
def boundary_f1_score(pred, gt, tolerance=3):
    matched = 0
    used = set()
    for ps, pe in pred:
        for i, (gs, ge) in enumerate(gt):
            if i in used:
                continue
            if abs(ps - gs) <= tolerance and abs(pe - ge) <= tolerance:
                matched += 1
                used.add(i)
                break
    precision = matched / len(pred) if pred else 0
    recall = matched / len(gt) if gt else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def evaluate(model, dataloader, device):
    model.eval()
    all_f1 = []
    with torch.no_grad():
        for skel, label_idx, label_probs, preseg, targets in dataloader:
            skel, label_idx, label_probs, preseg, targets = [x.to(device) for x in (skel, label_idx, label_probs, preseg, targets)]
            logits = model(skel, label_idx, label_probs, preseg)
            preds = torch.argmax(logits, dim=-1).cpu()
            targets = targets.cpu()
            for p_seq, t_seq in zip(preds, targets):
                p_seg = extract_segments_transition(p_seq.tolist())
                t_seg = extract_segments_transition(t_seq.tolist())
                f1 = boundary_f1_score(p_seg, t_seg)
                all_f1.append(f1)
    model.train()
    return sum(all_f1) / len(all_f1) if all_f1 else 0

def get_best_model_path(save_dir):
    models = []
    for f in os.listdir(save_dir):
        match = re.search(r'f1_(\d\.\d+).pth', f)
        if match:
            f1 = float(match.group(1))
            models.append((f1, f))
    if not models:
        raise FileNotFoundError("No saved models found in directory.")
    best_model = sorted(models, reverse=True)[0][1]
    return os.path.join(save_dir, best_model)

def main():
    save_dir = "./saved_models"
    skeleton_dim = 3
    label_vocab_size = 20
    label_embed_dim = 32
    embed_dim = 128
    num_heads = 4
    num_layers = 2
    num_classes = 3
    dropout = 0.1
    batch_size = 16

    joint_data = np.load("data/val_combined_data_joint.npy", allow_pickle=True)
    joint_data = np.squeeze(joint_data, axis=-1)
    joint_data = np.transpose(joint_data, (0, 2, 3, 1))

    # 添加 velocity（仅对 x, y 计算）
    velocity_features = []
    for seq in joint_data:
        prev_diff = np.zeros_like(seq)
        next_diff = np.zeros_like(seq)
        prev_diff[1:, :, :2] = seq[1:, :, :2] - seq[:-1, :, :2]
        next_diff[:-1, :, :2] = seq[:-1, :, :2] - seq[1:, :, :2]
        velocity = np.concatenate([seq, prev_diff[:, :, :2], next_diff[:, :, :2]], axis=-1)
        velocity_features.append(velocity)
    velocity_features = np.stack(velocity_features, axis=0)
    skel = torch.tensor(velocity_features.reshape(velocity_features.shape[0], velocity_features.shape[1], -1), dtype=torch.float32)

    with open("data/val_800_combined_label.pkl", "rb") as f:
        data = pickle.load(f)
    stacked_tensor = torch.tensor(data[1], dtype=torch.long)
    stacked_tensor[(stacked_tensor == 3) | (stacked_tensor == 4)] = 2
    targets = stacked_tensor

    label_idx = torch.randint(0, label_vocab_size, targets.shape)
    label_probs = torch.rand(targets.shape)
    txt_path = "data/Bloss_predicted_indices.txt"
    with open(txt_path, "r") as f:
        text = f.read()
    wrapped_text = "[" + text.strip().rstrip(",") + "]"
    preseg_list = ast.literal_eval(wrapped_text)
    preseg = torch.tensor(preseg_list, dtype=torch.long)

    # 评估 preseg 的自身效果
    print("Evaluating preseg-only performance...")
    dummy_preds = preseg
    preseg_f1 = []
    for p_seq, t_seq in zip(dummy_preds, targets):
        p_seg = extract_segments_transition(p_seq.tolist())
        t_seg = extract_segments_transition(t_seq.tolist())
        f1 = boundary_f1_score(p_seg, t_seg)
        preseg_f1.append(f1)
    print(f"Preseg-only Boundary F1: {sum(preseg_f1) / len(preseg_f1):.4f}\n")

    test_dataset = SegmentDataset(skel, label_idx, label_probs, preseg, targets)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = TransformerSegmenter(27 * 7, label_vocab_size, label_embed_dim, embed_dim, num_heads, num_layers, dropout, num_classes=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_model_path = get_best_model_path(save_dir)
    model.load_state_dict(torch.load(best_model_path))
    print(f"Loaded best model from {best_model_path}")

    f1 = evaluate(model, test_loader, device)
    print(f"Boundary-level mean F1 score on test set: {f1:.4f}")

if __name__ == "__main__":
    main()
