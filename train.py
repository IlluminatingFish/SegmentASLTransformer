
import torch
import os
import re
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset, random_split
from model import TransformerSegmenter
import ast
import torch.nn as nn
import torch.optim as optim

def keep_top_k_models(save_dir, k=5):
    models = []
    for f in os.listdir(save_dir):
        match = re.search(r'f1_(\d\.\d+).pth', f)
        if match:
            f1 = float(match.group(1))
            models.append((f1, f))
    models.sort(reverse=True)
    return models[:k]

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

def boundary_f1_score(pred, gt, tolerance=5):
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

def alignment_loss_fn(attn_maps, targets, tolerance=3):
    losses = []
    for attn, tgt in zip(attn_maps, targets):
        gt_segments = extract_segments_transition(tgt.tolist())
        if not gt_segments:
            continue
        target_attn = torch.zeros_like(attn)
        for s, e in gt_segments:
            s = max(0, s - tolerance)
            e = min(len(attn) - 1, e + tolerance)
            target_attn[s:e+1] = 1.0
        target_attn = target_attn / target_attn.sum()
        pred_attn = attn / attn.sum()
        loss = nn.functional.kl_div(pred_attn.log(), target_attn, reduction='batchmean')
        losses.append(loss)
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=attn_maps.device, requires_grad=True)

def evaluate(model, dataloader, device):
    model.eval()
    all_f1 = []
    with torch.no_grad():
        for skel, label_idx, label_probs, preseg, targets in dataloader:
            skel, label_idx, label_probs, preseg, targets = [x.to(device) for x in (skel, label_idx, label_probs, preseg, targets)]
            logits, _ = model(skel, label_idx, label_probs, preseg)
            preds = torch.argmax(logits, dim=-1).cpu()
            targets = targets.cpu()
            for p_seq, t_seq in zip(preds, targets):
                p_seg = extract_segments_transition(p_seq.tolist())
                t_seg = extract_segments_transition(t_seq.tolist())
                f1 = boundary_f1_score(p_seg, t_seg)
                all_f1.append(f1)
    model.train()
    return sum(all_f1) / len(all_f1) if all_f1 else 0

def soft_boundary_f1_loss(attn_maps, targets, tolerance=5, eps=1e-8):
    losses = []
    for attn, tgt in zip(attn_maps, targets):  # attn: (T,), tgt: (T,)
        gt_segments = extract_segments_transition(tgt.tolist())
        if not gt_segments:
            continue

        T = attn.shape[0]
        pred_boundary_score = attn / (attn.sum() + eps)  # soft scores → normalized to 1

        # 构造 ground truth 边界 mask
        gt_boundary_mask = torch.zeros_like(attn)
        for s, e in gt_segments:
            gt_boundary_mask[max(0, s - tolerance)] = 1.0
            gt_boundary_mask[min(T - 1, e + tolerance)] = 1.0
        gt_boundary_mask = gt_boundary_mask / (gt_boundary_mask.sum() + eps)

        # soft precision / recall / F1
        tp = (pred_boundary_score * gt_boundary_mask).sum()
        precision = tp / (pred_boundary_score.sum() + eps)
        recall = tp / (gt_boundary_mask.sum() + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        losses.append(1 - f1)  # minimize 1 - F1

    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=attn_maps.device, requires_grad=True)

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
    os.makedirs(save_dir, exist_ok=True)

    # === 数据加载部分保持不变 ===
    joint_data = np.load("data/train_combined_data_joint.npy", allow_pickle=True)
    joint_data = np.squeeze(joint_data, axis=-1)
    joint_data = np.transpose(joint_data, (0, 2, 3, 1))

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

    with open("data/train_800_combined_label.pkl", "rb") as f:
        data = pickle.load(f)
    stacked_tensor = torch.tensor(data[1], dtype=torch.long)
    stacked_tensor[(stacked_tensor == 3) | (stacked_tensor == 4)] = 2
    targets = stacked_tensor

    label_vocab_size = 20
    label_idx = torch.randint(0, label_vocab_size, targets.shape)
    label_probs = torch.rand(targets.shape)
    txt_path = "data/Train_Bloss_predicted_indices.txt"
    with open(txt_path, "r") as f:
        text = f.read()
    preseg_list = ast.literal_eval("[" + text.strip().rstrip(",") + "]")
    preseg = torch.tensor(preseg_list, dtype=torch.long)

    dataset = SegmentDataset(skel, label_idx, label_probs, preseg, targets)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(614)  # 你可以改成任何数字
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # === 模型初始化 ===
    model = TransformerSegmenter(27 * 7, label_vocab_size, 32, 128, 4, 2, dropout=0.1, num_classes=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # === Resume training if existing model found ===
    best_f1 = 0.0
    try:
        best_model_path = get_best_model_path(save_dir)
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        best_f1 = float(re.search(r'f1_(\d\.\d+).pth', best_model_path).group(1))
        print(f"✅ Resuming training from best model: {best_model_path} (F1 = {best_f1:.4f})")
    except Exception as e:
        print(f"🔁 No previous model found, starting fresh. ({str(e)})")

    # === Training Loop ===
    for epoch in range(100):
        total_loss = 0
        for batch in train_loader:
            skel, label_idx, label_probs, preseg, targets = [b.to(device) for b in batch]
            logits, attn_weights = model(skel, label_idx, label_probs, preseg)
            B, T, C = logits.shape
            ce_loss = criterion(logits.view(B * T, C), targets.view(B * T))
            align_loss = soft_boundary_f1_loss(attn_weights, targets, tolerance=5)
            loss =  ce_loss + align_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        mean_f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}, Mean F1: {mean_f1:.4f}")

        if mean_f1 > best_f1:
            best_f1 = mean_f1
            model_filename = f"transformer_model_f1_{mean_f1:.4f}.pth"
            model_path = os.path.join(save_dir, model_filename)

            top_models = keep_top_k_models(save_dir, k=5)
            if len(top_models) >= 5:
                lowest_f1, lowest_file = top_models[-1]
                if mean_f1 > lowest_f1:
                    os.remove(os.path.join(save_dir, lowest_file))
                else:
                    continue

            torch.save(model.state_dict(), model_path)
            print(f"📦 Saved better model to: {model_path}")

    print("✅ Training complete.")

if __name__ == "__main__":
    main()
