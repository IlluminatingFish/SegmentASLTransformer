import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saved_attn_weights = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False  # 不平均头
        )
        # ✅ 保留梯度，不使用 .detach()
        self.saved_attn_weights = attn_weights  # (B, num_heads, T, T)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerSegmenter(nn.Module):
    def __init__(self, skeleton_dim, label_vocab_size, label_embed_dim, embed_dim, num_heads, num_layers, dropout=0.1, num_classes=2):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.skel_proj = nn.Linear(skeleton_dim, embed_dim)
        self.label_embedding = nn.Embedding(label_vocab_size, label_embed_dim)
        self.label_proj = nn.Linear(label_embed_dim, embed_dim)
        self.preseg_proj = nn.Linear(1, embed_dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, embed_dim))  # [1, T, D]

        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, skel_feats, label_indices, label_probs, preseg):
        B, T, _ = skel_feats.size()

        # 骨架特征编码
        skel_embed = self.skel_proj(skel_feats)

        # 可选：label embedding + prob（暂时未使用）
        # label_embed = self.label_embedding(label_indices)
        # label_embed = self.label_proj(label_embed * label_probs.unsqueeze(-1))
        # skel_embed += label_embed

        # preseg 特征
        preseg = preseg.unsqueeze(-1).float()
        preseg_embed = self.preseg_proj(preseg)

        fused = skel_embed + preseg_embed

        # 添加位置编码
        if T > self.pos_embedding.size(1):
            raise ValueError("Input sequence too long")
        pos_encoded = fused + self.pos_embedding[:, :T, :]

        out = pos_encoded
        attention_scores = None

        for i, layer in enumerate(self.encoder_layers):
            out = layer(out)
            if i == self.num_layers - 1:
                attn = layer.saved_attn_weights  # shape: (B, H, T, T)
                attention_scores = attn.mean(dim=1).mean(dim=1)  # → (B, T)

        logits = self.classifier(out)  # shape: [B, T, num_classes]
        return logits, attention_scores
