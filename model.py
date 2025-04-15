import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerSegmenter(nn.Module):
    def __init__(self, skeleton_dim, label_vocab_size, label_embed_dim, embed_dim, num_heads, num_layers, dropout=0.1, num_classes=2):
        super(TransformerSegmenter, self).__init__()

        # 投影骨架特征
        self.skel_proj = nn.Linear(skeleton_dim, embed_dim)

        # 标签嵌入（可学习）
        self.label_embedding = nn.Embedding(label_vocab_size, label_embed_dim)
        self.label_proj = nn.Linear(label_embed_dim, embed_dim)

        # 引入预分割结果的通道（1维）
        self.preseg_proj = nn.Linear(1, embed_dim)

        # 可学习的位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, embed_dim))  # 假设最多512帧

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层：逐帧2分类（是否为段内）
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, skel_feats, label_indices, label_probs, preseg):
        # skel_feats: [B, T, skeleton_dim]
        # label_indices: [B, T] (int64)
        # label_probs: [B, T] (float, 与索引一一对应)
        # preseg: [B, T] (float or int), 预分割结果标签（0/1）

        B, T, _ = skel_feats.size()

        # 1. 处理骨架特征
        skel_embed = self.skel_proj(skel_feats)  # [B, T, embed_dim]

        # 2. 处理标签嵌入并乘以概率
        # label_embeds = self.label_embedding(label_indices)  # [B, T, label_embed_dim]
        # label_embeds = label_embeds * label_probs.unsqueeze(-1)  # 广播概率乘权重
        # label_embed_proj = self.label_proj(label_embeds)  # [B, T, embed_dim]

        # 3. 处理预分割嵌入
        preseg = preseg.unsqueeze(-1).float()  # [B, T, 1]
        preseg_embed = self.preseg_proj(preseg)  # [B, T, embed_dim]

        # 4. 相加融合
        fused = skel_embed + preseg_embed  # [B, T, embed_dim]

        # 5. 加上位置编码
        if T > self.pos_embedding.size(1):
            raise ValueError("Input sequence length exceeds maximum position embedding length.")
        pos_encoded = fused + self.pos_embedding[:, :T, :]  # [B, T, embed_dim]

        # 6. Transformer 编码
        encoded = self.encoder(pos_encoded)  # [B, T, embed_dim]

        # 7. Frame-wise 分类
        logits = self.classifier(encoded)  # [B, T, num_classes]
        return logits
