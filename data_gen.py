import torch
from torch.utils.data import DataLoader, TensorDataset

# 参数
input_dim = 10000  # 词汇表大小
seq_len = 10  # 序列长度
num_classes = 3  # 类别数量
num_samples = 1000  # 样本数量

# 生成随机输入数据 (整数代表token索引)
train_data = torch.randint(0, input_dim, (num_samples, seq_len))

# 生成随机标签
train_labels = torch.randint(0, num_classes, (num_samples,))

# 创建 DataLoader
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 检查生成的数据
for batch_data, batch_labels in train_loader:
    print("Input Batch Data Shape:", batch_data.shape)
    print("Input Batch Labels Shape:", batch_labels.shape)
    break  # 只输出一个批次以示例
