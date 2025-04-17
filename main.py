import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from Transformer.model.model_copy import TransformerClassifier
import os

# 设置参数
input_dim = 10000  # 词汇表大小
embed_dim = 128  # 嵌入维度
num_heads = 4  # 注意力头数
num_layers = 2  # Transformer 层数
num_classes = 3  # 类别数量
dropout = 0.1  # Dropout 概率
batch_size = 32  # 批量大小
seq_len = 10  # 序列长度
num_epochs = 5  # 训练轮数
learning_rate = 0.001  # 学习率
num_train_samples = 1000  # 训练样本数量
num_test_samples = 200  # 测试样本数量

# Step 1: 数据生成
def generate_data(num_samples, input_dim, seq_len, num_classes):
    data = torch.randint(0, input_dim, (num_samples, seq_len))
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Step 2: 训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            # 前向传播
            outputs = model(batch_data)

            # 计算损失
            loss = criterion(outputs, batch_labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

    # 保存模型
    torch.save(model.state_dict(), 'transformer_model.pth')
    print("Training complete. Model saved as 'transformer_model.pth'.")

# Step 3: 测试函数
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            # 前向传播
            outputs = model(batch_data)
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 生成数据
    train_loader = generate_data(num_train_samples, input_dim, seq_len, num_classes)
    test_loader = generate_data(num_test_samples, input_dim, seq_len, num_classes)

    # 创建模型
    model = TransformerClassifier(input_dim, embed_dim, num_heads, num_layers, num_classes, dropout).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    # 加载训练后的模型进行测试
    model.load_state_dict(torch.load('transformer_model.pth'))
    test_model(model, test_loader)
