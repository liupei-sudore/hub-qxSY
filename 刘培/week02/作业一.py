#conding:utf-8
# ce_loss = nn.CrossEntropyLoss()#交叉熵损失函数
# x=torch.randn(5, 3, requires_grad=True)#随机生成一个5行3列的向量
# target=torch.empty(5, dtype=torch.long).random_(3)#随机生成一个5行1列的向量，元素值在0-2之间
# loss=ce_loss(x, target)
# print(loss)
import os 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
'''
完成一个多分类任务的训练，一个随机向量，哪一维数字最大就属于第几类

'''
#定义模型
class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        x = self.fc(x)
        return x

#损失函数
loss_func = nn.CrossEntropyLoss()#交叉熵损失函数

#生成样本
def generate_data(num_samples=1000, num_features=10, num_classes=10):
    """
        生成随机特征向量，标签为最大值的索引。
    """ 
    X = np.random.randn(num_samples, num_features)
    y = np.argmax(X[:, :num_classes], axis=1)
    return X, y

# 参数设置
num_samples = 2000#样本数量
num_features = 10#特征数量
num_classes = 10#类别数量   
batch_size = 64#批量大小
epochs = 20#训练轮数
lr = 0.01#学习率

#生成训练集和测试集
X_train, y_train = generate_data(num_samples, num_features, num_classes)
X_test, y_test = generate_data(500, num_features, num_classes)

# 封装为 DataLoader
train_dataset = TensorDataset(
    torch.from_numpy(X_train).float(),
    torch.from_numpy(y_train).long(),
)
test_dataset = TensorDataset(
    torch.from_numpy(X_test).float(),
    torch.from_numpy(y_test).long(),
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#训练
model = Net(num_features, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("开始训练...")
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # 前向传播
        outputs = model(batch_x)
        loss = loss_func(outputs, batch_y)

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    
    #测试
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    accuracy = correct / total
    print(f"Epoch {epoch:2d} | 平均损失: {avg_loss:.4f} | 测试准确率: {accuracy:.4f}")
print("训练完成！")
