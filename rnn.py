import torch
import torch.nn as nn


# 定义RNN模型类
class RNNModel(nn.Module):
    # __init__: 初始化函数，定义模型的各个组件
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        参数说明：
        input_size: 输入特征维度，比如每个单词的词向量维度
        hidden_size: 隐藏层维度，RNN记忆的容量大小
        num_layers: RNN层数，堆叠多少层RNN
        output_size: 输出维度，比如分类问题的类别数
        dropout: 丢弃率，防止过拟合，0表示不使用dropout
        """

        # super().__init__(): 调用父类nn.Module的初始化方法
        # 确保RNNModel类继承了nn.Module的所有属性和方法
        super(RNNModel, self).__init__()

        # 创建RNN层
        self.rnn = nn.RNN(
            input_size=input_size,  # 输入数据的特征维度
            hidden_size=hidden_size,  # 隐藏状态的维度（记忆大小）
            num_layers=num_layers,  # RNN的层数
            batch_first=True,  # 输入数据的格式：(batch_size, seq_len, input_size)
            # 如果为False，格式为：(seq_len, batch_size, input_size)
            dropout=dropout if num_layers > 1 else 0  # dropout只在多层RNN中使用
            # dropout解释：随机丢弃一部分神经元，防止过拟合
            # 如果num_layers>1，使用dropout；如果只有1层，dropout=0
        )

        # 创建全连接层（线性层），将RNN的输出映射到最终的输出维度
        # Linear层计算公式：y = x * W^T + b
        # 这里将hidden_size维映射到output_size维
        self.fc = nn.Linear(hidden_size, output_size)

        # 创建Dropout层，在训练时随机将一些神经元的输出设为0
        # 防止神经网络过度依赖某些特定神经元，提高泛化能力
        self.dropout = nn.Dropout(dropout)

    # forward: 前向传播函数，定义数据如何通过模型
    def forward(self, x):
        """
        x: 输入数据，形状为(batch_size, seq_len, input_size)
        返回值: 模型输出，形状为(batch_size, output_size)
        """
        # x的形状解释：
        # batch_size: 批次大小，一次处理的样本数量
        # seq_len: 序列长度，时间步的数量
        # input_size: 每个时间步的特征维度

        # RNN前向传播
        # 输入x，返回两个值：
        # out: 所有时间步的隐藏状态，形状为(batch_size, seq_len, hidden_size)
        # hidden: 最后一个时间步的隐藏状态，形状为(num_layers, batch_size, hidden_size)
        out, hidden = self.rnn(x)

        # 取最后一个时间步的输出
        # out[:, -1, :]解释：
        # - [:, ...] 表示取所有批次
        # - -1 表示取最后一个时间步（序列的最后一个元素）
        # - : 表示取所有隐藏维度
        # 结果形状: (batch_size, hidden_size)
        # 为什么取最后一个时间步？因为最后一个时间步包含了整个序列的信息
        out = out[:, -1, :]  # 从(batch, seq, hidden)变为(batch, hidden)

        # 应用dropout
        # 随机丢弃一部分神经元，防止过拟合
        # 注意：Dropout只在训练模式下生效，在测试模式下自动关闭
        out = self.dropout(out)

        # 通过全连接层得到最终输出
        # 从(batch_size, hidden_size)变为(batch_size, output_size)
        out = self.fc(out)

        return out


# 创建RNN模型实例
# 示例：序列分类任务（比如判断电影评论是正面还是负面）
model = RNNModel(
    input_size=10,  # 每个时间步有10个特征，比如每个单词用10维向量表示
    hidden_size=64,  # 隐藏层64维，表示RNN用64个数字来"记住"信息
    num_layers=2,  # 使用2层RNN，意味着有2个堆叠的RNN层
    output_size=2  # 输出2维，表示二分类（比如正面/负面）
)

# 打印模型结构，看看内部参数
print("=" * 50)
print("模型结构:")
print(model)
print("=" * 50)

x = torch.randn(4, 5, 10)  # [batch, seq_len, input_size] 5个时间步，每步10个特征
output = model(x)