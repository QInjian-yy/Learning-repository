import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== 手动实现GCN层 =====================
class ManualGCNLayer(nn.Module):
    """
    手动实现的图卷积网络(GCN)层

    这个类实现了GCN的核心操作：在图上进行消息传递和特征变换。
    对应数学公式：H' = σ(Ã H W + b)
    其中：Ã = D^(-1/2) A D^(-1/2) 是归一化的邻接矩阵
    """

    def __init__(self, in_features, out_features, bias=True):
        """
        初始化GCN层

        Args:
            in_features (int): 输入特征的维度
                - 每个节点特征向量的长度
                - 例如：Cora数据集是1433维的词袋向量

            out_features (int): 输出特征的维度
                - 经过GCN层变换后，节点特征的维度
                - 通常被称为"隐藏层大小"
                - 例如：设置为64，表示每个节点输出64维特征

            bias (bool, optional): 是否使用偏置项
                - 如果为True，添加可学习的偏置参数
                - 如果为False，只使用权重变换
                - 默认值：True（通常建议使用偏置）
        """
        # 调用父类nn.Module的初始化方法
        # 这行代码确保nn.Module的正确初始化，注册所有的子模块和参数
        super(ManualGCNLayer, self).__init__()
        # Python 3+ 可以简写为：super().__init__()

        # 存储输入和输出维度，用于后续操作和模型描述
        self.in_features = in_features  # 输入维度，如：1433
        self.out_features = out_features  # 输出维度，如：64

        # ========== 定义可学习的权重参数 ==========
        # nn.Parameter是特殊的tensor，会被自动注册为模型参数，可以被优化器更新
        # 权重矩阵形状：[in_features, out_features]
        # 例如：[1433, 64] 表示将1433维特征映射到64维特征
        # 每个输入维度和输出维度之间都有一个权重连接
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))

        # ========== 定义可学习的偏置参数 ==========
        if bias:
            # 偏置向量形状：[out_features]
            # 每个输出维度有一个偏置值
            # 例如：[64] 表示64个输出维度各有一个偏置
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            # 如果不使用偏置，注册一个None参数
            # register_parameter是PyTorch的API，用于显式声明参数名称为'bias'但值为None
            self.register_parameter('bias', None)

        # 调用参数初始化方法
        # 这是关键步骤：未初始化的参数会导致训练不稳定
        self.reset_parameters()

    def reset_parameters(self):
        """
        初始化模型参数

        使用Xavier初始化（也称为Glorot初始化）方法：
        这种初始化策略可以保持前向传播和反向传播时信号的方差稳定，
        避免梯度消失或爆炸问题，特别适用于有激活函数的神经网络层。
        """
        # Xavier均匀分布初始化权重
        # 公式：参数从均匀分布U[-a, a]中采样，其中 a = sqrt(6 / (fan_in + fan_out))
        # fan_in = in_features, fan_out = out_features
        # 这样可以保证输入和输出的方差大致相同
        nn.init.xavier_uniform_(self.weight)

        # 初始化偏置为0
        # 偏置通常初始化为0，这样模型从一个对称的起点开始学习
        if self.bias is not None:
            # zeros_: 将偏置向量所有元素设为0
            # 也可以考虑：nn.init.constant_(self.bias, 0.1) 用小常数初始化
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        前向传播过程：执行图卷积操作

        Args:
            x (torch.Tensor): 节点特征矩阵
                形状：[N, in_features]
                例如：[2708, 1433] 表示2708个节点，每个节点1433维特征

            adj (torch.Tensor): 归一化的邻接矩阵
                形状：[N, N]
                要求：adj = D^(-1/2) A D^(-1/2)  （对称归一化的拉普拉斯矩阵）
                其中：A是邻接矩阵，D是度矩阵
                注：adj应该包含自环（节点与自己连接）

        Returns:
            torch.Tensor: 更新后的节点特征矩阵
                形状：[N, out_features]
                例如：[2708, 64] 表示2708个节点，每个节点64维新特征
        """
        # ========== 步骤1：线性特征变换 ==========
        # 公式：support = x @ weight
        # torch.matmul(x, self.weight) 等价于 x @ self.weight
        # 数学意义：对每个节点的特征进行线性变换
        # 从 [N, in_features] 变换到 [N, out_features]
        support = torch.matmul(x, self.weight)
        # support的形状：[N, out_features]

        # ========== 步骤2：邻居信息聚合 ==========
        # 这是GCN的核心：聚合邻居节点的特征
        # 根据邻接矩阵adj，将每个节点邻居的特征聚合起来
        if adj.is_sparse:
            # 如果邻接矩阵是稀疏格式，使用稀疏矩阵乘法
            # 稀疏矩阵乘法更高效，因为邻接矩阵通常是稀疏的（大部分元素为0）
            # torch.sparse.mm: 稀疏矩阵与稠密矩阵的乘法
            output = torch.sparse.mm(adj, support)
        else:
            # 如果邻接矩阵是稠密格式，使用普通矩阵乘法
            # torch.matmul(adj, support) 等价于 adj @ support
            output = torch.matmul(adj, support)
        # 此时output的形状依然是：[N, out_features]
        # 但每个节点的特征已经包含了邻居信息

        # ========== 步骤3：添加偏置（可选） ==========
        if self.bias is not None:
            # 添加偏置向量到输出
            # 这里利用了PyTorch的广播机制：
            # output形状: [N, out_features]
            # bias形状:   [out_features]
            # 结果是每个节点的同一维度都加上相同的偏置值
            output = output + self.bias
            # 数学等价于：output[i, j] = output[i, j] + bias[j]

        # 返回更新后的节点特征
        # 注意：这里没有应用激活函数，通常在GCN模型的外部应用ReLU等激活函数
        return output


def main():
    """GCN层测试"""

    # 1. 创建测试数据
    N = 4  # 4个节点
    in_features = 3  # 输入特征维度
    out_features = 2  # 输出特征维度

    # 节点特征
    x = torch.tensor([
        [1.0, 0.0, 1.0],  # 节点0
        [0.0, 1.0, 1.0],  # 节点1
        [1.0, 1.0, 0.0],  # 节点2
        [0.0, 0.0, 1.0],  # 节点3
    ], dtype=torch.float32)

    print(f"输入特征 x ({x.shape[0]}个节点, {x.shape[1]}维特征):")
    print(x)

    # 原始邻接矩阵（没有自环）
    adj = torch.tensor([
        [0, 1, 0, 1],  # 节点0连接到1和3
        [1, 0, 1, 0],  # 节点1连接到0和2
        [0, 1, 0, 1],  # 节点2连接到1和3
        [1, 0, 1, 0],  # 节点3连接到0和2
    ], dtype=torch.float32)

    print(f"邻接矩阵 adj ({adj.shape[0]}×{adj.shape[1]}):")
    print(adj)

    # 3. 归一化邻接矩阵（简化版，只添加自环）
    adj_with_selfloop = adj + torch.eye(N)  # 添加自环

    # 计算度矩阵
    degree = adj_with_selfloop.sum(dim=1)

    # 计算 D^(-1/2)
    """
    torch.diag(degree ** -0.5)
    
    degree ** -0.5 - 对度数向量中的每个元素取 - 0.5次方
    torch.diag() - 将向量转换为对角矩阵
    """
    degree_inv_sqrt = torch.diag(degree ** -0.5)

    # 归一化: D^(-1/2) A D^(-1/2)
    adj_normalized = degree_inv_sqrt @ adj_with_selfloop @ degree_inv_sqrt

    print(f"归一化邻接矩阵:")
    print(adj_normalized)

    # 4. 创建GCN层
    gcn_layer = ManualGCNLayer(in_features, out_features, bias=True)

    print(f" 创建的GCN层: {gcn_layer}")
    print(f"权重形状: {gcn_layer.weight.shape}")
    print(f"偏置形状: {gcn_layer.bias.shape}")

    # 5. 查看初始化后的参数
    print("初始化后的参数:")
    print("权重矩阵 W:")
    print(gcn_layer.weight.data)

    print("偏置向量 b:")
    print(gcn_layer.bias.data)


    # 6. 执行前向传播
    output = gcn_layer(x, adj_normalized)

    print(f"前向传播结果:")
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    print("输出值:")
    print(output)

    # 7. 验证计算公式
    print("\n 公式验证: output = adj_normalized @ (x @ weight) + bias")

    # 手动计算
    manual_output = adj_normalized @ (x @ gcn_layer.weight) + gcn_layer.bias

    diff = torch.abs(output - manual_output).max()
    print(f"最大差异: {diff.item():.10f}")

    if diff < 1e-6:
        print("验证通过")
    else:
        print("验证失败")


# 运行测试
if __name__ == "__main__":
    main()