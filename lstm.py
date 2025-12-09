import torch
from torch import nn


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        初始化LSTM
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏状态维度
        """
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size

        # 输入门参数
        self.W_xi = nn.Linear(input_size, hidden_size)
        self.W_hi = nn.Linear(hidden_size, hidden_size, bias=False)

        # 遗忘门参数
        self.W_xf = nn.Linear(input_size, hidden_size)
        self.W_hf = nn.Linear(hidden_size, hidden_size, bias=False)

        # 输出门参数
        self.W_xo = nn.Linear(input_size, hidden_size)
        self.W_ho = nn.Linear(hidden_size, hidden_size, bias=False)

        # 候选记忆参数
        self.W_xc = nn.Linear(input_size, hidden_size)
        self.W_hc = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, state=None):
        """
        前向传播
        Args:
            x: 输入序列 [batch_size, input_size]
            state: 初始状态元组 (h_0, c_0)，可选
        Returns:
            h_t: 隐藏状态
            c_t: 记忆细胞
            (h_t, c_t): 作为元组
        """
        batch_size = x.size(0)

        # 初始化隐藏状态和记忆细胞
        if state is None:
            h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        else:
            h_t, c_t = state

        # 1. 计算输入门 (决定要更新哪些信息)
        i_t = torch.sigmoid(self.W_xi(x) + self.W_hi(h_t))

        # 2. 计算遗忘门 (决定要遗忘哪些信息)
        f_t = torch.sigmoid(self.W_xf(x) + self.W_hf(h_t))

        # 3. 计算输出门 (决定要输出哪些信息)
        o_t = torch.sigmoid(self.W_xo(x) + self.W_ho(h_t))

        # 4. 计算候选记忆细胞 (新的信息)
        c_tilde = torch.tanh(self.W_xc(x) + self.W_hc(h_t))

        # 5. 更新记忆细胞 (核心步骤!)
        c_t = f_t * c_t + i_t * c_tilde

        # 6. 计算新的隐藏状态
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


def test_single_timestep():
    """测试1：单个时间步（就像你原来的代码）"""
    print("=" * 50)
    print("测试1：单个时间步")
    print("=" * 50)

    # 创建一个SimpleLSTM实例
    lstm_cell = SimpleLSTM(input_size=10, hidden_size=20)

    # 创建一个输入（假设batch_size=3，每个样本有10个特征）
    x = torch.randn(3, 10)  # [batch_size, input_size]

    # 前向传播
    h_t, c_t = lstm_cell(x)

    print(f"输入形状: {x.shape}")
    print(f"隐藏状态h_t形状: {h_t.shape}")  # [3, 20]
    print(f"记忆细胞c_t形状: {c_t.shape}")  # [3, 20]
    print(f"这是单个时间步的计算，没有考虑时间序列")
    return lstm_cell


def test_sequence_processing():
    """测试2：手动循环处理序列"""
    print("\n" + "=" * 50)
    print("测试2：手动循环处理序列")
    print("=" * 50)

    # 使用同一个LSTM实例（或者创建新的）
    lstm_cell = SimpleLSTM(input_size=10, hidden_size=20)

    # 创建一个序列输入 [seq_len, batch_size, input_size]
    seq_len = 5  # 序列长度5
    batch_size = 3
    x_sequence = torch.randn(seq_len, batch_size, 10)

    print(f"序列输入形状: {x_sequence.shape}")  # [5, 3, 10]
    print(f"这表示有5个时间步，每个时间步有3个样本，每个样本10个特征")

    # 初始化状态
    h_t = torch.zeros(batch_size, 20)
    c_t = torch.zeros(batch_size, 20)

    # 存储所有时间步的隐藏状态
    all_hidden_states = []
    all_cell_states = []

    # 手动循环处理每个时间步
    print("\n开始处理每个时间步：")
    for t in range(seq_len):
        # 获取当前时间步的输入
        x_t = x_sequence[t]  # [batch_size, input_size]

        # LSTM Cell 前向传播
        h_t, c_t = lstm_cell(x_t, (h_t, c_t))

        # 保存状态
        all_hidden_states.append(h_t)
        all_cell_states.append(c_t)

        print(f"时间步 {t}: 输入={x_t.shape}, 隐藏状态={h_t.shape}, 记忆细胞={c_t.shape}")

    # 堆叠所有时间步的输出
    all_hidden_states = torch.stack(all_hidden_states, dim=0)  # [seq_len, batch_size, hidden_size]
    all_cell_states = torch.stack(all_cell_states, dim=0)  # [seq_len, batch_size, hidden_size]

    print(f"\n汇总结果：")
    print(f"所有隐藏状态形状: {all_hidden_states.shape}")
    print(f"所有记忆细胞形状: {all_cell_states.shape}")
    print(f"最后一个时间步的隐藏状态: {all_hidden_states[-1].shape}")
    print(f"最后一个时间步的记忆细胞: {all_cell_states[-1].shape}")


if __name__ == "__main__":
    # 测试1：单个时间步
    lstm_cell = test_single_timestep()

    # 测试2：序列处理
    test_sequence_processing()