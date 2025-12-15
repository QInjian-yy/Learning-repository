import torch
from torch import nn
from transformer.TransformerBlock import TransformerBlock


class Encoder(nn.Module):
    """
    Transformer编码器

    功能：将输入符号序列（如单词）转换为富含上下文信息的向量表示
    包含：词嵌入 + 位置编码 + 多层Transformer块
    """

    def __init__(
            self,
            src_vocab_size,  # 源语言词汇表大小（如英文50000个单词）
            embed_size,  # 词向量维度（如512）
            num_layers,  # Transformer块层数（如6层）
            heads,  # 注意力头数（如8）
            device,  # 计算设备（CPU/GPU）
            forward_expansion,  # 前馈网络扩展倍数（通常为4）
            dropout,  # Dropout概率（如0.1）
            max_length,  # 最大序列长度（如100）
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device  # 用于指定张量存放位置

        # ============== 核心组件1：词嵌入层 ==============
        # 作用：将离散的单词ID转换为连续的词向量
        # 例如：单词"apple"的ID=123 → 512维向量[0.1, -0.2, 0.3, ...]
        # 形状：nn.Embedding(词汇表大小, 词向量维度)
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)

        # ============== 核心组件2：位置编码层 ==============
        # 作用：为每个位置生成一个独特的向量，表示单词在序列中的位置
        # RNN/CNN天然知道位置信息，但Transformer没有，所以需要显式添加
        # 形状：nn.Embedding(最大序列长度, 词向量维度)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        # ============== 核心组件3：多层Transformer块 ==============
        # 堆叠多个TransformerBlock来构建深层编码器
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_size,
                heads,
                dropout=dropout,
                forward_expansion=forward_expansion,
            )
            for _ in range(num_layers)  # 创建num_layers个相同的块
        ])

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        前向传播

        输入：
        - x: 输入序列的单词ID，形状为 (batch_size, seq_length)
          例如：[[12, 45, 23, 0], [7, 89, 34, 1]]  batch_size=2, seq_length=4
        - mask: 注意力掩码，用于屏蔽填充位置，形状为 (batch_size, 1, seq_length, seq_length)

        输出：
        - out: 编码后的序列表示，形状为 (batch_size, seq_length, embed_size)
        """
        N, seq_length = x.shape  # N=batch_size, seq_length=序列长度

        # ============== 步骤1：创建位置索引 ==============
        # torch.arange(0, seq_length): 生成[0, 1, 2, ..., seq_length-1]
        # expand(N, seq_length): 扩展到批次维度 [[0,1,2,...], [0,1,2,...], ...]
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        # positions形状: (batch_size, seq_length)
        # 示例：如果batch_size=2, seq_length=4
        # positions = [[0, 1, 2, 3],
        #              [0, 1, 2, 3]]

        # ============== 步骤2：词嵌入 + 位置编码 ==============
        # 词嵌入：将单词ID转换为词向量
        word_embeddings = self.word_embedding(x)  # 形状: (batch_size, seq_length, embed_size)

        # 位置编码：为每个位置生成位置向量
        position_embeddings = self.position_embedding(positions)  # 形状: (batch_size, seq_length, embed_size)

        # 将词向量和位置向量相加
        combined_embeddings = word_embeddings + position_embeddings

        # 应用Dropout
        out = self.dropout(combined_embeddings)
        # 现在out的形状: (batch_size, seq_length, embed_size)

        # ============== 步骤3：通过多层Transformer块 ==============
        # 编码器自注意力：Q=K=V=同一个张量
        for layer in self.layers:
            out = layer(out, out, out, mask)
            # 每层输入输出形状相同: (batch_size, seq_length, embed_size)

        return out


if __name__ == '__main__':
    # 检查GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 模型移动到GPU
    encoder = Encoder(
        src_vocab_size=50000,
        embed_size=512,
        num_layers=6,
        heads=8,
        device=device,  # 使用检测到的设备
        forward_expansion=4,
        dropout=0.1,
        max_length=100
    )
    encoder = encoder.to(device)  # 确保模型在指定设备上

    # 2. 数据也移动到GPU
    input_ids = torch.tensor([
        [23, 456, 12, 0, 0],
        [78, 1234, 56, 89, 7]
    ]).to(device)  # 明确移动到设备

    # 3. 掩码也移动到GPU
    mask = torch.tensor([
    [[  # 样本1
        [1., 1., 1., 0., 0.],  # 查询位置0能看到键位置0,1,2
        [1., 1., 1., 0., 0.],  # 查询位置1能看到键位置0,1,2
        [1., 1., 1., 0., 0.],  # 查询位置2能看到键位置0,1,2
        [0., 0., 0., 0., 0.],  # 查询位置3是填充，什么都看不到
        [0., 0., 0., 0., 0.]   # 查询位置4是填充，什么都看不到
    ]],
    [[  # 样本2
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1.]
    ]]
]).to(device)

    # 4. 前向传播
    output = encoder(input_ids, mask)
    print(f"输入形状: {input_ids.shape}")  # (2, 5)
    print(f"输出形状: {output.shape}")  # (2, 5, 512)
    print(f"输出设备: {output.device}")  # cuda:0 (如果在GPU上)