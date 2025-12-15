import torch
from torch import nn

from transformer.attention import SelfAttention  # 导入之前实现的自注意力模块


class TransformerBlock(nn.Module):
    """
    Transformer块（标准结构）

    每个Transformer Block包含：
    1. 多头自注意力层 (Multi-head Self-Attention)
    2. 前馈神经网络层 (Feed-Forward Network)
    3. 残差连接和层归一化 (Residual Connection + LayerNorm)
    4. Dropout正则化
    """

    def __init__(self, embed_size, heads, dropout, forward_expansion):
        """
        参数说明：
        - embed_size: 词嵌入维度（如512）
        - heads: 注意力头数量（如8）
        - dropout: Dropout概率（如0.1）
        - forward_expansion: 前馈网络扩展倍数（通常为4）
        """
        super(TransformerBlock, self).__init__()

        # 1. 多头自注意力层
        self.attention = SelfAttention(embed_size, heads)

        # 2. 层归一化（LayerNorm） - 用于稳定训练
        # 对每个样本的特征维度进行归一化（batch_size, seq_len, embed_size）-> 对最后一个维度归一化
        self.norm1 = nn.LayerNorm(embed_size)  # 输入形状：(..., embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # 3. 前馈神经网络（FFN）
        # 标准结构：线性层 -> ReLU -> 线性层
        # 扩展：通常将维度扩大4倍后再缩回原维度
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),  # 扩展维度
            nn.ReLU(),  # 激活函数
            nn.Linear(forward_expansion * embed_size, embed_size),  # 恢复维度
        )

        # 4. Dropout层（防止过拟合）
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        """
        前向传播

        输入形状：
        - query, key, value: (batch_size, seq_len, embed_size)
        - mask: (batch_size, 1, seq_len, seq_len) 或 None

        输出形状：
        - out: (batch_size, seq_len, embed_size)
        """
        # ============== 第1部分：自注意力 ==============
        # attention形状: (batch_size, seq_len, embed_size)
        attention = self.attention(query, key, value, mask)

        # ============== 第2部分：残差连接 + 层归一化 + Dropout ==============
        # 1. 残差连接: attention + query（跳跃连接）
        # 2. 层归一化: 稳定训练
        # 3. Dropout: 防止过拟合
        # 形状保持: (batch_size, seq_len, embed_size)
        x = self.dropout(self.norm1(attention + query))

        # ============== 第3部分：前馈网络 ==============
        # 形状: (batch_size, seq_len, embed_size)
        forward = self.feed_forward(x)

        # ============== 第4部分：第二次残差连接 ==============
        # 再次进行残差连接和归一化
        # 形状: (batch_size, seq_len, embed_size)
        out = self.dropout(self.norm2(forward + x))

        return out


# ================== 使用示例 ==================
if __name__ == "__main__":
    # 示例参数
    batch_size = 4  # 批大小
    seq_len = 20  # 序列长度
    embed_size = 512  # 嵌入维度
    heads = 8  # 注意力头数
    dropout = 0.1  # Dropout率
    forward_expansion = 4  # 前馈网络扩展倍数

    # 创建Transformer块
    transformer_block = TransformerBlock(
        embed_size=embed_size,
        heads=heads,
        dropout=dropout,
        forward_expansion=forward_expansion
    )

    # 创建输入数据
    query = torch.randn(batch_size, seq_len, embed_size)
    key = torch.randn(batch_size, seq_len, embed_size)
    value = torch.randn(batch_size, seq_len, embed_size)

    # 前向传播
    output = transformer_block(query, key, value, mask=None)

    print(f"输入形状: {query.shape}")  # (4, 20, 512)
    print(f"输出形状: {output.shape}")  # (4, 20, 512)
    print(f"是否保持形状: {query.shape == output.shape}")  # True