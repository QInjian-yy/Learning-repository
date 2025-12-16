import torch
from torch import nn

from transformer.TransformerBlock import TransformerBlock
from transformer.attention import SelfAttention


class DecoderBlock(nn.Module):
    """
    Transformer解码器块

    解码器块包含：
    1. 带掩码的自注意力机制（用于解码时只能看到前面位置）
    2. 编码器-解码器注意力机制（通过TransformerBlock实现）
    3. 层归一化和残差连接

    Args:
        embed_size: 嵌入维度
        heads: 注意力头的数量
        forward_expansion: 前馈网络扩展因子
        dropout: Dropout概率
        device: 设备（CPU/GPU）
    """

    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()

        # 层归一化（用于残差连接后的归一化）
        self.norm = nn.LayerNorm(embed_size)

        # 带掩码的自注意力机制（用于处理目标序列）
        # 在解码时，每个位置只能关注当前位置及之前的位置
        self.attention = SelfAttention(embed_size, heads=heads)

        # Transformer块，这里用作编码器-解码器注意力机制
        # 包含多头注意力、前馈网络、层归一化和残差连接
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )

        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key, value, src_mask, trg_mask):
        """
        前向传播

        Args:
            x: 目标序列输入 (query) [batch_size, trg_seq_length, embed_size]
            key: 编码器输出的key [batch_size, src_seq_length, embed_size]
            value: 编码器输出的value [batch_size, src_seq_length, embed_size]
            src_mask: 源序列掩码 [batch_size, 1, 1, src_seq_length]
                    用于在编码器-解码器注意力中屏蔽padding位置
            trg_mask: 目标序列掩码 [batch_size, 1, trg_seq_length, trg_seq_length]
                    用于在自注意力中屏蔽未来位置和padding位置

        Returns:
            out: 解码器输出 [batch_size, trg_seq_length, embed_size]
        """
        # 第一步：带掩码的自注意力（目标序列的自注意力）
        # 使用目标序列掩码确保每个位置只能看到当前位置及之前的位置
        attention = self.attention(x, x, x, trg_mask)

        # 第二步：残差连接 + Dropout + 层归一化
        # 公式：LayerNorm(x + Dropout(SelfAttention(x)))
        query = self.dropout(self.norm(attention + x))

        # 第三步：编码器-解码器注意力（通过TransformerBlock实现）
        # query来自解码器，key和value来自编码器输出
        # 使用源序列掩码屏蔽编码器输出的padding位置
        out = self.transformer_block(query, key, value, src_mask)

        return out

if __name__ == '__main__':
    """
        理解
        src_mask: 源序列掩码 [batch_size, 1, 1, src_seq_length]
                    用于在编码器-解码器注意力中屏蔽padding位置
        trg_mask: 目标序列掩码 [batch_size, 1, trg_seq_length, trg_seq_length]
                    用于在自注意力中屏蔽未来位置和padding位置
    """


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 参数
    batch_size = 2
    src_len = 3  # 源序列长度
    trg_len = 4  # 目标序列长度
    d_model = 8  # 嵌入维度

    # 1. 目标序列输入 [batch, trg_len, d_model]
    x = torch.tensor([
        [[1.0] * d_model, [2.0] * d_model, [3.0] * d_model, [0.0] * d_model],  # 最后一个是padding
        [[4.0] * d_model, [5.0] * d_model, [0.0] * d_model, [0.0] * d_model]  # 后两个是padding
    ], device=device)

    # 2. 编码器输出 [batch, src_len, d_model]
    key = torch.tensor([
        [[10.0] * d_model, [11.0] * d_model, [0.0] * d_model],  # 最后一个是padding
        [[20.0] * d_model, [21.0] * d_model, [22.0] * d_model]  # 全是真实词
    ], device=device)

    value = key * 2  # 简单创建value

    # 3. src_mask [batch, 1, 1, src_len] - 屏蔽padding
    src_mask = torch.tensor([
        [[[1, 1, 0]]],  # 屏蔽第3个位置(padding)
        [[[1, 1, 1]]]  # 不屏蔽
    ], device=device, dtype=torch.float)

    # 4. trg_mask [batch, 1, trg_len, trg_len] - 因果掩码 + padding屏蔽
    trg_mask = torch.tensor([
        [[  # 句子1: 前3个词，第4个padding
            [1, 0, 0, 0],  # 位置0: 只看自己
            [1, 1, 0, 0],  # 位置1: 看0,1
            [1, 1, 1, 0],  # 位置2: 看0,1,2
            [0, 0, 0, 0]  # 位置3(padding): 不看任何
        ]],
        [[  # 句子2: 前2个词，后2个padding
            [1, 0, 0, 0],  # 位置0: 只看自己
            [1, 1, 0, 0],  # 位置1: 看0,1
            [0, 0, 0, 0],  # 位置2(padding): 不看任何
            [0, 0, 0, 0]  # 位置3(padding): 不看任何
        ]]
    ], device=device, dtype=torch.float)