import torch
from torch import nn

from transformer.DecoderBlock import DecoderBlock


class Decoder(nn.Module):
    """
    Transformer解码器

    将解码器块的输出转换为词汇表上的概率分布
    包含词嵌入、位置编码、多个DecoderBlock层和输出线性层

    Args:
        trg_vocab_size: 目标词汇表大小
        embed_size: 嵌入维度
        num_layers: 解码器层数
        heads: 注意力头数
        forward_expansion: 前馈网络扩展因子
        dropout: Dropout概率
        device: 设备
        max_length: 最大序列长度（用于位置编码）
    """

    def __init__(
            self,
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device

        # 词嵌入层：将token id转换为向量 [vocab_size, embed_size]
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)

        # 位置编码层：学习的位置嵌入 [max_length, embed_size]
        self.position_embedding = nn.Embedding(max_length, embed_size)

        # 堆叠多个DecoderBlock层
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )

        # 输出线性层：将隐藏状态映射回词汇表 [embed_size, vocab_size]
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        """
        前向传播

        Args:
            x: 目标序列token id [batch_size, seq_length]
            enc_out: 编码器输出 [batch_size, src_seq_length, embed_size]
            src_mask: 源序列掩码 [batch_size, 1, 1, src_seq_length]
            trg_mask: 目标序列掩码 [batch_size, 1, trg_seq_length, trg_seq_length]

        Returns:
            out: 词汇表上的概率分布 [batch_size, seq_length, vocab_size]
        """
        # 获取输入形状
        N, seq_length = x.shape  # N=batch_size

        # 创建位置索引 [0, 1, 2, ..., seq_length-1]，扩展为批次大小
        # positions形状: [batch_size, seq_length]
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        # 1. 词嵌入 + 位置编码
        # word_embedding(x): [N, seq_length] -> [N, seq_length, embed_size]
        # position_embedding(positions): [N, seq_length] -> [N, seq_length, embed_size]
        # 相加后dropout
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        # 2. 通过多个DecoderBlock层
        # 每个DecoderBlock接收：
        #   x: 当前层输入 [N, seq_length, embed_size]
        #   enc_out, enc_out: 作为key和value [N, src_seq_length, embed_size]
        #   src_mask: 编码器padding掩码
        #   trg_mask: 解码器因果+padding掩码
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
            # 输出形状保持: [N, seq_length, embed_size]

        # 3. 输出投影到词汇表
        # fc_out: [N, seq_length, embed_size] -> [N, seq_length, vocab_size]
        out = self.fc_out(x)

        return out