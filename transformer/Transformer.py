import torch
from torch import nn

from transformer.Decoder import Decoder
from transformer.Encoder import Encoder


class Transformer(nn.Module):
    """
    完整的Transformer模型（编码器-解码器架构）

    Args:
        src_vocab_size: 源语言词汇表大小
        trg_vocab_size: 目标语言词汇表大小
        src_pad_idx: 源语言padding token的索引
        trg_pad_idx: 目标语言padding token的索引
        embed_size: 嵌入维度（默认512）
        num_layers: 编码器和解码器的层数（默认6）
        forward_expansion: 前馈网络扩展因子（默认4）
        heads: 注意力头数（默认8）
        dropout: Dropout概率（默认0）
        device: 设备（默认"cuda"）
        max_length: 最大序列长度（默认100）
    """

    def __init__(
            self,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            trg_pad_idx,
            embed_size=512,
            num_layers=6,
            forward_expansion=4,
            heads=8,
            dropout=0,
            device="cuda",
            max_length=100,
    ):
        super(Transformer, self).__init__()

        # 创建编码器
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        # 创建解码器
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        # 保存padding索引和设备
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

        # 将整个模型移动到指定设备
        self.to(device)

    def make_src_mask(self, src):
        """
        创建源序列掩码

        Args:
            src: 源序列token id [batch_size, src_len]

        Returns:
            src_mask: 源序列掩码 [batch_size, 1, 1, src_len]
                    1表示真实token，0表示padding
        """
        # src != self.src_pad_idx: [batch_size, src_len]，padding位置为False
        # unsqueeze(1).unsqueeze(2): 添加两个维度用于广播 [batch_size, 1, 1, src_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)  # 移动到模型所在的设备

    def make_trg_mask(self, trg):
        """
        创建目标序列掩码（因果掩码）

        Args:
            trg: 目标序列token id [batch_size, trg_len]

        Returns:
            trg_mask: 目标序列掩码 [batch_size, 1, trg_len, trg_len]
                    下三角矩阵，1表示允许关注，0表示屏蔽
        """
        N, trg_len = trg.shape

        # 创建下三角矩阵作为因果掩码（只能看到当前位置及之前）
        # torch.tril(torch.ones((trg_len, trg_len))): [trg_len, trg_len]
        # expand(N, 1, trg_len, trg_len): 扩展到批次维度 [batch_size, 1, trg_len, trg_len]
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask.to(self.device)  # 移动到模型所在的设备

    def forward(self, src, trg):
        """
        前向传播

        Args:
            src: 源序列token id [batch_size, src_len]
            trg: 目标序列token id [batch_size, trg_len]

        Returns:
            out: 词汇表上的概率分布 [batch_size, trg_len, trg_vocab_size]
        """
        # 1. 创建掩码
        src_mask = self.make_src_mask(src)  # 源序列padding掩码

        # 使用简单的因果掩码（原代码）或包含padding的掩码
        trg_mask = self.make_trg_mask(trg)  # 目标序列因果掩码

        # 2. 编码器前向传播
        # enc_src: [batch_size, src_len, embed_size]
        enc_src = self.encoder(src, src_mask)

        # 3. 解码器前向传播
        # out: [batch_size, trg_len, trg_vocab_size]
        out = self.decoder(trg, enc_src, src_mask, trg_mask)

        return out


if __name__ == "__main__":
    # 1. 设备检测 - 自动选择GPU或CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)  # 输出: cuda 或 cpu

    # 2. 创建源序列(src)和目标序列(trg)的token id
    # 假设词汇表大小=10 (token 0-9)
    # token 0: <pad> (填充)
    # token 1: <sos> (序列开始)
    # token 2: <eos> (序列结束)
    # token 3-9: 普通词汇

    # 源序列 (英文句子)
    x = torch.tensor([
        [1, 5, 6, 4, 3, 9, 5, 2, 0],  # 句子1: <sos> 5 6 4 3 9 5 <eos> <pad>
        [1, 8, 7, 3, 4, 5, 6, 7, 2]  # 句子2: <sos> 8 7 3 4 5 6 7 <eos>
    ]).to(device)

    # 目标序列 (法文句子)
    trg = torch.tensor([
        [1, 7, 4, 3, 5, 9, 2, 0],  # 句子1: <sos> 7 4 3 5 9 <eos> <pad>
        [1, 5, 6, 2, 4, 7, 6, 2]  # 句子2: <sos> 5 6 <eos> 4 7 6 <eos>
    ]).to(device)

    # 3. 模型参数设置
    src_pad_idx = 0  # 源语言填充token索引
    trg_pad_idx = 0  # 目标语言填充token索引
    src_vocab_size = 10  # 源词汇表大小 (0-9)
    trg_vocab_size = 10  # 目标词汇表大小 (0-9)

    # 4. 创建Transformer模型并移到设备
    model = Transformer(
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        device=device
    ).to(device)

    # 5. 前向传播 - 关键：训练时的教师强制(teacher forcing)
    # trg[:, :-1] = 去掉最后一个token
    # 输入: [<sos>, 7, 4, 3, 5, 9, <eos>]
    # 预测: [7, 4, 3, 5, 9, <eos>, <pad>]
    out = model(x, trg[:, :-1])

    # 6. 输出形状解释
    print(out.shape)  # 输出: torch.Size([2, 7, 10])
    # [batch_size=2, sequence_length=7, vocab_size=10]