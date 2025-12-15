import torch
from torch import nn


class SelfAttention(nn.Module):
    """
    多头自注意力机制模块

    基于Transformer架构中的注意力机制，允许序列中的每个位置同时关注
    序列中所有其他位置的信息，并通过多头机制并行学习不同的表示子空间。

    Args:
        embed_size (int): 输入/输出的嵌入维度（通常为512、768、1024等）
        heads (int): 注意力头的数量（通常为8、12、16等）
    """

    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        # 保存超参数
        self.embed_size = embed_size  # 总嵌入维度
        self.heads = heads  # 注意力头数量
        self.head_dim = embed_size // heads  # 每个头的维度

        # 验证嵌入维度能被头数整除
        assert (
                self.heads * self.head_dim == self.embed_size
        ), "嵌入维度必须能被注意力头数整除"

        # Q、K、V的线性变换层
        # 这三个层将输入分别投影到查询、键、值空间
        # 输出维度保持embed_size不变，但内部权重不同
        self.queries = nn.Linear(embed_size, embed_size)  # 查询投影
        self.keys = nn.Linear(embed_size, embed_size)  # 键投影
        self.values = nn.Linear(embed_size, embed_size)  # 值投影

        # 输出线性层，将多头注意力输出融合回原始嵌入空间
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, queries, keys, values, mask):
        """
        前向传播过程

        Args:
            queries: 查询张量，形状为 (batch_size, query_len, embed_size)
            keys: 键张量，形状为 (batch_size, key_len, embed_size)
            values: 值张量，形状为 (batch_size, value_len, embed_size)
            mask: 注意力掩码，用于屏蔽不需要关注的位置
                  None 或形状为 (batch_size, 1, query_len, key_len) 的张量
                  mask中为0的位置将被替换为极小的负数

        Returns:
            out: 注意力输出，形状为 (batch_size, query_len, embed_size)
        """
        batch_size = queries.shape[0]  # 批大小

        # 获取各序列长度
        query_len = queries.shape[1]  # 查询序列长度
        key_len = keys.shape[1]  # 键序列长度
        value_len = values.shape[1]  # 值序列长度（通常key_len=value_len）

        # ============= 第1步：线性投影 =============
        # 将输入分别投影到Q、K、V空间
        # 形状保持不变：(batch_size, seq_len, embed_size)
        queries = self.queries(queries)  # 形状: (batch_size, query_len, embed_size)
        keys = self.keys(keys)  # 形状: (batch_size, key_len, embed_size)
        values = self.values(values)  # 形状: (batch_size, value_len, embed_size)

        # ============= 第2步：多头拆分 =============
        # 将embed_size维度拆分为 (heads × head_dim)
        # 重塑为四维张量：(batch_size, seq_len, heads, head_dim)
        queries = queries.reshape(batch_size, query_len, self.heads, self.head_dim)
        keys = keys.reshape(batch_size, key_len, self.heads, self.head_dim)
        values = values.reshape(batch_size, value_len, self.heads, self.head_dim)

        # ============= 第3步：计算注意力分数 =============
        # 使用einsum计算查询和键的点积（缩放前）
        # "nqhd,nkhd->nhqk" 的解释：
        #   n: batch_size维度
        #   q: query_len维度
        #   k: key_len维度
        #   h: heads维度
        #   d: head_dim维度
        # 对head_dim(d)维度进行求和，得到每个查询对每个键的注意力分数
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # energy形状: (batch_size, heads, query_len, key_len)
        # 每个元素表示：对于批次n、头h、查询位置q、键位置k的注意力分数

        # ============= 第4步：应用掩码（可选） =============
        # 掩码用于处理变长序列或防止未来信息泄露（解码器自注意力）
        if mask is not None:
            # 将掩码中为0的位置替换为极小的负数（-1e20）
            # 这样在softmax后，这些位置的权重接近0
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # ============= 第5步：计算注意力权重 =============
        # 1. 缩放：除以sqrt(embed_size)以稳定梯度
        # 2. softmax：将注意力分数转换为概率分布
        #    dim=3表示对key_len维度进行softmax
        #    每个查询位置对所有键位置的注意力权重和为1
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention形状: (batch_size, heads, query_len, key_len)
        # 每个元素表示：对于批次n、头h、查询位置q、键位置k的注意力权重（0到1之间）

        # ============= 第6步：加权求和 =============
        # 使用注意力权重对值进行加权求和
        # "nhqk,nkhd->nqhd" 的解释：
        #   使用注意力权重(nhqk)对值(nkhd)进行加权
        #   对key_len(k)维度进行求和
        out = torch.einsum("nhqk,nkhd->nqhd", [attention, values])
        # 中间形状: (batch_size, query_len, heads, head_dim)

        # 重塑为三维张量：合并最后两个维度(heads × head_dim = embed_size)
        out = out.reshape(batch_size, query_len, self.heads * self.head_dim)
        # 形状: (batch_size, query_len, embed_size)

        # ============= 第7步：输出投影 =============
        # 最后的线性层融合所有头的输出信息
        out = self.fc_out(out)
        # 形状保持不变: (batch_size, query_len, embed_size)

        return out


# ============= 使用示例 =============
if __name__ == "__main__":
    # 示例参数
    batch_size = 4  # 批大小
    seq_len = 10  # 序列长度
    embed_size = 512  # 嵌入维度
    heads = 8  # 注意力头数量

    # 创建自注意力模块
    attention = SelfAttention(embed_size, heads)

    # 创建示例输入（通常Q、K、V相同，用于自注意力）
    queries = torch.randn(batch_size, seq_len, embed_size)
    keys = torch.randn(batch_size, seq_len, embed_size)
    values = torch.randn(batch_size, seq_len, embed_size)

    # 创建示例掩码（可选）
    # 假设前5个位置有效，后5个位置填充
    """
    维度解释：
        - batch_size: 批大小（独立处理每个样本）
        - 1:         多头注意力中，所有头共享相同的掩码（可广播到heads维度）
        - query_len: 查询序列长度（行数）
        - key_len:   键序列长度（列数）
    """
    mask = torch.ones(batch_size, 1, seq_len, seq_len)
    mask[:, :, :, 5:] = 0  # 将后5个键位置掩码掉

    # 前向传播
    output = attention(queries, keys, values, mask)

    print(f"输入形状: {queries.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力头数: {heads}")
    print(f"每个头维度: {embed_size // heads}")

    # 验证维度
    assert output.shape == (batch_size, seq_len, embed_size), "输出形状不正确"
    print("✓ 自注意力模块运行正常")