import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,  # 词表的token数
        embedding_dim: int,  # 词表的维度数
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()  # 获取当前显卡物理编号
        self.tp_size = dist.get_world_size()  # 获取显卡总数
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings  # 当前词表中的token总数
        # 每个显卡负责的token数
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        # 本显卡负责的起始token索引
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        # 本显卡负责的结束token索引
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        # 申请一块张量空间
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )
        # 猴子补丁,为张量加入weight_loader方法
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data  # 获取之前申请显存的指针
        # 获取当前显存空间在第0维的长度，也就是当前显卡负责的token数量
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size  # 算出当前显卡起始toekn的idx
        # 切分词表
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight) #复制负责部分词表进入显存

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            #显卡数大于1,说明当前输入的token中可能有不存在当前显卡上的
            #建立掩码张量，若当前token存在则对应位置为True，否则为False
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            #此时不存在的token的ID全部被设置为了0，防止访问到了非法的显存空间
            x = mask * (x - self.vocab_start_idx)
        #进行Embedding
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            #由于我们使用了错误的idx：0,现在需要将错误计算出来的值全部归0
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)#整合所有显卡上的结果(矩阵向加)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        #因为入口矩阵和出口矩阵在矩阵规模上是一样的，所以只需调用父类的构造函数
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context() # 获取当前上下文
        if context.is_prefill:
            # 获取每个请求的最后一个 token 用来做预测
            last_indices = context.cu_seqlens_q[1:] - 1
            # 产生一个新的包含每个请求的最后一个 token的新张量
            x = x[last_indices].contiguous()
        #进行矩阵相乘
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = (
                [torch.empty_like(logits) for _ in range(self.tp_size)]
                if self.tp_rank == 0
                else None
            )
            # 将所有数据汇总到0号显卡上
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
