import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim  # Wq Wk Wv矩阵如何被切分(竖着还是横着)
        self.tp_rank = dist.get_rank()  # 获取当前显卡物理编号
        self.tp_size = dist.get_world_size()  # 获取显卡总个数
        self.weight = nn.Parameter(
            torch.empty(output_size, input_size)
        )  # 向显存申请output_size * input_size 的空间
        #猴子补丁，加入weight_loader方法，子类实现
        self.weight.weight_loader = self.weight_loader 
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    # 将目标文件按显卡切分并放入对应的显存
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int
    ):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

# Wq Wk Wv权重矩阵
class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,# 维度数
        head_size: int,# 每个头的维度数
        total_num_heads: int, # 总共的Q的头数
        total_num_kv_heads: int | None = None, # 总共的KV头数
        bias: bool = False,
    ):
        tp_size = dist.get_world_size() # 获取总共的显卡数
        # kv头数
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        # 每个头的维度数
        self.head_size = head_size
        # 本显卡负责的头数
        self.num_heads = divide(total_num_heads, tp_size)
        # 本显卡负责的kv头数
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        # 总矩阵的列数
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        # 初始化父类，开辟空间
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str
    ):
        # 获取显存地址
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            # Wq权重矩阵占据了多少空间
            shard_size = self.num_heads * self.head_size
            # Wq权重矩阵在本矩阵的偏移量
            shard_offset = 0
        elif loaded_shard_id == "k":
            # Wk权重矩阵占据了多少空间
            shard_size = self.num_kv_heads * self.head_size
            # Wk权重矩阵在本矩阵的偏移量
            shard_offset = self.num_heads * self.head_size
        else:
            # Wv权重矩阵占据了多少空间
            shard_size = self.num_kv_heads * self.head_size
            # Wv权重矩阵在本矩阵的偏移量
            shard_offset = (
                self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            ) 
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 找出在权重文件中应该加入到本显卡的部分
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        # 加载权重矩阵
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
