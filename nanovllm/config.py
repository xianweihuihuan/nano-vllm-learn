import os
from dataclasses import dataclass
from transformers import AutoConfig


# 推理引擎的配置信息
@dataclass
class Config:
    model: str  # 模型文件的路径
    max_num_batched_tokens: int = 16384  # GPU同时处理token的上限(用户输入的语句的token上限)
    max_num_seqs: int = 512  # 最大并发请求数
    max_model_len: int = 4096 #单条序列最大上下文长度
    gpu_memory_utilization: float = 0.9  # 显存预留率
    tensor_parallel_size: int = 1#张量并行度(多少张显卡跑这个模型)
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None  # 模型的配置信息
    eos: int = -1  # EOF的tokenid，推理引擎检测到这个id会停止生成
    kvcache_block_size: int = 256  # 一个kvcache中容纳的K V张量的个数
    num_kvcache_blocks: int = -1  # kvcache_block的个数

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        #从模型目录中自动解析该模型的配置
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )
        assert self.max_num_batched_tokens >= self.max_model_len
