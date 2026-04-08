from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False #当前推理阶段处于什么状态
    
    # 推理分两个阶段，第一阶段是用户的输入 + Prompt，此时的Q是一个由多个token向量构成的矩阵
    # 在这个阶段中，将器用flashattention策略
    # 第二个阶段，是decoder已经完成了第一个token的预测，此时以及之后的Q只是一个token的向量
    # 此时应该采用page_attention策略
    
    # 在第一个阶段中，为处理多个用户传来的请求，我们需要将多个用户的请求进行批处理
    # 旧的方法是根据多个请求的token创建矩阵，但是为了对齐，需要将token数少的请求补齐无意义的token
    # 这就可能造成了极大的资源浪费
    # 为了性能的提高，放弃了上述这种用法，而是将不同的请求首尾相连，形成一个一维的向量
    # 为了分割每个句子，于是需要距离每个句子在一维向量中起始的索引
    
    # 下面两个成员中记录了每个用户请求的起始token的ID
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    # 下面的两个成员记录了本批次请求的最大的token数，用以申请SRAM的空间
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    
    # 每个批次每个对话下一个kv应该储存的位置
    slot_mapping: torch.Tensor | None = None
    # 当前批次的上下文长度，也就是历史kv个数
    context_lens: torch.Tensor | None = None
    # 每个对话所使用的block编号
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
