from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    # 每个显存块的容量是多少个kv
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        # 该请求的编号
        self.seq_id = next(Sequence.counter)
        # 状态，正在等待GPU处理计算
        self.status = SequenceStatus.WAITING
        # 本次请求的所有token
        self.token_ids = copy(token_ids)
        # 最后一个token的id
        self.last_token = token_ids[-1]
        # 本次请求的所有token的长度
        self.num_tokens = len(self.token_ids)
        # 用户输入的token长度
        self.num_prompt_tokens = len(token_ids)
        # 目前有多少个词的k v被缓存进了显存
        self.num_cached_tokens = 0
        # 目前该请求使用的显存块
        self.block_table = []
        # 用户的采样偏好
        self.temperature = sampling_params.temperature
        # 最大允许生成token数量
        self.max_tokens = sampling_params.max_tokens
        # 忽略eos，继续生成
        self.ignore_eos = sampling_params.ignore_eos

    # 返回目前的所有token的长度
    def __len__(self):
        return self.num_tokens

    # 允许访问Sequence[key]
    def __getitem__(self, key):
        return self.token_ids[key]

    # 本次请求是否结束
    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    # 本次请求生成的token长度
    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    # 本次请求的输入tokenID列表
    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    # 本次请求生成的tokenID列表
    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    # 缓存的kv用了多少个显存块
    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    # 向上取整计算当前所有token需要使用多少个显存块
    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    # 本请求的最后一个显存块中的token数量
    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    # 获得对应显存块中的对应的tokenID 
    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    # 追加一个token
    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1


    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
