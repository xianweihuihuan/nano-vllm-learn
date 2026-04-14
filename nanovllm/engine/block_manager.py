from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        # 申请一个块
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # 重置块的信息
        block.reset()
        # 清除该块的空闲状态
        self.free_block_ids.remove(block_id)
        # 加入以使用的块的set
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            # 获取当前块中所有的tokenID
            token_ids = seq.block(i)
            # 若不是最后一个块，则计算并更新hash前缀值，否则为-1
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            # 若是这个hash值出现过，则取对应的块以及之前的块都存在
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            # 若是缓存未命中，则申请新的块，同时之后的也不需要查缓存了
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 命中缓存，若是这个块正在被用，则加一个引用，否则就去空闲队列申请
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            # 更新块的hash信息
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        # 倒序释放
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            # 引用计数减1
            block.ref_count -= 1
            # 若是此时引用计数器为0,则可以释放该块
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        # 清理请求的信息
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            # 此时需要申请新的缓存块
            # 目前最后一块的hash值不能为-1
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            # 此时最后一块显存块正好被填满，不需要申请新的但需要生成hash值
            assert last_block.hash == -1
            # 获取最后一块的tokenID列表
            token_ids = seq.block(seq.num_blocks-1)
            # 获取前一块的hash值，若该块是第一块，则为-1
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            # 计算hash值
            h = self.compute_hash(token_ids, prefix)
            # 更新信息
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            # 该块没满，不做处理
            assert last_block.hash == -1
