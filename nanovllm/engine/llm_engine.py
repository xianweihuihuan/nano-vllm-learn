import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}  # 解析配置字段
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)  # 构建配置信息
        self.ps = []  # 子进程列表
        self.events = []  # 通信机制
        ctx = mp.get_context("spawn")  # 创建子进程上下文，确定创建方式为孵化
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()  # 信号量
            process = ctx.Process(
                target=ModelRunner, args=(config, i, event)
            )  # 确定子进程的执行目标
            process.start()  # 开始执行子进程
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, use_fast=True
        )  # 建立分词器
        config.eos = self.tokenizer.eos_token_id
        # 主显卡，建立调度器
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    # 退出函数，包含资源清理逻辑
    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            # 回收子进程资源
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            # 如果传入的是一个字符串，就将其tokenization
            prompt = self.tokenizer.encode(prompt)
        # 生成对应请求
        seq = Sequence(prompt, sampling_params)
        # 添加请求到调度器
        self.scheduler.add(seq)

    # 执行向前传播
    def step(self):
        # 获取待执行列表
        seqs, is_prefill = self.scheduler.schedule()
        # 执行推理，进行向前传递，预测下一个token
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # 更新每个请求的状态
        self.scheduler.postprocess(seqs, token_ids)
        # 获取完成的请求
        outputs = [
            (seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished
        ]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]], # 用户请求，可能是字符串或者token的列表
        sampling_params: SamplingParams | list[SamplingParams], # 采样器
        use_tqdm: bool = True, # 是否打印进度条
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            # 如果只传入了一个采样器，则生成对应长度的列表
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            # 添加该请求
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        while not self.is_finished():
            t = perf_counter()
            # 完成一次完整的向前传播
            output, num_tokens = self.step()
            if use_tqdm:
                # 下面的是测速代码
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )
            
            for seq_id, token_ids in output:
                # 收集已完成的请求
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]
        if use_tqdm:
            pbar.close()
        return outputs
