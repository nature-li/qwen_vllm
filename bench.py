import time
import os
from vllm import LLM, SamplingParams

llm = LLM(
    model=os.path.expanduser('~/models/Qwen2.5-3B-Instruct'),
    dtype='float16',
    disable_log_stats=True,
)

prompts = ["讲一下 transformer"] * 10
sampling_params = SamplingParams(temperature=0, max_tokens=200)

t0 = time.time()
outputs = llm.generate(prompts, sampling_params)
t1 = time.time()

total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
elapsed = t1 - t0

print(f"Total tokens : {total_tokens}")
print(f"Time         : {elapsed:.2f}s")
print(f"Throughput   : {total_tokens/elapsed:.2f} tokens/s")