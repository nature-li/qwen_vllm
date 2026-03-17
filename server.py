import asyncio
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
import os

app = FastAPI()

# 全局 engine
engine: AsyncLLMEngine = None


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.8
    priority: int = 0


class GenerateResponse(BaseModel):
    request_id: str
    text: str


@app.on_event("startup")
async def startup():
    global engine
    engine_args = AsyncEngineArgs(
        model=os.path.expanduser("~/models/Qwen2.5-3B-Instruct"),
        dtype="float16",
        enable_prefix_caching=True,
        max_model_len=4096,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    request_id = random_uuid()

    # async generator，逐 token 流式生成
    results_generator = engine.generate(
        request.prompt,
        sampling_params,
        request_id,
    )

    # 等最终结果
    final_output = None
    async for output in results_generator:
        final_output = output

    text = final_output.outputs[0].text
    return GenerateResponse(request_id=request_id, text=text)


@app.post("/generate_stream")
async def generate_stream(request: GenerateRequest):
    from fastapi.responses import StreamingResponse

    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    request_id = random_uuid()

    async def stream():
        prev_len = 0
        async for output in engine.generate(
            request.prompt, sampling_params, request_id
        ):
            text = output.outputs[0].text
            # 只输出新增的部分
            new_text = text[prev_len:]
            prev_len = len(text)
            if new_text:
                yield new_text

    return StreamingResponse(stream(), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
