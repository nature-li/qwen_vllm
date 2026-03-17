# 普通请求
curl -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "讲一下 transformer", "max_tokens": 200}'

# 流式请求
curl -X POST http://localhost:8000/generate_stream \
    -H "Content-Type: application/json" \
    -d '{"prompt": "讲一下 transformer", "max_tokens": 200}'