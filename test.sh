#!/bin/bash

BASE_URL=http://127.0.0.1:8888
OPENAI_API_KEY="NotNeeded"

curl $BASE_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "你好，可以介绍一下自己吗？"}],
    "stream": true,
    "temperature": 0.9
  }'
