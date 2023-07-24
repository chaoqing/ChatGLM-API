FROM python:3.10-slim as build
RUN apt update && apt install -y gcc g++ make cmake --no-install-recommends && \
    pip install -U chatglm-cpp && \
    find /root/.cache/pip/wheels/ -name 'chatglm_cpp-*.whl' -exec mv {} /tmp/ \;

FROM python:3.10-slim
COPY --from=build /tmp/chatglm_cpp-*.whl /tmp/
RUN pip install -U /tmp/chatglm_cpp-*.whl

ADD . /opt/
WORKDIR /opt
RUN pip install -U -r requirements.txt

VOLUME /opt/models

CMD ["python3", "api.py", "-m", "./models/chatglm2-ggml.bin"]