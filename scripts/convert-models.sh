#!/bin/bash

# One can copy following shared Google colab notebook, which provide enough resourse, to run following code
#  https://colab.research.google.com/drive/1GqZu6zmCy2vMNMZqO78DDuHvwv_9vJcp?usp=sharing

HOME_DIR=/content/
cd $HOME_DIR

git clone --recursive https://github.com/li-plus/chatglm.cpp.git
mkdir -p $HOME_DIR/chatglm.cpp/THUDM $HOME_DIR/chatglm.cpp/models

cd $HOME_DIR/chatglm.cpp/THUDM
git clone https://huggingface.co/THUDM/chatglm2-6b

# GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/THUDM/chatglm2-6b
# cd chatglm2-6b
# wget 'https://cloud.tsinghua.edu.cn/d/674208019e314311ab5c/files/?p=%2Fchatglm2-6b%2Ftokenizer.model&dl=1' -O tokenizer.model
# wget 'https://cloud.tsinghua.edu.cn/d/674208019e314311ab5c/files/?p=%2Fchatglm2-6b%2Fpytorch_model-00001-of-00007.bin&dl=1' -O pytorch_model-00001-of-00007.bin
# wget 'https://cloud.tsinghua.edu.cn/d/674208019e314311ab5c/files/?p=%2Fchatglm2-6b%2Fpytorch_model-00002-of-00007.bin&dl=1' -O pytorch_model-00002-of-00007.bin
# wget 'https://cloud.tsinghua.edu.cn/d/674208019e314311ab5c/files/?p=%2Fchatglm2-6b%2Fpytorch_model-00003-of-00007.bin&dl=1' -O pytorch_model-00003-of-00007.bin
# wget 'https://cloud.tsinghua.edu.cn/d/674208019e314311ab5c/files/?p=%2Fchatglm2-6b%2Fpytorch_model-00004-of-00007.bin&dl=1' -O pytorch_model-00004-of-00007.bin
# wget 'https://cloud.tsinghua.edu.cn/d/674208019e314311ab5c/files/?p=%2Fchatglm2-6b%2Fpytorch_model-00005-of-00007.bin&dl=1' -O pytorch_model-00005-of-00007.bin
# wget 'https://cloud.tsinghua.edu.cn/d/674208019e314311ab5c/files/?p=%2Fchatglm2-6b%2Fpytorch_model-00006-of-00007.bin&dl=1' -O pytorch_model-00006-of-00007.bin
# wget 'https://cloud.tsinghua.edu.cn/d/674208019e314311ab5c/files/?p=%2Fchatglm2-6b%2Fpytorch_model-00007-of-00007.bin&dl=1' -O pytorch_model-00007-of-00007.bin

cd $HOME_DIR/chatglm.cpp
pip install tabulate tqdm transformers sentencepiece
python3 convert.py -i THUDM/chatglm2-6b -t q8_0 -o models/chatglm2-ggml.bin