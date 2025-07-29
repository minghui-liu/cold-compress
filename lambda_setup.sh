#!/bin/bash

## download miniconda
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh
# source ~/miniconda3/bin/activate
# ~/miniconda3/bin/conda init
# source ~/.bashrc

## initialize conda
# conda create --name hashevict python=3.10 -y
# conda activate hashevict

# clone the repo
git clone https://github.com/minghui-liu/cold-compress.git

cd cold-compress
# install the dependencies
# pip install --user -r requirements.txt --extra-index-url https://download.pytorch.org/whl/nightly/
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/nightly/

python -c "import torch; print('CUDA', torch.version.cuda, 'available', torch.cuda.is_available())"


git config --global credential.helper store

## create .env file
# echo "HUGGINGFACE_TOKEN=your_token_here" > .env
# echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env
export HUGGINGFACE_TOKEN=
export OPENAI_API_KEY=

# huggingface login with env variable
# export $(grep -v '^#' .env | xargs -d '\n')
huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential

# setup llama3 
bash scripts/prepare_llama31.sh
# bash scripts/prepare_llama2.sh
# bash scripts/prepare_qwen2.sh

# print 'Setup done!'
echo 'Setup done!'


# sudo apt-get install git-lfs
# cd RULER
# git lfs install
# git lfs pull
# # RULER dependencies
# pip install wonderwords tenacity nemo nemo_toolkit[all] html2text huggingface_hub[cli]