#!/bin/sh
#$ -cwd
#$ -l node_f=32
#$ -l h_rt=96:00:00
#$ -o outputs/Llama-2-1.56b/$JOB_ID
#$ -e outputs/Llama-2-1.56b/$JOB_ID
#$ -p -5

set -e

# module load
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.1
module load ylab/cudnn/8.9.7
module load ylab/nccl/cuda-12.2/2.20.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

# python virtualenv
source .env/bin/activate

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
export NUM_GPU_PER_NODE=4
NODE_TYPE="h100"

NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r hostname _ rest; do
  echo "${hostname} slots=${NUM_GPU_PER_NODE}"
done <"$PE_HOSTFILE" >"$HOSTFILE_NAME"

# model config
HIDDEN_SIZE=2048
FFN_HIDDEN_SIZE=7168 # intermediate size (HuggingFace)
NUM_LAYERS=24
NUM_HEADS=16
NUM_KEY_VALUE_HEADS=8
SEQ_LENGTH=4096

# distributed settings
TENSOR_PARALLEL_SIZE=1   # fixed
PIPELINE_PARALLEL_SIZE=1  # num layers 24
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${TENSOR_PARALLEL_SIZE} * ${PIPELINE_PARALLEL_SIZE})))

# training config
MICRO_BATCH_SIZE=8
GLOBAL_BATCH_SIZE=1024
TRAIN_STEPS=238419 # 1000*(10**9)/1024/4096
LR_DECAY_ITERS=238419

LR=3e-4
MIN_LR=3e-5
LR_WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL=/gs/fs/tga-bayes-crest/taishi/workspace/llm-jp-tokenizer/models/ver2.2/code10K_en20K_ja30K.ver2.2.model
CHECKPOINT_SAVE_DIR=/gs/bs/tgh-NII-LLM/checkpoints/Llama-2-1.56b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-LR${LR}-MINLR${MIN_LR}-WD${WEIGHT_DECAY}-WARMUP${LR_WARMUP_STEPS}

mkdir -p ${CHECKPOINT_SAVE_DIR}

# data config
DATA_PATH=""

# Code:Stack
DATA_PATH="${DATA_PATH} 15171450262 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/code/stack_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 13328193698 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/code/stack_0001.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17975394047 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/code/stack_0002.jsonl_text_document"
DATA_PATH="${DATA_PATH} 9317460115 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/code/stack_0003.jsonl_text_document"
DATA_PATH="${DATA_PATH} 7113041631 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/code/stack_0004.jsonl_text_document"
DATA_PATH="${DATA_PATH} 9717201093 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/code/stack_0005.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 19657765759 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/code/stack_0006.jsonl_text_document"
DATA_PATH="${DATA_PATH} 12764560113 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/code/stack_0007.jsonl_text_document"
DATA_PATH="${DATA_PATH} 15805724468 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/code/stack_0008.jsonl_text_document"

# Ja:CC-1
DATA_PATH="${DATA_PATH} 15482587533 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/ja/cc-1_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 23734172418 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/ja/cc-1_0001.jsonl_text_document"
DATA_PATH="${DATA_PATH} 23137056566 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/ja/cc-1_0002.jsonl_text_document"
DATA_PATH="${DATA_PATH} 23421942863 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/ja/cc-1_0003.jsonl_text_document"
DATA_PATH="${DATA_PATH} 30155555857 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/ja/cc-1_0004.jsonl_text_document"
DATA_PATH="${DATA_PATH} 26354736055 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/ja/cc-1_0005.jsonl_text_document"
DATA_PATH="${DATA_PATH} 21812154498 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/ja/cc-1_0006.jsonl_text_document"
DATA_PATH="${DATA_PATH} 21125833303 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/ja/cc-1_0007.jsonl_text_document"
DATA_PATH="${DATA_PATH} 23558374586 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/ja/cc-1_0008.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17182985300 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/ja/cc-1_0009.jsonl_text_document"
DATA_PATH="${DATA_PATH} 15674053007 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/ja/cc-1_0010.jsonl_text_document"
DATA_PATH="${DATA_PATH} 22793633847 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/ja/cc-1_0011.jsonl_text_document"
DATA_PATH="${DATA_PATH} 19286706780 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/ja/cc-1_0012.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17318579808 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/ja/cc-1_0013.jsonl_text_document"

# Ja:CC-2
DATA_PATH="${DATA_PATH} 23306709262 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/ja/cc-2_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 26092739133 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/ja/cc-2_0001.jsonl_text_document"
DATA_PATH="${DATA_PATH} 20363062347 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/ja/cc-2_0002.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16389010619 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/ja/cc-2_0003.jsonl_text_document"
DATA_PATH="${DATA_PATH} 12371011542 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/ja/cc-2_0004.jsonl_text_document"

# Ja:CC-3
DATA_PATH="${DATA_PATH} 26693976874 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/ja/cc-3_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 26961161244 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/ja/cc-3_0001.jsonl_text_document"
DATA_PATH="${DATA_PATH} 19484491995 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/ja/cc-3_0002.jsonl_text_document"
DATA_PATH="${DATA_PATH} 14917267317 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/ja/cc-3_0003.jsonl_text_document"
DATA_PATH="${DATA_PATH} 10160804497 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/ja/cc-3_0004.jsonl_text_document"

# Ja:Kaken
DATA_PATH="${DATA_PATH} 1135596116 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/ja/kaken_0000.jsonl_text_document"

# Ja:WARP/HTML01-06
# DATA_PATH="${DATA_PATH} 857121835 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/ja/warp-html-01-06_0000.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 901076021 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/ja/warp-html-07-12_0000.jsonl_text_document"

# Ja:WARP/PDFe0
# DATA_PATH="${DATA_PATH} 18794584548 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train2_ver3.1.0/ja/warp-pdf-e00_0000.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 18790139221 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train2_ver3.1.0/ja/warp-pdf-e00_0001.jsonl_text_document"

# Ja:WARP/PDFe0.2
# DATA_PATH="${DATA_PATH} 18661909722 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train2_ver3.1.0/ja/warp-pdf-e02_0000.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 16036586560 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train2_ver3.1.0/ja/warp-pdf-e02_0001.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 15067110947 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train2_ver3.1.0/ja/warp-pdf-e02_0002.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 17821898899 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train2_ver3.1.0/ja/warp-pdf-e02_0003.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 17933270898 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train2_ver3.1.0/ja/warp-pdf-e02_0004.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 17975848780 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train2_ver3.1.0/ja/warp-pdf-e02_0005.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 16210294041 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train2_ver3.1.0/ja/warp-pdf-e02_0006.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 17415053593 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train2_ver3.1.0/ja/warp-pdf-e02_0007.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 17013559839 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train2_ver3.1.0/ja/warp-pdf-e02_0008.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 17895268935 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train2_ver3.1.0/ja/warp-pdf-e02_0009.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 11953657161 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train2_ver3.1.0/ja/warp-pdf-e02_0010.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 17640899937 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train2_ver3.1.0/ja/warp-pdf-e02_0011.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 13044648744 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train2_ver3.1.0/ja/warp-pdf-e02_0012.jsonl_text_document"

# Ja:Wiki
DATA_PATH="${DATA_PATH} 1595058577 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/ja/wiki_0000.jsonl_text_document"

# En:Dolma/Gutenberg
DATA_PATH="${DATA_PATH} 5746482500 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-books_0000.jsonl_text_document"

# En:Dolma/C4
DATA_PATH="${DATA_PATH} 18069218639 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-c4_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 18068328906 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-c4_0001.jsonl_text_document"
DATA_PATH="${DATA_PATH} 18073617596 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-c4_0002.jsonl_text_document"
DATA_PATH="${DATA_PATH} 18074635974 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-c4_0003.jsonl_text_document"
DATA_PATH="${DATA_PATH} 18063702909 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-c4_0004.jsonl_text_document"
DATA_PATH="${DATA_PATH} 18061539178 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-c4_0005.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 18073951770 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-c4_0006.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 18066274128 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-c4_0007.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 18068030456 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-c4_0008.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 15822418911 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-c4_0009.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 13926234618 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-c4_0010.jsonl_text_document"

# En:Dolma/CC-head
DATA_PATH="${DATA_PATH} 16195213857 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16503711901 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0001.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17445841783 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0002.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17662192632 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0003.jsonl_text_document"
DATA_PATH="${DATA_PATH} 18015826387 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0004.jsonl_text_document"
DATA_PATH="${DATA_PATH} 17127262673 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0005.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16008607643 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0006.jsonl_text_document"
DATA_PATH="${DATA_PATH} 16496313125 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0007.jsonl_text_document"
DATA_PATH="${DATA_PATH} 14121145758 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0008.jsonl_text_document"
DATA_PATH="${DATA_PATH} 13155940780 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0009.jsonl_text_document"
DATA_PATH="${DATA_PATH} 14878948932 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0010.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 19333382266 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0011.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 19241916625 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0012.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 17064226185 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0013.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 15940474012 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0014.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 16353675513 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0015.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 17154919407 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0016.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 20536337909 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0017.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 18677317383 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0018.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 18862116539 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0019.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 20104519686 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0020.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 17675011221 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0021.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 16397058376 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0022.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 15746485184 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0023.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 20986051396 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0024.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 20213130925 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0025.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 18674485478 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0026.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 19585297471 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0027.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 19322466290 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0028.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 20074246647 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0029.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 20516355021 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0030.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 16201703492 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0031.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 17763740102 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0032.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 17527117171 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0033.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 17138277977 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0034.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 19183840930 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-cc-head_0035.jsonl_text_document"

# En:Dolma/PeS2o
DATA_PATH="${DATA_PATH} 8265395534 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-pes2o_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 21234790081 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-pes2o_0001.jsonl_text_document"
DATA_PATH="${DATA_PATH} 19619878090 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-pes2o_0002.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 16811496686 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-pes2o_0003.jsonl_text_document"

# En:Dolma/Reddit
DATA_PATH="${DATA_PATH} 18358437277 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-reddit_0000.jsonl_text_document"
DATA_PATH="${DATA_PATH} 18257582206 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-reddit_0001.jsonl_text_document"
DATA_PATH="${DATA_PATH} 18023663449 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-reddit_0002.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 16664454560 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-reddit_0003.jsonl_text_document"
# DATA_PATH="${DATA_PATH} 16318329112 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-reddit_0004.jsonl_text_document"

# En:Dolma/Wiki
DATA_PATH="${DATA_PATH} 4144873646 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/dolma-wiki_0000.jsonl_text_document"

# En:Wiki
DATA_PATH="${DATA_PATH} 5091489833 /gs/bs/tgh-NII-LLM/datasets/pretrain/llm-jp-corpus/v3.0.0/training_resharded_tokenize_ver2.2_code10K_en20K_ja30K.ver2.2/train/en/wiki_0000.jsonl_text_document"

# job name

JOB_NAME="TSUBAME4-Llama-2-1.56b-${NODE_TYPE}-${NUM_NODES}node-${NUM_GPUS}gpu-${SEQ_LENGTH}s-DP=${DATA_PARALLEL_SIZE}-TP=${TENSOR_PARALLEL_SIZE}-PP=${PIPELINE_PARALLEL_SIZE}-BS=${GLOBAL_BATCH_SIZE}-LR=${LR}-MINLR=${MIN_LR}-WARMUP=${LR_WARMUP_STEPS}-WD=${WEIGHT_DECAY}-GC=${GRAD_CLIP}-z-loss"

CHECKPOINT_ARGS="--load ${CHECKPOINT_SAVE_DIR}"

# run
mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -x LD_LIBRARY_PATH \
  -x PATH \
  -bind-to none \
  -x PATH \
  python pretrain_gpt.py \
  --tensor-model-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE} \
  --sequence-parallel \
  --use-distributed-optimizer \
  --num-layers ${NUM_LAYERS} \
  --hidden-size ${HIDDEN_SIZE} \
  --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
  --num-attention-heads ${NUM_HEADS} \
  --group-query-attention \
  --num-query-groups ${NUM_KEY_VALUE_HEADS} \
  --seq-length ${SEQ_LENGTH} \
  --max-position-embeddings ${SEQ_LENGTH} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_STEPS} \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  ${CHECKPOINT_ARGS} \
  --save ${CHECKPOINT_SAVE_DIR} \
  --data-path ${DATA_PATH} \
  --split 998,1,1 \
  --distributed-backend nccl \
  --init-method-std 0.02 \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-decay-style cosine \
  --lr-decay-iters ${LR_DECAY_ITERS} \
  --weight-decay ${WEIGHT_DECAY} \
  --clip-grad ${GRAD_CLIP} \
  --lr-warmup-iters ${LR_WARMUP_STEPS} \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --adam-eps 1e-8 \
  --log-interval 1 \
  --save-interval 500 \
  --eval-interval 500 \
  --eval-iters 10 \
  --bf16 \
  --untie-embeddings-and-output-weights \
  --no-position-embedding \
  --position-embedding-type rope \
  --rope-theta 10000.0 \
  --disable-bias-linear \
  --use-mcore-models \
  --normalization RMSNorm \
  --norm-epsilon 1e-5 \
  --no-masked-softmax-fusion \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --swiglu \
  --use-flash-attn \
  --recompute-activations \
  --recompute-granularity "selective" \
  --attention-softmax-in-fp32 \
  --transformer-impl "transformer_engine" \
  --use-mpi \
  --use-z-loss \
  --distributed-timeout-minutes 30 \
  --log-throughput \
  --wandb-name ${JOB_NAME} \
  --wandb-project "TSUBAME4-Llama-2-1.56b" \
  --wandb-entity "llm-jp"