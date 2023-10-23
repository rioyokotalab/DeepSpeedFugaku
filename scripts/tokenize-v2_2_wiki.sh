#!/bin/bash
#YBATCH -r epyc-7543_2
#SBATCH --job-name=tokenizer
#SBATCH --time=7-00:00:00
#SBATCH --output outputs/%j.out
#SBATCH --error errors/%j.err
set -euxo pipefail

. /etc/profile.d/modules.sh
module load cuda/11.7
module load cudnn/cuda-11.x/8.9.0
module load nccl/cuda-11.7/2.14.3
module load openmpi/4.0.5

CODE_VOCAB_SIZE=20
EN_VOCAB_SIZE=40
JA_VOCAB_SIZE=60

# Set the output directory:
export OUTDIR=/mnt/nfs/Users/tn/binarized/fugaku/v2_2-code${CODE_VOCAB_SIZE}k_en${EN_VOCAB_SIZE}k_ja${JA_VOCAB_SIZE}k
mkdir -p $OUTDIR
export MODELDIR=/home/tn/DeepSpeedFugaku/llm-jp-tokenizer/models/ver2.2/code${CODE_VOCAB_SIZE}K_en${EN_VOCAB_SIZE}K_ja${JA_VOCAB_SIZE}K.ver2.2.model
export INPUTDIR="/mnt/nfs/Users/tn/datasets/llm-jp-corpus/v1.0.2-merge/ja_wiki"

python /home/tn/DeepSpeedFugaku/tools/preprocess_data.py \
    --input $INPUTDIR/ja_wiki_merged_train.jsonl \
    --output-prefix $OUTDIR/ja_wiki_merged_train_0 \
    --vocab-file $MODELDIR \
    --dataset-impl mmap \
    --tokenizer-type JapaneseSentencePiece \
    --workers 128 \
    --append-eod

export INPUTDIR="/mnt/nfs/Users/tn/datasets/llm-jp-corpus/v1.0.2-merge/en_wiki"

python /home/tn/DeepSpeedFugaku/tools/preprocess_data.py \
    --input $INPUTDIR/en_wiki_merged_train.jsonl \
    --output-prefix $OUTDIR/en_wiki_merged_train_0 \
    --vocab-file $MODELDIR \
    --dataset-impl mmap \
    --tokenizer-type JapaneseSentencePiece \
    --workers 128 \
    --append-eod

