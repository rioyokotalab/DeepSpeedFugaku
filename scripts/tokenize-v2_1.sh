#!/bin/bash
#YBATCH -r threadripper-3960x_8
#SBATCH --job-name=en_pile
#SBATCH --time=2-00:00:00
#SBATCH --output outputs/%j.out
#SBATCH --error errors/%j.err
. /etc/profile.d/modules.sh
module load cuda/11.7
module load cudnn/cuda-11.x/8.9.0
module load nccl/cuda-11.7/2.14.3
module load openmpi/4.0.5

CODE_VOCAB_SIZE=30
EN_VOCAB_SIZE=80
JA_VOCAB_SIZE=120

# CODE_VOCAB_SIZE=20
# EN_VOCAB_SIZE=60
# JA_VOCAB_SIZE=100

# CODE_VOCAB_SIZE=20
# EN_VOCAB_SIZE=40
# JA_VOCAB_SIZE=80

# Set the output directory:
export OUTDIR=/mnt/nfs/Users/tn/fugaku/datasets/wikipedia/binarized/v2_1-code${CODE_VOCAB_SIZE}k_en${EN_VOCAB_SIZE}k_ja${JA_VOCAB_SIZE}k
mkdir -p $OUTDIR
export MODELDIR=/home/tn/DeepSpeedFugaku/tokenizer/models/ver2/code${CODE_VOCAB_SIZE}k_en${EN_VOCAB_SIZE}k_ja${JA_VOCAB_SIZE}k.ver2.1.model


python /home/tn/DeepSpeedFugaku/tools/preprocess_data.py \
            --input /mnt/nfs/Users/tn/llm-jp-corpus/v1.0.1/filter/fugaku/merge/en_pile/en_pile_merge.json \
            --output-prefix $OUTDIR/en_pile \
            --vocab-file $MODELDIR \
            --dataset-impl mmap \
            --tokenizer-type JapaneseSentencePiece \
            --workers 128 \
            --append-eod

# python /home/tn/DeepSpeedFugaku/tools/preprocess_data.py \
#             --input /mnt/nfs/Users/tn/llm-jp-corpus/v1.0.1/filter/fugaku/merge/ja_cc/ja_cc_merge.json \
#             --output-prefix $OUTDIR/ja_cc \
#             --vocab-file $MODELDIR \
#             --dataset-impl mmap \
#             --tokenizer-type JapaneseSentencePiece \
#             --workers 128 \
#             --append-eod


# python /home/tn/DeepSpeedFugaku/tools/preprocess_data.py \
#             --input /mnt/nfs/Users/tn/llm-jp-corpus/v1.0.1/filter/fugaku/merge/ja_wiki/ja_wiki_merged.json \
#             --output-prefix $OUTDIR/ja_wiki \
#             --vocab-file $MODELDIR \
#             --dataset-impl mmap \
#             --tokenizer-type JapaneseSentencePiece \
#             --workers 128 \
#             --append-eod

# python /home/tn/DeepSpeedFugaku/tools/preprocess_data.py \
#             --input /mnt/nfs/Users/tn/llm-jp-corpus/v1.0.1/filter/fugaku/merge/en_wiki/en_wiki_merge.json \
#             --output-prefix $OUTDIR/en_wiki \
#             --vocab-file $MODELDIR \
#             --dataset-impl mmap \
#             --tokenizer-type JapaneseSentencePiece \
#             --workers 128 \
#             --append-eod

# python /home/tn/DeepSpeedFugaku/tools/preprocess_data.py \
#             --input /mnt/nfs/Users/tn/llm-jp-corpus/v1.0.1/filter/fugaku/merge/code_stack/code_stack_merge.json \
#             --output-prefix $OUTDIR/code_stack \
#             --vocab-file $MODELDIR \
#             --dataset-impl mmap \
#             --tokenizer-type JapaneseSentencePiece \
#             --workers 128 \
#             --append-eod

