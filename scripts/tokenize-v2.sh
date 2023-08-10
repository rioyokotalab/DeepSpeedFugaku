#!/bin/bash
#YBATCH -r a100_1
#SBATCH --job-name=tokenize
#SBATCH --time=2-00:00:00
#SBATCH --output outputs/%j.out
#SBATCH --error errors/%j.err
. /etc/profile.d/modules.sh
module load cuda/11.7
module load cudnn/cuda-11.x/8.9.0
module load nccl/cuda-11.7/2.14.3
module load openmpi/4.0.5

CODE_VOCAB_SIZE=20
EN_VOCAB_SIZE=40
JA_VOCAB_SIZE=80

# source .env/bin/activate

# Set the output directory:
export OUTDIR=datasets/wikipedia/binarized/v2-code${CODE_VOCAB_SIZE}k_en${EN_VOCAB_SIZE}k_ja${JA_VOCAB_SIZE}k
mkdir -p $OUTDIR
export MODELDIR=tokenizer/models/ver2/code${CODE_VOCAB_SIZE}k_en${EN_VOCAB_SIZE}k_ja${JA_VOCAB_SIZE}k.ver2.model

# # Tokenize and binarize Japanese
# python /home/tn/DeepSpeedFugaku/tools/preprocess_data.py \
#   --input dataset/wikipedia/merged/ja/ja_merged.json \
#   --output-prefix $OUTDIR/ja_wiki \
#   --vocab-file $MODELDIR \
#   --dataset-impl mmap \
#   --tokenizer-type JapaneseSentencePiece \
#   --workers 64 \
#   --append-eod

# folder='/mnt/nfs/Users/tn/llm-jp-corpus/v1.0.1/filter/filter/ja_wiki'
# files=$(find $folder -type f -printf "%p,")
# files=${files%,}  # Remove the trailing comma
# python /home/tn/DeepSpeedFugaku/tools/preprocess_data.py \
#             --input /mnt/nfs/Users/tn/llm-jp-corpus/v1.0.1/filter/fugaku/merge/ja_wiki/ja_wiki_merged.json \
#             --output-prefix $OUTDIR/ja_wiki \
#             --vocab-file /home/tn/gpt-neox/tokenizer/cc100_ja40K_en10K.symbolRemoved.vocab.reestimated.model \
#             --dataset-impl mmap \
#             --tokenizer-type JapaneseSentencePiece \
#             --workers 128 \
#             --append-eod

python /home/tn/DeepSpeedFugaku/tools/preprocess_data.py \
            --input /mnt/nfs/Users/tn/llm-jp-corpus/v1.0.1/filter/fugaku/merge/en_wiki/en_wiki_merged.json \
            --output-prefix $OUTDIR/en_wiki \
            --vocab-file /home/tn/gpt-neox/tokenizer/cc100_ja40K_en10K.symbolRemoved.vocab.reestimated.model \
            --dataset-impl mmap \
            --tokenizer-type JapaneseSentencePiece \
            --workers 128 \
            --append-eod




# # Tokenize and binarize English
# python /home/tn/DeepSpeedFugaku/tools/preprocess_data.py \
#   --input dataset/wikipedia/merged/en/en_merged.json \
#   --output-prefix $OUTDIR/en_wiki \
#   --vocab-file $MODELDIR \
#   --dataset-impl mmap \
#   --tokenizer-type JapaneseSentencePiece \
#   --workers 64 \
#   --append-eod
