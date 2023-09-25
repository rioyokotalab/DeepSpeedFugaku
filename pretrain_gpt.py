from megatron.training import pretrain

if __name__ == "__main__":
    pretrain(None, None, None,
                args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
                data_post_process=None)
