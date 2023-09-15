from megatron.training import pretrain

if __name__ == "__main__":
    # from torch.profiler import profile, ProfilerActivity
    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    pretrain(None, None, None,
                args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
                data_post_process=None)
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
