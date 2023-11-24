#!/bin/python3

import pandas as pd
import os
import glob
import argparse

def read_csvs(dirpath):
    # files = glob.glob(os.path.join(dirpath, 'log_stat', '*.csv'))
    files = glob.glob(os.path.join(dirpath, '*.csv'))
    df = pd.concat([pd.read_csv(file) for file in files]).reset_index(drop=True).dropna()
    return df


def check_among_tp(df, target_name, target_metrics):
    tmp_df = df[['step', 'dp_rank', 'pp_rank', 'name'] + target_metrics]
    tmp_df = tmp_df[tmp_df['name'].isin(target_name)]
    tmp_df_min = tmp_df.groupby(['dp_rank', 'pp_rank', 'name', 'step']).min()
    tmp_df_max = tmp_df.groupby(['dp_rank', 'pp_rank', 'name', 'step']).max()
    pd.testing.assert_frame_equal(tmp_df_min, tmp_df_max)


parser = argparse.ArgumentParser()
parser.add_argument('target_dir', type=str)
parser.add_argument('--compare_dir', type=str, default=None)

args = parser.parse_args()

tgt_df = read_csvs(args.target_dir)
if args.compare_dir is not None:
    cmp_df = read_csvs(args.compare_dir)

target_metrics = ['sum', 'l1_norm', 'l2_norm']

if args.compare_dir is None:
    target_names = ['input_ids', 'position_ids', 'attention_mask', 'cpu_rng_state', 'lm_output']
    check_among_tp(tgt_df, target_names, target_metrics)
else:
    raise "not implemented"

print("OK")
