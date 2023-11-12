from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import argparse
import numpy as np
import os
import pandas as pd
from util import StandardScaler
import torch
import gc
from util import normalize_features



cvae_model = torch.load("./cvae_models/3convtest_wo_grus.pkl",map_location=torch.device('cpu'))
def get_augmented_features(hist_data):
    # num_samples = hist_data.shape[0]
    num_nodes = hist_data.shape[0]
    z = torch.randn([num_nodes, cvae_model.latent_size])
    augmented_features = cvae_model.inference(z,torch.tensor(data=hist_data, dtype=torch.float32)).detach()
    augmented_features = np.expand_dims(augmented_features.T.numpy(),axis=-1)
    return augmented_features

def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, x_hist_offsets, add_time_in_day=False, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets
    :param y_offsets
    :param add_time_in_day
    :param add_day_in_week:
    :param scaler:
    :return
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    hist_len=len(x_hist_offsets)
    last_train_index=int((num_samples-hist_len+1)*0.7+(hist_len-1))
    scaler=StandardScaler(mean=df[:last_train_index,:].mean(),
                                     std=df[:last_train_index,:].std())
    data=scaler.transform(df)
    data = np.expand_dims(data, axis=-1).astype(np.float32)  # [34272,207]->[34272,207,1]
    df = np.expand_dims(df, axis=-1).astype(np.float32)  # [34272,207]->[34272,207,1]
    x_aug, x, y = [], [], []
    min_t = abs(min(x_hist_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        print("t:{:d}".format(t))
        aug_f=get_augmented_features(np.squeeze(data[t + x_hist_offsets, ...], axis=-1).T)
        x_aug.append(aug_f[-12:,:,:])
        x.append(df[t + x_offsets, ...])
        y.append(df[t + y_offsets, ...])
    x_aug = np.stack(x_aug, axis=0)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    print(f"x_aug:{x_aug.shape}, x:{x.shape}, y:{y.shape}")
    return x_aug, x, y


def generate_train_val_test(args):
    seq_length_x, seq_length_y, seq_length_x_hist = args.seq_length_x, args.seq_length_y, args.seq_length_x_hist

    df = pd.read_hdf(args.traffic_df_filename).to_numpy()  # data
    # 0 is the latest observed sample.
    x_hist_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x_hist - 1), 1, 1),)))
    print(x_hist_offsets)
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    print(x_offsets)
    # Predict the next one hour
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x_aug, x, y = generate_graph_seq2seq_io_data(
        df[:17186,:],
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        x_hist_offsets=x_hist_offsets,
        add_time_in_day=False,
        add_day_in_week=False,
    )
    x_aug1, x1, y1 = generate_graph_seq2seq_io_data(
        df[15171:, :],
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        x_hist_offsets=x_hist_offsets,
        add_time_in_day=False,
        add_day_in_week=False,
    )
    x=np.concatenate((x, x1), axis=0)
    x_aug=np.concatenate((x_aug, x_aug1), axis=0)
    y=np.concatenate((y, y1), axis=0)
    del x1,x_aug1,y1
    gc.collect()

    print("x_hist shape: ", x_aug.shape, "x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x_aug.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_aug_train, x_train, y_train = x_aug[:num_train], x[:num_train], y[:num_train]
    print(f"x_train shape:{x_train.shape}")
    # val
    x_aug_val, x_val, y_val = (
        x_aug[num_train: num_train + num_val],
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_aug_test, x_test, y_test = x_aug[-num_test:], x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x_aug, _x, _y = locals()["x_aug_" + cat], locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x_hist: ", _x_aug.shape, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x_aug=_x_aug,
            x=_x,
            y=_y,
            x_hist_offsets=x_hist_offsets.reshape(list(x_hist_offsets.shape) + [1]),
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/temporal_data/metr-la-12-with_hist_wo_gru",
                        help="Output directory.")
    parser.add_argument("--traffic_df_filename", type=str, default="data/original_data/metr-la.h5",
                        help="Raw traffic readings.", )
    parser.add_argument("--seq_length_x", type=int, default=12, help="Sequence Length.", )
    parser.add_argument("--seq_length_y", type=int, default=12, help="Sequence Length.", )
    parser.add_argument("--seq_length_x_hist", type=int, default=2016, help="Sequence Length.", )
    parser.add_argument("--y_start", type=int, default=1, help="Y pred start", )
    parser.add_argument("--dow", action='store_true', )

    args = parser.parse_args()
    generate_train_val_test(args)


