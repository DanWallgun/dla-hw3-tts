import os
from argparse import ArgumentParser

import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from audio import tools
from configs import train_config


def calculate_energy_pitch(n_jobs=32):
    os.makedirs(train_config.energy_path)
    os.makedirs(train_config.pitch_path)

    def proc(index, wav_name):
        _, energy, pitch = tools.get_mel_v2_energy_pitch(
            os.path.join(train_config.wav_path, wav_name)
        )
        np.save(
            os.path.join(train_config.energy_path, 'ljspeech-energy-%05d.npy' % index),
            energy, allow_pickle=False
        )
        np.save(
            os.path.join(train_config.pitch_path, 'ljspeech-pitch-%05d.npy' % index),
            pitch, allow_pickle=False
        )

    all_wav_names = sorted(os.listdir(train_config.wav_path))
    _ = Parallel(n_jobs=n_jobs)(delayed(proc)(idx, wav) for idx, wav in enumerate(all_wav_names))

    
def calculate_min_max():
    n_samples = len(os.listdir(train_config.energy_path))
    min_e, max_e, min_p, max_p = np.inf, -np.inf, np.inf, -np.inf
    for index in tqdm(range(n_samples)):
        energy_gt_name = os.path.join(train_config.energy_path, "ljspeech-energy-%05d.npy" % (index))
        energy_gt_target = np.load(energy_gt_name)
        min_e = min(min_e, energy_gt_target.min())
        max_e = max(max_e, energy_gt_target.max())

        pitch_gt_name = os.path.join(train_config.pitch_path, "ljspeech-pitch-%05d.npy" % (index))
        pitch_gt_target = np.load(pitch_gt_name)
        min_p = min(min_p, pitch_gt_target.min())
        max_p = max(max_p, pitch_gt_target.max())
    print(f'{(min_e, max_e, min_p, max_p)=}')


def normalize():
    from sklearn.preprocessing import StandardScaler

    e_scaler, p_scaler = StandardScaler(), StandardScaler()
    n_samples = len(os.listdir(train_config.energy_path))

    for index in tqdm(range(n_samples)):
        energy_gt_name = os.path.join(train_config.energy_path, "ljspeech-energy-%05d.npy" % (index))
        energy_gt_target = np.load(energy_gt_name)
        e_scaler.partial_fit(energy_gt_target.reshape((-1, 1)))

        pitch_gt_name = os.path.join(train_config.pitch_path, "ljspeech-pitch-%05d.npy" % (index))
        pitch_gt_target = np.load(pitch_gt_name)
        p_scaler.partial_fit(pitch_gt_target.reshape((-1, 1)))

    for index in tqdm(range(n_samples)):
        energy_gt_name = os.path.join(train_config.energy_path, "ljspeech-energy-%05d.npy" % (index))
        energy_gt_target = np.load(energy_gt_name)
        energy_gt_target = (energy_gt_target - e_scaler.mean_[0]) / e_scaler.scale_[0]
        np.save(energy_gt_name, energy_gt_target)

        pitch_gt_name = os.path.join(train_config.pitch_path, "ljspeech-pitch-%05d.npy" % (index))
        pitch_gt_target = np.load(pitch_gt_name)
        pitch_gt_target = (pitch_gt_target - p_scaler.mean_[0]) / p_scaler.scale_[0]
        np.save(pitch_gt_name, pitch_gt_target)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('function', choices=['calculate_energy_pitch', 'calculate_min_max', 'normalize'])
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    globals()[args.function]()