import os
from argparse import ArgumentParser
from itertools import chain, product

import numpy as np
import torch
import wandb

from scipy.io.wavfile import read

import waveglow
import text
import utils

from modules import FastSpeech
from configs import model_config, train_config


def synthesis(model, text, duration_alpha=1.0, energy_alpha=1.0, pitch_alpha=1.0):
    text = np.array(text)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to(train_config.device)
    src_pos = torch.from_numpy(src_pos).long().to(train_config.device)
    with torch.no_grad():
        mel = model.forward(
            sequence, src_pos,
            length_alpha=duration_alpha, energy_alpha=energy_alpha, pitch_alpha=pitch_alpha
        )
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)

        
def parse_args():
    parser = ArgumentParser()
    variance_group = parser.add_argument_group(title='Variance')
    variance_group.add_argument('--duration-alpha', default=1.0, required=False, type=float, nargs='+')
    variance_group.add_argument('--energy-alpha', default=1.0, required=False, type=float, nargs='+')
    variance_group.add_argument('--pitch-alpha', default=1.0, required=False, type=float, nargs='+')
    text_group = parser.add_argument_group(title='Text')
    text_group.add_argument('--text', help='text to synthesize', required=False, type=str, nargs='*')
    text_group.add_argument('--text-file', help='path to file with texts to synthesize (each on own line)', required=False, type=str)
    output_group = parser.add_argument_group(title='Output')
    output_group.add_argument('--output-directory', default='results', required=False, type=str)
    output_group.add_argument('--wandb-project', required=False, type=str)
    parser.add_argument('--checkpoint-path', required=True)
    args = parser.parse_args()
    assert all(x > 0 for x in chain(args.duration_alpha, args.energy_alpha, args.pitch_alpha))
    assert (not args.text) ^ (args.text_file is None)
    return args
        
if __name__ == '__main__':
    args = parse_args()
    
    # texts
    texts = args.text
    if args.text_file is not None:
        for line in open(args.text_file):
            texts.append(line.rstrip())
    data_list = list(text.text_to_sequence(test, train_config.text_cleaners) for test in texts)
    
    # spectrogram model
    model = FastSpeech(model_config)
    model = model.to(train_config.device)
    loaded_state = torch.load(args.checkpoint_path, map_location=train_config.device)
    try:
        model.load_state_dict(loaded_state['model'])
    except KeyError:
        # checkpoint is just a model
        model.load_state_dict(loaded_state)
    model = model.eval()

    # waveglow
    WaveGlow = utils.get_WaveGlow()
    WaveGlow = WaveGlow.cuda()

    # synthesis
    if args.wandb_project:
        wandb.init(project=args.wandb_project)
    os.makedirs(args.output_directory, exist_ok=True)
    for d, e, p in product(args.duration_alpha, args.energy_alpha, args.pitch_alpha):
        for index, text in enumerate(data_list):
            filename = f'text{index + 1}_duration={d}_energy={e}_pitch={p}_waveglow.wav'
            filepath = os.path.join(args.output_directory, filename)

            _, mel_cuda = synthesis(model, text, d, e, p)
            waveglow.inference.inference(mel_cuda, WaveGlow, filepath)

            if args.wandb_project:
                wandb.log({filename: wandb.Audio(filepath, caption=texts[index])})
    