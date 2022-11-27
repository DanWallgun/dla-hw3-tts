import os

import torch
from torch import nn
from torch.optim.lr_scheduler  import OneCycleLR
from tqdm.auto import tqdm

from wandb_writer import WanDBWriter
from dataset import get_training_loader
from loss import FastSpeechLoss
from modules import FastSpeech
from configs import model_config, train_config


def main():
    model = FastSpeech(model_config)
    model = model.to(train_config.device)

    training_loader = get_training_loader()

    fastspeech_loss = FastSpeechLoss()
    current_step = 0

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9)

    scheduler = OneCycleLR(optimizer, **{
        "steps_per_epoch": len(training_loader) * train_config.batch_expand_size,
        "epochs": train_config.epochs,
        "anneal_strategy": "cos",
        "max_lr": train_config.learning_rate,
        "pct_start": 0.1
    })

    logger = WanDBWriter(train_config)


    tqdm_bar = tqdm(total=train_config.epochs * len(training_loader) * train_config.batch_expand_size - current_step)

    model.train()
    for epoch in range(train_config.epochs):
        for i, batchs in enumerate(training_loader):
            # real batch start here
            for j, db in enumerate(batchs):
                current_step += 1
                tqdm_bar.update(1)
                
                logger.set_step(current_step)

                # Get Data
                character = db["text"].long().to(train_config.device)
                mel_target = db["mel_target"].float().to(train_config.device)
                energy_target = db["energy_target"].float().to(train_config.device)
                pitch_target = db["pitch_target"].float().to(train_config.device)
                duration_target = db["duration"].int().to(train_config.device)
                mel_pos = db["mel_pos"].long().to(train_config.device)
                src_pos = db["src_pos"].long().to(train_config.device)
                max_mel_len = db["mel_max_len"]

                # Forward
                mel_predicted, energy_predicted, pitch_predicted, log_duration_predicted = \
                    model(
                        character, src_pos,
                        mel_pos=mel_pos, mel_max_length=max_mel_len,
                        energy_target=energy_target, pitch_target=pitch_target,
                        length_target=duration_target,
                    )

                # Calc Loss
                total_loss, mel_loss, energy_loss, pitch_loss, duration_loss = \
                    fastspeech_loss(
                        mel_predicted, energy_predicted,
                        pitch_predicted, log_duration_predicted,
                        mel_target, energy_target, pitch_target, duration_target
                    )

                # Logger
                t_l = total_loss.detach().cpu().numpy()
                m_l = mel_loss.detach().cpu().numpy()
                e_l = energy_loss.detach().cpu().numpy()
                p_l = pitch_loss.detach().cpu().numpy()
                d_l = duration_loss.detach().cpu().numpy()

                logger.add_scalar("duration_loss", d_l)
                logger.add_scalar("mel_loss", m_l)
                logger.add_scalar("energy_loss", e_l)
                logger.add_scalar("pitch_loss", p_l)
                logger.add_scalar("total_loss", t_l)

                # Backward
                total_loss.backward()

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(
                    model.parameters(), train_config.grad_clip_thresh)
                
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                if current_step % train_config.frequent_save_current_model == 0:
                    torch.save(
                        model.state_dict(),
                        os.path.join(train_config.checkpoint_path, 'current_model-state_dict-N.pth')
                    )

                if current_step % train_config.save_step == 0:
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                    )}, os.path.join(train_config.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                    print("save model at step %d ..." % current_step)


if __name__ == '__main__':
    main()