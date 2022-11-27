import torch
import torch.nn as nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(
        self,
        mel_predicted, energy_predicted, pitch_predicted, log_duration_predicted,
        mel_target, energy_target, pitch_target, duration_predictor_target
    ):
        # masks?
        log_duration_predictor_target = torch.log(duration_predictor_target.float() + 1)

        mel_loss = self.l1_loss(mel_predicted, mel_target)
        energy_loss = self.mse_loss(energy_predicted, energy_target)
        pitch_loss = self.mse_loss(pitch_predicted, pitch_target)
        duration_loss = self.mse_loss(log_duration_predicted, log_duration_predictor_target)
        
        total_loss = mel_loss + energy_loss + pitch_loss + duration_loss
        
        return (
            total_loss,
            mel_loss,
            energy_loss,
            pitch_loss,
            duration_loss,
        )