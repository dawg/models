import os
import numpy as np
import torch 

from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
from vusic.transcription.training.evaluate import evaluate
from vusic.utils.transcription_settings import constants, training_settings
from vusic.utils.transcription_dataset import TranscriptionDataset
from vusic.transcription.modules.onset_frame_model import OnsetFrameModel
from vusic.utils.transcription_utils import summary, cycle
from vusic.transcription.modules.mel import melspectrogram


def train():
    model_dir = training_settings["model_dir"]
    device = constants["default_device"]
    iterations = training_settings["iterations"]
    resume_iteration = training_settings["resume_iteration"]
    checkpoint_interval = training_settings["checkpoint_interval"]
    batch_size = training_settings["batch_size"]
    sequence_length = training_settings["sequence_length"]
    model_complexity = training_settings["model_complexity"]
    learning_rate = training_settings["learning_rate"]
    learning_rate_decay_steps = training_settings["learning_rate_decay_steps"]
    learning_rate_decay_rate = training_settings["learning_rate_decay_rate"]
    clip_gradient_norm = training_settings["clip_gradient_norm"]

    os.makedirs(model_dir, exist_ok=True)
    writer = SummaryWriter(model_dir)

    dataset = TranscriptionDataset(sequence_length=sequence_length)
    loader = DataLoader(dataset, batch_size, shuffle=True)

    if resume_iteration is None:
        model = OnsetFrameModel(
            constants["n_mels"],
            constants["max_midi"] - constants["min_midi"] + 1,
            model_complexity,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(model_dir, f"model-{resume_iteration}.pt")
        model = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(
            torch.load(os.path.join(model_dir, "last-optimizer-state.pt"))
        )

    summary(model)
    scheduler = StepLR(
        optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate
    )

    loop = tqdm(range(resume_iteration + 1, iterations + 1))
    for i, batch in zip(loop, cycle(loader)):
        scheduler.step()

        mel = melspectrogram(
            batch["audio"].reshape(-1, batch["audio"].shape[-1])[:, :-1]
        ).transpose(-1, -2)

        predictions, losses = model.run_on_batch(batch, mel)

        loss = sum(losses.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if clip_gradient_norm:
            for parameter in model.parameters():
                clip_grad_norm_([parameter], clip_gradient_norm)

        for key, value in {"loss": loss, **losses}.items():
            writer.add_scalar(key, value.item(), global_step=i)

        if i % checkpoint_interval == 0:
            torch.save(model, os.path.join(model_dir, f"model-{i}.pt"))
            torch.save(
                optimizer.state_dict(),
                os.path.join(model_dir, "last-optimizer-state.pt"),
            )


if __name__ == "__main__":
    train()
