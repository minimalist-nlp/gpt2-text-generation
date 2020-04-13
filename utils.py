# -*- coding: utf-8 -*-
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F


from pytorch_lightning.logging import TestTubeLogger


def setup_testube_logger() -> TestTubeLogger:
    """ Function that sets the TestTubeLogger to be used. """
    try:
        job_id = os.environ["SLURM_JOB_ID"]
    except Exception:
        job_id = None

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y--%H-%M-%S")
    return TestTubeLogger(
        save_dir="experiments/",
        version=job_id if job_id else dt_string,
        name="lightning_logs",
    )


def top_k_top_p_filtering(
    logits, top_k, top_p, temperature, filter_value=-float("Inf")
):
    # Hugging Face script to apply top k and nucleus sampling
    logits = logits / temperature

    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits


def load_weights_lm_head(model, original_model):
    pretrained_dict = original_model.state_dict()
    pretrained_dict["lm_head.weight"] = torch.cat(
        (
            pretrained_dict["lm_head.weight"],
            torch.rand(model.gpt2.vocab_size - model.tokenizer.vocab_size, 768),
        ),
        0,
    )
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
