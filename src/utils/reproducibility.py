# ensures identical results across machines and runs given same seed

import os
import random
import numpy as np
import torch
from src.utils.logger import get_logger

logger = get_logger("reproducibility")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # pytorch gpu — all GPUs
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # makes cudnn deterministic — slightly slower but reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False  # must be False for determinism
    # environment variable for hash randomization in Python
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Seed set to {seed} — fully reproducible mode enabled")