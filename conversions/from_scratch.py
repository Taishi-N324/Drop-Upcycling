import argparse
import logging
import random

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initialize_and_save_model(config_path: str, save_path: str, seed: int) -> None:
    logger.info(f"Setting random seed to {seed}")
    set_seed(seed)

    logger.info(f"Loading config from {config_path}")
    config = AutoConfig.from_pretrained(config_path)

    logger.info("Initializing model from config")
    model = AutoModelForCausalLM.from_config(config)

    logger.info(f"Saving model and config to {save_path}")
    model.save_pretrained(save_path)
    config.save_pretrained(save_path)

    logger.info("Model and config have been saved successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Initialize and save a model from config"
    )

    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to the model configuration"
    )

    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Directory path where to save the model and config",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    initialize_and_save_model(
        config_path=args.config_path, save_path=args.save_path, seed=args.seed
    )


if __name__ == "__main__":
    main()
