import argparse
import logging
import random
import re

import numpy as np
import torch
from tqdm import tqdm
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


def initialize_gate_weights(size, method, std=0.02):
    if method == "torch_rand":
        return torch.rand(size)
    elif method == "torch_rand_mean0":
        weights = torch.rand(size)
        weights_mean = weights.mean()
        return weights - weights_mean
    elif method == "torch_normal_002":
        return torch.normal(mean=0, std=0.02, size=size)
    elif method == "torch_normal_028":
        return torch.normal(mean=0, std=0.2886751345948129, size=size)
    elif method == "torch_rand_002":
        weights = torch.rand(size)
        weights_mean = weights.mean()
        return (weights - weights_mean) * 0.02 * (12**0.5)
    else:
        raise ValueError(f"Unknown initialization method: {method}")


def replace_model_parameters(
    source_model_path,
    target_config_path,
    output_path,
    num_experts,
    num_layers,
    seed,
    init_method,
):
    set_seed(seed)

    source_model = AutoModelForCausalLM.from_pretrained(
        source_model_path, torch_dtype=torch.bfloat16
    )
    target_config = AutoConfig.from_pretrained(target_config_path)
    target_model = AutoModelForCausalLM.from_config(
        target_config, torch_dtype=torch.bfloat16
    )

    exclude_pattern = r"model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight"
    exclude_layers = set()
    for name in target_model.state_dict().keys():
        if re.match(exclude_pattern, name):
            exclude_layers.add(name)

    base_src = "model.layers.{}.block_sparse_moe.experts.{}"
    base_tgt = "model.layers.{}.mlp"
    replace_mapping = {
        f"{base_src}.w1.weight": f"{base_tgt}.gate_proj.weight",
        f"{base_src}.w2.weight": f"{base_tgt}.down_proj.weight",
        f"{base_src}.w3.weight": f"{base_tgt}.up_proj.weight",
    }

    source_state_dict = source_model.state_dict()
    target_state_dict = target_model.state_dict()

    for name, param in tqdm(target_state_dict.items(), desc="Replacing parameters"):
        if name not in exclude_layers and name in source_state_dict:
            target_state_dict[name] = source_state_dict[name]
            logger.info(f"Parameter {name} replaced")

    for layer_idx in tqdm(range(num_layers), desc="Initializing gate weights"):
        gate_weight_name = f"model.layers.{layer_idx}.block_sparse_moe.gate.weight"
        if gate_weight_name in target_state_dict:
            target_state_dict[gate_weight_name] = initialize_gate_weights(
                target_state_dict[gate_weight_name].size(), init_method
            )
            logger.info(
                f"Gate weight {gate_weight_name} initialized with {init_method}"
            )

    for layer_idx in tqdm(range(num_layers), desc="Replacing FFN layers"):
        for expert_idx in range(num_experts):
            for target_pattern, source_pattern in replace_mapping.items():
                target_name = target_pattern.format(layer_idx, expert_idx)
                source_name = source_pattern.format(layer_idx)
                if (
                    target_name in target_state_dict
                    and source_name in source_state_dict
                ):
                    target_state_dict[target_name] = source_state_dict[source_name]
                    logger.info(f"FFN layer {target_name} replaced with {source_name}")

    target_model.load_state_dict(target_state_dict)
    target_model.save_pretrained(output_path, torch_dtype=torch.bfloat16)
    logger.info(f"Modified model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Replace model parameters")
    parser.add_argument(
        "--source_model_path", type=str, required=True, help="Path to the source model"
    )
    parser.add_argument(
        "--target_config_path",
        type=str,
        required=True,
        help="Path to the target model config",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the modified model"
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        required=True,
        help="Number of experts in the MoE model",
    )
    parser.add_argument(
        "--num_layers", type=int, required=True, help="Number of layers in the model"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--init_method",
        type=str,
        choices=[
            "torch_rand",
            "torch_rand_mean0",
            "torch_normal_002",
            "torch_normal_028",
            "torch_rand_002",
        ],
        default="torch_rand",
        help="Method for initializing gate weights",
    )
    args = parser.parse_args()

    replace_model_parameters(
        args.source_model_path,
        args.target_config_path,
        args.output_path,
        args.num_experts,
        args.num_layers,
        args.seed,
        args.init_method,
    )


if __name__ == "__main__":
    main()
