import argparse
import pickle as pkl
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
token = None

start_time = time.time()


# Adapted from transformers.models.mixtral.modeling_mixtral.load_balancing_loss_func
# Fixed aggregating over all layers
def load_balancing_loss_func(
    gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2
) -> float:
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.stack(
            [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=1
        )

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    # Compute the percentage of tokens routed to each experts
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    # Compute the average probability of routing to these experts
    router_prob_per_expert = torch.mean(routing_weights, dim=0)

    overall_loss = torch.sum(
        tokens_per_expert * router_prob_per_expert.unsqueeze(-2)
    ) / len(gate_logits)
    return overall_loss * num_experts


def load_analysis_data(tokenizer, domain, bs):
    np.random.seed(2024)
    tokens = []

    data_path = f"routing_output/text/{domain}_texts.txt"
    with open(data_path) as f:
        text = f.read()
        tokens = tokenizer(text, truncation=False)["input_ids"]
        while len(tokens) >= bs:
            yield tokens[:bs]
            tokens = tokens[bs:]


def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, device_map="auto", token=token
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=token
    )
    return model, tokenizer


def print_expert_percentage(exp_counts):
    total = sum(exp_counts.values())
    for eid, ecount in exp_counts.most_common():
        print(f"Expert {eid}: {ecount/total*100:.2f}")


def run_analysis(domain, model_name=None):
    layer_counters = defaultdict(Counter)
    crosslayer_counters = defaultdict(Counter)
    eid2token = defaultdict(lambda: defaultdict(Counter))

    total_token_count = 0

    aux_losses = []

    # for each expert, what are some commen tokens that are assigned to it?
    # write a code to count that and print it out: {expert_id: Counter({token1: 32, token2: 23})...}
    for i, input_ids in tqdm(
        enumerate(load_analysis_data(tokenizer, domain=domain, bs=length))
    ):
        input_ids = torch.LongTensor(input_ids).reshape(1, -1).to(DEVICE)
        out = model(input_ids=input_ids, output_router_logits=True)

        aux_loss = load_balancing_loss_func(
            out["router_logits"], model.num_experts, model.num_experts_per_tok,
        )
        aux_losses.append(aux_loss.cpu().item())

        # input id shapes: 2048 seqlen
        input_ids = input_ids[0].detach().cpu().numpy().tolist()
        total_token_count += len(input_ids)

        router_logits = [l.detach().cpu().numpy() for l in out["router_logits"]]
        # top2
        exp_ids = np.stack(
            [np.argsort(-logits, -1)[:, :2].tolist() for logits in router_logits], -1
        )

        num_layers = exp_ids.shape[2]

        for id, token in enumerate(input_ids):
            for layer in range(num_layers):
                experts = exp_ids[id, :, layer]
                for e in experts:
                    eid2token[layer][e][token] += 1

        for layer in range(num_layers):
            exp_counts = Counter(exp_ids[:, :, layer].flatten())
            layer_counters[layer].update(exp_counts)

        for layer_i in range(num_layers):
            for layer_j in range(num_layers):
                if layer_i != layer_j:
                    exps_counts = Counter(
                        zip(
                            exp_ids[:, :, layer_i].flatten(),
                            exp_ids[:, :, layer_j].flatten(),
                        )
                    )
                    crosslayer_counters[(layer_i, layer_j)].update(exps_counts)

        if total_token_count > 204800:
            break

    print(f"Average aux loss: {np.mean(aux_losses)}")

    return layer_counters, crosslayer_counters, eid2token


name2finaldata = {
    "c4": "c4",
    "ja_wikipedia": "ja_wikipedia",
    "en_wikipedia": "en_wikipedia",
    "ja_mc4": "ja_mc4",
    "code_stack": "code_stack",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run analysis on specified model")
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model to analyze"
    )
    args = parser.parse_args()

    model_name = args.model_name
    print(f"Analyzing model: {model_name}")
    model, tokenizer = load_model(model_name)

    length = 2048
    for domain in tqdm(["c4", "ja_wikipedia", "en_wikipedia", "ja_mc4", "code_stack",]):
        print(f"Domain: {domain}")
        layer_counters, crosslayer_counters, eid2token = run_analysis(
            domain, model_name
        )
        Path(f"routing_output/{model_name}/expert_counts").mkdir(
            parents=True, exist_ok=True
        )
        Path(f"routing_output/{model_name}/expert_counts_crosslayer").mkdir(
            parents=True, exist_ok=True
        )
        Path(f"routing_output/{model_name}/eid2token").mkdir(
            parents=True, exist_ok=True
        )
        with open(
            f"routing_output/{model_name}/expert_counts/{name2finaldata[domain]}.pkl",
            "wb",
        ) as f:
            pkl.dump(dict(layer_counters), f)
        with open(
            f"routing_output/{model_name}/expert_counts_crosslayer/{name2finaldata[domain]}.pkl",
            "wb",
        ) as f:
            pkl.dump(dict(crosslayer_counters), f)
        with open(
            f"routing_output/{model_name}/eid2token/{name2finaldata[domain]}.pkl", "wb"
        ) as f:
            pkl.dump(dict(eid2token), f)
