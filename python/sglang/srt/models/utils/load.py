import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from typing import Callable, Dict, List, Tuple

import torch
from safetensors import safe_open
from transformers.utils.hub import cached_file

from sglang.srt.model_loader.weight_utils import default_weight_loader


def get_actual_hf_path(weight_path: str):
    return os.path.dirname(cached_file(weight_path, "config.json"))


def load_weights_with_hf_path_fast(
    model: torch.nn.Module,
    weight_path: str,
    load_weights_with_worker_fn: Callable,
    stacked_params_mapping: List[Tuple[str, str, str]] | None = None,
    expert_params_mapping: List[Tuple[str, str, str]] | None = None,
    tie_word_embeddings: bool = False,
    max_workers: int = None,
):
    if not os.path.exists(weight_path):
        weight_path = get_actual_hf_path(weight_path)
    index_file = os.path.join(weight_path, "model.safetensors.index.json")
    index = {}
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            index = json.load(f)["weight_map"]
    else:
        # Search all safetensors files
        safetensor_files = glob(os.path.join(weight_path, "*.safetensors"))
        # If there are safetensors files
        if safetensor_files:
            # Iterate through each safetensors file
            for safetensor_file in safetensor_files:
                with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        index[k] = safetensor_file
        else:
            raise FileNotFoundError("No safetensors found in the model path to load.")

    params = dict(model.named_parameters())
    local_names = list(params.keys())

    worker_args = []

    # local name -> set of filenames that contains the weight
    local_to_file_map = defaultdict(set)
    # model.layers.31.mlp.experts
    for local_name in local_names:
        hf_names = []
        if "mlp.experts" not in local_name and stacked_params_mapping is not None:
            for param_name, shard_name, _ in stacked_params_mapping:
                if param_name in local_name:
                    hf_names.append(local_name.replace(param_name, shard_name))
        if expert_params_mapping is not None:
            for param_name, shard_name, _, _ in expert_params_mapping:
                if param_name in local_name:
                    hf_names.append(local_name.replace(param_name, shard_name))
        if tie_word_embeddings and "lm_head.weight" in local_name:
            hf_names.append("model.embed_tokens.weight")
        if len(hf_names) == 0:
            hf_names.append(local_name)
        for name in hf_names:
            filename = index[name]
            if filename not in local_to_file_map[local_name]:
                local_to_file_map[local_name].add(filename)

    # Use union find to create local_name groups with no file conflicts
    parent = {name: name for name in local_names}
    weight_groups = {name: [name] for name in local_names}
    file_groups = {name: local_to_file_map[name] for name in local_names}
    roots = [name for name in local_names]
    ranks = {name: 0 for name in local_names}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            if ranks[root_x] > ranks[root_y]:
                parent[root_y] = root_x
                roots.remove(root_y)
            elif ranks[root_x] < ranks[root_y]:
                parent[root_x] = root_y
                roots.remove(root_x)
            else:
                parent[root_y] = root_x
                roots.remove(root_y)
                ranks[root_x] += 1
            # Merge file groups
            file_groups[root_x].update(file_groups[root_y])
            file_groups[root_y] = file_groups[root_x]
            # Merge weight groups
            weight_groups[root_x].extend(weight_groups[root_y])
            weight_groups[root_y] = weight_groups[root_x]
            return True
        return False

    for i, weight1 in enumerate(local_names):
        for weight2 in local_names[i + 1 :]:
            # If two weights share any files, they conflict
            if any(fn in file_groups[weight1] for fn in file_groups[weight2]):
                union(weight1, weight2)

    grouped_local_names = [weight_groups[root] for root in roots]
    grouped_filenames = [list(file_groups[root]) for root in roots]

    if max_workers is None:
        # assume all GPUs are used by SGLang servers
        max_workers = min(8, max(1, os.cpu_count() // torch.cuda.device_count()))

    for local_names, filenames in zip(grouped_local_names, grouped_filenames):
        worker_args.append(
            dict(
                params=params,
                local_names=local_names,
                filenames=filenames,
                weight_path=weight_path,
            )
        )

    max_workers = min(max_workers, len(worker_args))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            lambda kwargs: load_weights_with_worker_fn(**kwargs), worker_args
        )
        # Consume all results to make result all tasks complete
        for _ in results:
            pass
