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


def make_filename_bins(
    local_to_file_map: Dict[str, List[str]],
) -> Tuple[List[List[str]], List[List[str]]]:
    # Allocate local weight name into bins, where each bin access independent files
    # Then we can use multiple threads to concurrently load each bin's parameters.
    # This function has a complexity of O(F + L²)
    # where F = total number of files, L = number of local names
    if not local_to_file_map:
        return [], []

    local_names = list(local_to_file_map.keys())
    n = len(local_names)

    # Convert file lists to sets for O(1) lookups and create file-to-locals mapping
    local_to_files = {name: set(local_to_file_map[name]) for name in local_names}
    file_to_locals = defaultdict(set)
    for local_name, files in local_to_files.items():
        for file in files:
            file_to_locals[file].add(local_name)

    # Union-Find with path compression and union by rank
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]

    def union(x, y):
        root_x, root_y = find(x), find(y)
        if root_x == root_y:
            return

        # Union by rank
        if rank[root_x] < rank[root_y]:
            root_x, root_y = root_y, root_x
        parent[root_y] = root_x
        if rank[root_x] == rank[root_y]:
            rank[root_x] += 1

    # Create name-to-index mapping for O(1) lookups
    name_to_idx = {name: i for i, name in enumerate(local_names)}

    # Union locals that share files - O(F) where F is total number of files
    for locals_sharing_file in file_to_locals.values():
        if len(locals_sharing_file) > 1:
            locals_list = list(locals_sharing_file)
            first_idx = name_to_idx[locals_list[0]]
            for local_name in locals_list[1:]:
                union(first_idx, name_to_idx[local_name])

    # Group by root - O(L)
    root_to_group = defaultdict(list)
    for i, name in enumerate(local_names):
        root_to_group[find(i)].append(name)

    # Build result groups - O(L + F)
    grouped_local_names = []
    grouped_filenames = []

    for group in root_to_group.values():
        grouped_local_names.append(group)
        # Use set union to merge files from all locals in group
        all_files = set()
        for local_name in group:
            all_files.update(local_to_files[local_name])
        grouped_filenames.append(list(all_files))

    return grouped_local_names, grouped_filenames


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

    grouped_local_names, grouped_filenames = make_filename_bins(local_to_file_map)

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
