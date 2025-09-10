import os
import json
from glob import glob
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from typing import Tuple, List, Dict, Callable

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

    # local name -> list of filenames that contains the weight
    local_to_file_map = defaultdict(list)
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
                local_to_file_map[local_name].append(filename)

    # Allocate local weight name into bins, where each bin access independent files
    # Then we can use multiple threads to concurrently load each bin's parameters
    weight_name_bins = {}
    file_name_to_bin_index = {}
    bin_index = 0
    for local_name, filenames in local_to_file_map.items():
        if not all(filename in file_name_to_bin_index for filename in filenames):
            # Some required filenames doesn't have existing bins
            # Allocate a new bin, and merge all previous bins into this new bin
            weight_name_bins[bin_index] = [local_name]
            for filename in filenames:
                if filename in file_name_to_bin_index:
                    i = file_name_to_bin_index.pop(filename)
                    if i in weight_name_bins:
                        weight_name_bins[bin_index] += weight_name_bins.pop(i)
                file_name_to_bin_index[filename] = bin_index
            bin_index += 1
        else:
            # All required filenames have existing bins
            # Use the head bin as the master bin, and merge all other bins into the master bin
            head_i = file_name_to_bin_index[filenames[0]]
            weight_name_bins[head_i].append(local_name)
            if not all(
                file_name_to_bin_index[filename] == head_i for filename in filenames
            ):
                # merge to head bin
                for filename in filenames[1:]:
                    if file_name_to_bin_index[filename] != head_i:
                        i = file_name_to_bin_index.pop(filename)
                        if i in weight_name_bins:
                            weight_name_bins[head_i] += weight_name_bins.pop(i)
                        file_name_to_bin_index[filename] = head_i

    bin_index_to_file_names = defaultdict(list)
    for filename, bin_index in file_name_to_bin_index.items():
        bin_index_to_file_names[bin_index].append(filename)

    grouped_local_names = list(weight_name_bins.values())
    grouped_filenames = list(bin_index_to_file_names[i] for i in weight_name_bins)

    for local_names, filenames in zip(grouped_local_names, grouped_filenames):
        worker_args.append(
            dict(
                params=params,
                local_names=local_names,
                filenames=filenames,
                weight_path=weight_path,
            )
        )

    if max_workers is None:
        # assume all GPUs are used by SGLang servers
        max_workers = min(8, max(1, os.cpu_count() // torch.cuda.device_count()))
    max_workers = min(max_workers, len(worker_args))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            lambda kwargs: load_weights_with_worker_fn(**kwargs), worker_args
        )
        # Consume all results to make result all tasks complete
        for _ in results:
            pass
            