#!/usr/bin/env python3
"""Utility functions for benchmarks"""
import torch


def get_target_and_backend(target, arch=None):
    """
    Parse target string and return normalized target and execution backend.
    
    Args:
        target: Target platform string (e.g., "cutedsl", "cuda")
        arch: Optional architecture string (e.g., "sm_80", "sm_90a")
    
    Returns:
        tuple: (target_string, execution_backend)
    """
    actual_target = target
    if target == "cutedsl":
        target_string = f"{actual_target} -arch={arch}" if arch is not None else actual_target
        execution_backend = "cutedsl"
    else:
        target_string = f"{actual_target} -arch={arch}" if arch is not None else actual_target
        execution_backend = "tvm_ffi"
    
    return target_string, execution_backend


def get_gpu_info():
    """
    Get complete GPU information: architecture and model
    
    Returns:
        dict: {
            "arch": "sm_90",  # SM architecture
            "model": "NVIDIA H100 PCIe",  # GPU model
            "compute_capability": "9.0"
        }
    """
    # Get compute capability
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    compute_cap = f"{major}.{minor}"
    
    # Get GPU name
    try:
        gpu_name = torch.cuda.get_device_name(0)
    except Exception as e:
        print(f"Error getting GPU name: {e}")
        gpu_name = "Unknown GPU"
    
    return {
        "arch": arch,
        "model": gpu_name,
        "compute_capability": compute_cap
    }
