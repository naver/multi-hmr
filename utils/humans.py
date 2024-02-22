# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import numpy as np
import torch.nn.functional as F
import torch
import roma
from smplx.joint_names import JOINT_NAMES

def rot6d_to_rotmat(x):
    """
    6D rotation representation to 3x3 rotation matrix.
    Args:
        x: (B,6) Batch of 6-D rotation representations.
    Returns:
        torch.Tensor: Batch of corresponding rotation matrices with shape (B,3,3).
    """
    x = x.reshape(-1,2,3).permute(0, 2, 1).contiguous()
    y = roma.special_gramschmidt(x)
    return y

def get_smplx_joint_names(*args, **kwargs):
    return JOINT_NAMES[:127]