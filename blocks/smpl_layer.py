# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import torch
from torch import nn
from torch import nn
import smplx
import torch
import numpy as np
import utils
from utils import inverse_perspective_projection, perspective_projection
import roma
import pickle
import os
from utils.constants import SMPLX_DIR

class SMPL_Layer(nn.Module):
    """
    Extension of the SMPL Layer with information about the camera for (inverse) projection the camera plane.
    """
    def __init__(self, 
                 type='smplx', 
                 gender='neutral', 
                 num_betas=10,
                 kid=False,
                 person_center=None,
                 *args, 
                 **kwargs,
                 ):
        super().__init__()

        # Args
        assert type == 'smplx'
        self.type = type
        self.kid = kid
        self.num_betas = num_betas
        self.bm_x = smplx.create(SMPLX_DIR, 'smplx', gender=gender, use_pca=False, flat_hand_mean=True, num_betas=num_betas)

        # Primary keypoint - root
        self.joint_names = eval(f"utils.get_{self.type}_joint_names")()
        self.person_center = person_center
        self.person_center_idx = None
        if self.person_center is not None:
            self.person_center_idx = self.joint_names.index(self.person_center)

    def forward(self,
                pose, shape,
                loc, dist, transl,
                K,
                expression=None, # facial expression
                ):
        """
        Args:
            - pose: pose of the person in axis-angle - torch.Tensor [bs,24,3]
            - shape: torch.Tensor [bs,10]
            - loc: 2D location of the pelvis in pixel space - torch.Tensor [bs,2]
            - dist: distance of the pelvis from the camera in m - torch.Tensor [bs,1]
        Return:
            - dict containing a bunch of useful information about each person
        """
        
        if loc is not None and dist is not None:
            assert pose.shape[0] == shape.shape[0] == loc.shape[0] == dist.shape[0]
        if self.type == 'smpl':
            assert len(pose.shape) == 3 and list(pose.shape[1:]) == [24,3]
        elif self.type == 'smplx':
            assert len(pose.shape) == 3 and list(pose.shape[1:]) == [53,3] # taking root_orient, body_pose, lhand, rhan and jaw for the moment
        else:
            raise NameError
        assert len(shape.shape) == 2 and (list(shape.shape[1:]) == [self.num_betas] or list(shape.shape[1:]) == [self.num_betas+1])
        if loc is not None and dist is not None:
            assert len(loc.shape) == 2 and list(loc.shape[1:]) == [2]
            assert len(dist.shape) == 2 and list(dist.shape[1:]) == [1]

        bs = pose.shape[0]

        out = {}

        # No humans
        if bs == 0:
            return {}
        
        # Low dimensional parameters        
        kwargs_pose = {
            'betas': shape,
        }
        kwargs_pose['global_orient'] = self.bm_x.global_orient.repeat(bs,1)
        kwargs_pose['body_pose'] = pose[:,1:22].flatten(1)
        kwargs_pose['left_hand_pose'] = pose[:,22:37].flatten(1)
        kwargs_pose['right_hand_pose'] = pose[:,37:52].flatten(1)
        kwargs_pose['jaw_pose'] = pose[:,52:53].flatten(1)

        if expression is not None:
            kwargs_pose['expression'] = expression.flatten(1) # [bs,10]
        else:
            kwargs_pose['expression'] = self.bm_x.expression.repeat(bs,1)

        # default - to be generalized
        kwargs_pose['leye_pose'] = self.bm_x.leye_pose.repeat(bs,1)
        kwargs_pose['reye_pose'] = self.bm_x.reye_pose.repeat(bs,1)        
        
        # Forward using the parametric 3d model SMPL-X layer
        output = self.bm_x(**kwargs_pose)
        verts = output.vertices
        j3d = output.joints # 45 joints
        R = roma.rotvec_to_rotmat(pose[:,0])

        # Apply global orientation on 3D points
        pelvis = j3d[:,[0]]
        j3d = (R.unsqueeze(1) @ (j3d - pelvis).unsqueeze(-1)).squeeze(-1)
        
        # Apply global orientation on 3D points - bis
        verts = (R.unsqueeze(1) @ (verts - pelvis).unsqueeze(-1)).squeeze(-1)

        # Location of the person in 3D
        if transl is None:
            if K.dtype == torch.float16:
                # because of torch.inverse - not working with float16 at the moment
                transl = inverse_perspective_projection(loc.unsqueeze(1).float(), K.float(), dist.unsqueeze(1).float())[:,0]    
                transl = transl.half()
            else:
                transl = inverse_perspective_projection(loc.unsqueeze(1), K, dist.unsqueeze(1))[:,0]

        # Updating transl if we choose a certain person center
        transl_up = transl.clone()

        # Definition of the translation depend on the args: 1) vanilla SMPL - 2) computed from a given joint
        if self.person_center_idx is None:
            # Add pelvis to transl - standard way for SMPLX layer
            transl_up = transl_up + pelvis[:,0]
        else:
            # Center around the joint because teh translation is computed from this joint
            person_center = j3d[:, [self.person_center_idx]]
            verts = verts - person_center
            j3d = j3d - person_center

        # Moving into the camera coordinate system
        j3d_cam = j3d + transl_up.unsqueeze(1)
        verts_cam = verts + transl_up.unsqueeze(1)

        # Projection in camera plane
        j2d = perspective_projection(j3d_cam, K)

        out.update({
            'verts_smplx_cam': verts_cam,
            'j3d': j3d_cam, 
            'j2d': j2d, 
            'transl': transl, # translation of the primary keypoint
            'transl_pelvis': j3d_cam[:,[0]], # root=pelvis
        })
            
        return out