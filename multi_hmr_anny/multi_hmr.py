# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import torch
from torch import nn
import numpy as np
import roma
from utils import unpatch, inverse_perspective_projection, perspective_projection
import torch.nn.functional as F
import anny

from multi_hmr_anny.hph import HPH
from multi_hmr_anny.pos_embed import get_2d_sincos_pos_embed
from multi_hmr_anny.encoder import Encoder

from utils import rotation_to_homogeneous

class Multi_HMR(nn.Module):
    """ A ViT backbone followed by a "HPH" head (stack of cross attention layers with queries corresponding to detected humans.) """

    def __init__(self,
                 img_size=896,
                 # Backbone
                 backbone='dinov2_vits14',
                 pretrained_backbone=False,
                 # HPH
                 xat_dim=512, # XAT dim
                 xat_depth=8, # number of cross attention block (SA, CA, MLP) in the HPH head.
                 xat_heads=16, # Number of attention heads
                 xat_dim_head=32, #
                 xat_mlp_dim=4*512,
                 xat_dropout=0.0,
                 # Anny
                 person_center='head',
                 num_betas=11,
                 default_pose_parameterization='root_relative_world',
                *args, **kwargs):
        super().__init__()

        assert kwargs['simple_depth_encoding'] == 1
        self.simple_depth_encoding = 1

        # Encoder
        self.img_size = img_size
        self.encoder = Encoder(backbone, pretrained=pretrained_backbone)
        assert self.img_size % self.encoder.patch_size == 0, "Invalid img size"
        self.patch_size = self.encoder.patch_size

        # Positional/Token embedding
        dec_pos_emb = get_2d_sincos_pos_embed(embed_dim=xat_dim, grid_size=self.img_size//self.patch_size)
        self.register_buffer('dec_pos_emb', torch.from_numpy(dec_pos_emb).float())
        self.dec_to_token = nn.Linear(self.encoder.embed_dim, xat_dim)
        
        # HumanPerceptionHead
        self.decoder = HPH(dim=xat_dim, depth=xat_depth, heads=xat_heads, dim_head=xat_dim_head, mlp_dim=xat_mlp_dim, dropout=xat_dropout)

        # 2D-Offset
        self.mlp_offset = nn.Sequential(*[nn.Linear(self.decoder.dim, self.decoder.dim), nn.ReLU(), nn.Linear(self.decoder.dim,2)])
        
        # Human properties
        self.n_joints = 163
        self.num_betas = num_betas
        self.mlp_pose = nn.Sequential(*[nn.Linear(self.decoder.dim+self.n_joints*6, self.decoder.dim), nn.ReLU(), nn.Linear(self.decoder.dim,self.n_joints*6)])
        self.mlp_shape = nn.Sequential(*[nn.Linear(self.decoder.dim, self.decoder.dim), nn.ReLU(), nn.Linear(self.decoder.dim,self.num_betas)])
        self.mlp_dist = nn.Sequential(*[nn.Linear(self.decoder.dim, self.decoder.dim), nn.ReLU(), nn.Linear(self.decoder.dim,1)])

        # Parametric 3D model
        self.person_center = person_center
        self.body_model = anny.create_fullbody_model(remove_unattached_vertices=False, all_phenotypes=True).to(dtype=torch.float32)
        joint_names = self.body_model.bone_labels
        self.person_center_idx = joint_names.index(f"{self.person_center}")
        self.body_model.shape_keys = [k for k in self.body_model.phenotype_labels if k != 'race']
        self.body_model.shape_keys.extend(self.body_model.phenotype_labels[-3:])
        self.eye = nn.Parameter(torch.eye(3).unsqueeze(0), requires_grad=False)
        self.body_model.set_skinning_method('lbs')
        self.body_model.name = 'anny'

        self.useful_rotmat = nn.Parameter(torch.Tensor([1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                        0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                        0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1.,
                                                        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                                        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                                        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0.,
                                                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                        0.]).unsqueeze(0), requires_grad=False)

        # init
        # for root it should be [np.pi/2.,0,0.]
        init_root_pose = roma.rotvec_to_rotmat(torch.tensor([[np.pi/2.,0,0.]]))[:,:,:2].flatten(1).reshape(1, -1)
        init_body_pose = torch.eye(3).reshape(1,3,3).repeat(self.n_joints-1,1,1)[:,:,:2].flatten(1).reshape(1, -1)
        init_body_pose = torch.cat([init_root_pose, init_body_pose], -1)
        self.register_buffer('init_body_pose', init_body_pose)
        

    def forward(self, x, K=None, 
                idx=None, 
                is_training=False, 
                det_thresh=0.3, nms_kernel_size=3,
                conf_thresh=None, dist_thresh_nms = None,
                *args, **kwargs,
                ):
        persons = []
        out = {}

        # Image encoder
        z = self.encoder(x)

        # Intrinsics considered
        K_regressed = z['K']
        K = K_regressed if K is None else K

        # Easy post-processing for detection
        if not is_training:
            if nms_kernel_size > 1:
                pad = (nms_kernel_size - 1) // 2 if nms_kernel_size not in [2, 4] else nms_kernel_size // 2
                scores_max = nn.functional.max_pool2d(z['scores'].unsqueeze(1), (nms_kernel_size, nms_kernel_size), stride=1, padding=pad)[:,0]
                z['scores'] = z['scores'] * (scores_max == z['scores']).float()
                # print(z['scores'].reshape(-1).sort().values[-5:])
            idx = torch.where(z['scores'] >= det_thresh) if idx is None else idx # take ground-truth idx if given
            if len(idx[0]) == 0:
                return {}, []

        # Token for decoder
        dec_pos_emb = unpatch(self.dec_pos_emb.unsqueeze(0), patch_size=1, c=self.decoder.dim, img_size=int(np.sqrt(self.dec_pos_emb.shape[0]))).permute(0,2,3,1) # [1,sqrt(np),sqrt(np),d] - positional embedding
        dec_emb = self.dec_to_token(z['feat']) + dec_pos_emb # [bs',np,np,D]

        # Queries - Q
        queries = dec_emb[idx[0],idx[1],idx[2]]
        values, counts = torch.unique(idx[0], sorted=True, return_counts=True) # count the number of person per image
        queries = torch.stack([F.pad(q, (0,0,0,max(counts)-counts[i]), mode='constant', value=0) for i, q in enumerate(torch.split(queries, tuple(counts), dim=0))]) # [bs,max(counts),emb]
        mask = torch.cat([F.pad(torch.ones(1,c,1).to(x.device), (0,0,0,max(counts)-c), mode='constant', value=0) for c in tuple(counts)], dim=0)[...,0] # [bs,max(counts)]

        # Context - QV
        context = dec_emb[values].flatten(1,2) # [bs',np*np,D]

        # Decoder
        y = self.decoder(x=queries, context=context, mask=mask) # [bs,max(counts),emb]
        y = torch.cat([y[i,:c] for i, c in enumerate(tuple(counts))], dim=0) # [bs',emb]
        
        # Human primary keypoint 2D location
        offset = self.mlp_offset(y)
        loc_coarse = torch.stack([idx[2],idx[1]], dim=1) # swapping x and y axis
        loc = (loc_coarse + 0.5 + offset) * self.encoder.patch_size

        # Distance from camera
        _K = K[idx[0]]
        focal = _K[:,0,0]
        _dist = self.mlp_dist(y)
        dist = focal.unsqueeze(1) / torch.clamp(torch.exp(_dist), 1e-5)
        transl = inverse_perspective_projection(loc.unsqueeze(1), _K, dist.unsqueeze(1))[:,0]

        # Human-centered 3D properties
        shape = self.mlp_shape(y)
        rot6d = self.mlp_pose(torch.cat([y, self.init_body_pose.repeat(y.shape[0],1)],1)) + self.init_body_pose
        rotmat = roma.special_gramschmidt(rot6d.reshape(-1,3,2))
        rotmat = rotmat.view(-1,self.n_joints,3,3)
        
        # Taking only useful rotmat into account
        eye = torch.eye(3).to(rotmat.device).reshape(1,1,3,3).repeat(rotmat.shape[0],self.n_joints,1,1)
        rotmat = self.useful_rotmat.reshape(1,-1,1,1) * rotmat + \
                     (1-self.useful_rotmat.reshape(1,-1,1,1)) * eye
        rotvec = roma.rotmat_to_rotvec(rotmat)

        # shape
        shape = torch.sigmoid(shape) # because between 0 and 1
        _shape = {}
        for l, k in enumerate(self.body_model.phenotype_labels):
            if k in ['age', 'gender', 'weight', 'height', 'muscle', 'proportions']:
                _shape[k] = shape[:,l]

        # rotmat to homogenous matrix
        rotmat_homo = rotation_to_homogeneous(rotmat)

        # anny model
        output = self.body_model(pose_parameters=rotmat_homo, phenotype_kwargs=_shape)

        v3d = output['vertices']
        j3d = output['bone_poses'][:,:,:3,-1]
        person_center = j3d[:, [self.person_center_idx]]

        # Adding translation
        v3d, j3d = [x - person_center + transl.unsqueeze(1) for x in [v3d, j3d]]
        v2d, j2d = [perspective_projection(x, _K) for x in [v3d, j3d]]

        # Return
        out = {
            'scores': z['scores'],
            'scores_logits': z['scores_logits'],
            'K': K,
            'K_regressed' : K_regressed,
            'fov_regressed': z['fov'],
            'loc': loc,
            'offset': offset,
            'dist': dist,
            'dist_postprocessed': _dist,
            'shape': shape,
            'rotvec': rotvec,
            'rotmat': rotmat,
            'v3d': v3d, # in 3d camera space
            'j3d': j3d, # in 3d camera space
            'j2d': j2d, 
            'v2d': v2d, 
            'transl': transl, # translation of the primary keypoint
            'transl_pelvis': j3d[:,[0]], # root=pelvis
            'feat': z['feat'], # encoder features,
            'blendshape_coeffs': output['blendshape_coeffs'],
        }

        if is_training:
            return out
        else:
            for i in range(idx[0].shape[0]):
                person = {
                    # Camera parameters (redundantly repeated for each person)
                    'K': K[idx[0]][i],
                    'K_regressed': K_regressed[idx[0]][i],
                    # Detection
                    'loc': out['loc'][i], # 2d pixel location of the primary keypoints
                    # Params
                    'transl': out['transl'][i], # from the primary keypoint i.e. the head
                    'transl_pelvis': out['transl_pelvis'][i], # of the pelvis joint
                    'rotvec': out['rotvec'][i],
                    'rotmat': out['rotmat'][i],
                    'shape': out['shape'][i],
                    # 2D/3D points
                    'v3d': out['v3d'][i],
                    'j3d': out['j3d'][i],
                    'j2d': out['j2d'][i],
                    'fov': z['fov'],
                    }

                persons.append(person)

            # Re-order persons so that the closest to the camera is first (smallest z in 'transl')
            persons = sorted(persons, key=lambda p: p['transl'][2].item())

            glob_output = {
                'K': K,
                'K_regressed' : K_regressed,
            }

            # return glob_output, persons
            return persons