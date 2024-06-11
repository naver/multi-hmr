# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import torch
from argparse import ArgumentParser

def _neg_loss(pred, gt):
  '''
  Code modified from: https://github.com/xingyizhou/CenterNet/blob/4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c/src/lib/models/losses.py#L42
    Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    
  '''
  assert pred.shape == gt.shape

  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  eps = 1e-7

  pos_loss = torch.log(pred + eps) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred + eps) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

class Loss(torch.nn.Module):
    def __init__(self, parser_args, *args, **kwargs):
        super().__init__()
        self.parser_args = parser_args

    def forward(self, y_hat, y, epoch=None, img_size=None):
        
        # Detection
        bce = _neg_loss(y_hat['scores'], (y['scores'] >= 1).to(int).unsqueeze(-1))
        reg_offset = (y_hat['offset'] - y['offset']).abs().sum(-1).mean(0)

        # SMPL-X params
        reg_rotmat = (y_hat['rotmat'] - y['rotmat']).abs().sum([1,2,3]).mean(0) # root/body/hands/jaw
        shape_dim = min([y_hat['shape'].shape[1], y['shape'].shape[1]])
        reg_shape = (y_hat['shape'][:,:shape_dim] - y['shape'][:,:shape_dim]).abs().sum(-1).mean(0)
        reg_dist = (y_hat['dist_postprocessed'].squeeze(1) - y['dist_postprocessed']).abs().mean(0)
        reg_transl = (y_hat['transl'] - y['transl']).abs().sum(-1).mean(0)
        # TODO expression???

        # 3D points
        # pelvis, pelvis_hat = y['j3d'][:,[0]], y_hat['j3d'][:,[0]]
        pelvis, pelvis_hat = y['transl_pelvis'].reshape(-1,1,3), y_hat['transl_pelvis'].reshape(-1,1,3)
        j3d, j3d_hat = y['j3d'] - pelvis, y_hat['j3d'] - pelvis_hat
        v3d, v3d_hat = y['v3d'] - pelvis, y_hat['v3d'] - pelvis_hat
        reg_j3d = (j3d - j3d_hat).abs().sum(-1).mean(-1).mean(0)
        reg_v3d = (v3d - v3d_hat).abs().sum(-1).mean(-1).mean(0)

        # 2D reprojection
        idx_v2d = torch.where(((y['v2d'] > 0).int() * (y['v2d'] < img_size).int()).sum(-1) == 2)
        reg_v2d = (y_hat['v2d'][idx_v2d[0],idx_v2d[1]] - y['v2d'][idx_v2d[0],idx_v2d[1]]).abs().sum(-1).mean(0)
        idx_j2d = torch.where(((y['j2d'] > 0).int() * (y['j2d'] < img_size).int()).sum(-1) == 2)
        reg_j2d = (y_hat['j2d'][idx_j2d[0],idx_j2d[1]] - y['j2d'][idx_j2d[0],idx_j2d[1]]).abs().sum(-1).mean(0)

        # handle nan/inf
        bce = torch.nan_to_num(bce, nan=0.0, posinf=0.0, neginf=0.0)
        reg_offset = torch.nan_to_num(reg_offset, nan=0.0, posinf=0.0, neginf=0.0)
        reg_rotmat = torch.nan_to_num(reg_rotmat, nan=0.0, posinf=0.0, neginf=0.0)
        reg_shape = torch.nan_to_num(reg_shape, nan=0.0, posinf=0.0, neginf=0.0)
        reg_dist = torch.nan_to_num(reg_dist, nan=0.0, posinf=0.0, neginf=0.0)
        reg_transl = torch.nan_to_num(reg_transl, nan=0.0, posinf=0.0, neginf=0.0)
        reg_j3d = torch.nan_to_num(reg_j3d, nan=0.0, posinf=0.0, neginf=0.0)
        reg_v3d = torch.nan_to_num(reg_v3d, nan=0.0, posinf=0.0, neginf=0.0)
        reg_j2d = torch.nan_to_num(reg_j2d, nan=0.0, posinf=0.0, neginf=0.0)
        reg_v2d = torch.nan_to_num(reg_v2d, nan=0.0, posinf=0.0, neginf=0.0)

        # total
        total_loss = self.parser_args.alpha_bce * bce +\
                     self.parser_args.alpha_offset * reg_offset +\
                     self.parser_args.alpha_rotmat * reg_rotmat +\
                     self.parser_args.alpha_shape * reg_shape +\
                     self.parser_args.alpha_dist * reg_dist +\
                     self.parser_args.alpha_transl * reg_transl +\
                     self.parser_args.alpha_j3d * reg_j3d +\
                     self.parser_args.alpha_v3d * reg_v3d
        if epoch >= self.parser_args.start_2d_epoch:
            total_loss += self.parser_args.alpha_j2d * reg_j2d +\
                          self.parser_args.alpha_v2d * reg_v2d

        # dict for tensorboard
        dict_loss = {
           'total': total_loss,
            'bce': bce,
            'offset': reg_offset,
            'rotmat': reg_rotmat,
            'shape': reg_shape,
            'dist': reg_dist,
            'transl': reg_transl,
            'j3d': reg_j3d,
            'v3d': reg_v3d,
            'j2d': reg_j2d,
            'v2d': reg_v2d,
        }

        return total_loss, dict_loss
    
    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # Detection
        parser.add_argument('--alpha_bce', type=float, default=10.0)
        parser.add_argument('--alpha_offset', type=float, default=1.0)

        # SMPL-X params
        parser.add_argument('--alpha_rotmat', type=float, default=0.1)
        parser.add_argument('--alpha_shape', type=float, default=1.0)
        parser.add_argument('--alpha_dist', type=float, default=1.0)
        parser.add_argument('--alpha_transl', type=float, default=1.0)

        # 3D
        parser.add_argument('--alpha_j3d', type=float, default=100.0)
        parser.add_argument('--alpha_v3d', type=float, default=100.0)

        # 2D
        parser.add_argument('--alpha_j2d', type=float, default=1.0)
        parser.add_argument('--alpha_v2d', type=float, default=1.0)
        parser.add_argument('--start_2d_epoch', type=int, default=10)

        
    
        return parser
   