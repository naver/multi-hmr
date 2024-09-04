# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ['EGL_DEVICE_ID'] = '0'

from argparse import ArgumentParser
import torch
from datasets.bedlam import BEDLAM, collate_fn
from datasets.ehf import EHF
from datasets.threedpw import THREEDPW
from model import Model
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import time
import numpy as np
import smplx
from utils import perspective_projection, log_depth, focal_length_normalization, render_meshes, denormalize_rgb, SMPLX_DIR, AverageMeter, compute_prf1, match_2d_greedy, SMPLX2SMPL_REGRESSOR
from smplx.joint_names import JOINT_NAMES
import roma
from torch.utils.tensorboard import SummaryWriter
from loss import Loss
from PIL import Image
import pickle

class Trainer(object):
    def __init__(self, model, loss, optimizer, device, args, best_val=1e5):
        self.model = model
        self.loss = loss
        self.device = device
        self.args = args
        self.optimizer = optimizer
        self.best_val = best_val
        self.current_epoch = 0
        self.current_iter = 0

        # Parametric 3D human models
        self.smplx_neutral_11 = smplx.create(SMPLX_DIR, 'smplx', gender='neutral', use_pca=False, flat_hand_mean=True, num_betas=11).to(self.device)
        self.smpl_male_10 = smplx.create(SMPLX_DIR, 'smpl', gender='male').to(self.device)
        self.smpl_female_10 = smplx.create(SMPLX_DIR, 'smpl', gender='female').to(self.device)
        with open(SMPLX2SMPL_REGRESSOR, 'rb') as f:
            self.smplx2smpl_regressor = torch.from_numpy(pickle.load(f)['matrix'].astype(np.float32)).to(self.device)

        self.args.log_dir = os.path.join(self.args.save_dir, self.args.name)
        os.makedirs(self.args.log_dir, exist_ok=True)

        self.args.ckpt_dir = os.path.join(self.args.log_dir, 'checkpoints')
        os.makedirs(self.args.ckpt_dir, exist_ok=True)

        self.args.visu_dir = os.path.join(self.args.log_dir, 'visu')
        os.makedirs(self.args.visu_dir, exist_ok=True)

        self.writer = SummaryWriter(self.args.log_dir)

    def prepare_gt(self, y):
        target = {}

        bs, nhmax = y['valid_humans'].shape

        # Valid humans
        valid_h = y['valid_humans'] # [bs,nh_max]
        idx_h = torch.where(valid_h) # tuple of lenght=2
        nhv = int(valid_h.sum())
        K = y['K'][idx_h[0]]
        
        has_smplx_params = 0
        if 'smplx_vertices' in y:
            # EHF - only one person
            verts = y['smplx_vertices'].reshape(1,-1,3)
            jts = self.smplx_neutral_11.J_regressor @ verts
        elif 'smpl_root_pose' in y:
            # 3DPW - eval only
            out = self.smpl_male_10(
                    global_orient=y['smpl_root_pose'][idx_h[0],idx_h[1]].reshape(-1,3),
                    body_pose=y['smpl_body_pose'][idx_h[0],idx_h[1]].reshape(-1,23*3),
                    betas=y['smpl_shape'][idx_h[0],idx_h[1]].reshape(-1,10),
                    transl=y['smpl_transl'][idx_h[0],idx_h[1]].reshape(-1,3),
                    )
            verts, jts = out.vertices.reshape(nhv,-1,3), out.joints.reshape(nhv,-1,3)

            # update verts/joints if this is not the right gender
            if int(y['smpl_gender_id'].max()) == 2:
                out_female = self.smpl_female_10(
                    global_orient=y['smpl_root_pose'][idx_h[0],idx_h[1]].reshape(-1,3),
                    body_pose=y['smpl_body_pose'][idx_h[0],idx_h[1]].reshape(-1,23*3),
                    betas=y['smpl_shape'][idx_h[0],idx_h[1]].reshape(-1,10),
                    transl=y['smpl_transl'][idx_h[0],idx_h[1]].reshape(-1,3),
                    )
                idx = torch.where(y['smpl_gender_id'] == 2)[1]
                verts[idx] = out_female.vertices.reshape(nhv,-1,3)[idx]
                jts[idx] = out_female.joints.reshape(nhv,-1,3)[idx]
        elif 'smplx_root_pose' in y:
            # SMPLX forward on valid humans only - BEDLAM
            has_smplx_params = 1
            out = self.smplx_neutral_11(
                    global_orient=y['smplx_root_pose'][idx_h[0],idx_h[1]].reshape(-1,3),
                    body_pose=y['smplx_body_pose'][idx_h[0],idx_h[1]].reshape(-1,21*3),
                    jaw_pose=y['smplx_jaw_pose'][idx_h[0],idx_h[1]].reshape(-1,3),
                    leye_pose=y['smplx_leye_pose'][idx_h[0],idx_h[1]].reshape(-1,3),
                    reye_pose=y['smplx_reye_pose'][idx_h[0],idx_h[1]].reshape(-1,3),
                    left_hand_pose=y['smplx_left_hand_pose'][idx_h[0],idx_h[1]].reshape(-1,15*3),
                    right_hand_pose=y['smplx_right_hand_pose'][idx_h[0],idx_h[1]].reshape(-1,15*3),
                    betas=y['smplx_shape'][idx_h[0],idx_h[1]].reshape(-1,11),
                    transl=y['smplx_transl'][idx_h[0],idx_h[1]].reshape(-1,3),
                    expression=self.smplx_neutral_11.expression.repeat(nhv,1),
                    )
            verts, jts = out.vertices.reshape(nhv,-1,3), out.joints.reshape(nhv,-1,3)
        else:
            return None # no humans in the image - test time only
        j2d = perspective_projection(jts, K)
        v2d = perspective_projection(verts, K)

        # Translation of the primary keypoint
        root_joint_idx = JOINT_NAMES.index(self.args.person_center)
        target['transl'] = jts[:,root_joint_idx] # [nhv,3]
        target['transl_pelvis'] = jts[:,0] # [nhv,3]
        target['dist'] = jts[:,0,-1] # [nhv]

        # We may predict dist in log space, or normalized values.
        if self.model.nearness:
            non_euclidean_dist = log_depth(target['dist'])
        # Normalise by focal
        focal = K[:,0,0] # only focal of x
        non_euclidean_dist = focal_length_normalization(non_euclidean_dist, focal, fovn=60, img_size=self.model.img_size)
        target['dist_postprocessed'] = non_euclidean_dist

        # Fill in target
        target['v3d'] = verts
        target['j3d'] = jts
        target['j2d'] = j2d
        target['v2d'] = v2d

        # Creating the target heatmap for the primary keypoint
        n_patch = args.img_size // self.model.patch_size
        pk = target['transl'].unsqueeze(1) # (nhv,3)
        pk_loc = perspective_projection(pk, K).squeeze(1)
        pk_coarse_loc = (pk_loc // self.model.patch_size).int() # (nhv,2)
        pk_idx = torch.clamp(pk_coarse_loc, 0, n_patch - 1) # (nhv,2)
        pk_offset = (pk_loc - (pk_idx + 0.5) * self.model.patch_size) / self.model.patch_size # normalize from -0.5 to 0.5 from the center of the patch
            
        # Scores & updating valid_humans according to occlusion - wap X and Y for scores only
        scores = torch.zeros((bs, n_patch, n_patch)).to(self.device)
        visible_humans = torch.ones(nhv).to(self.device) # by default no occlusion so all visible
        for k in range(nhv):
            i = int(idx_h[0][k]) # index of the batch size
            j = int(idx_h[1][k]) # index of the human in this image
            _x = pk_idx[k,1]
            _y = pk_idx[k,0]
            if scores[i,_x,_y] == 1:
                valid_h[i,j] = 0
                visible_humans[k] = 0
            else:
                scores[i,_x,_y] = 1
        target['loc'] = pk_loc
        target['offset'] = pk_offset
        if has_smplx_params:
            target['rotvec'] = torch.cat([y['smplx_root_pose'],
                                        y['smplx_body_pose'],
                                        y['smplx_left_hand_pose'],
                                        y['smplx_right_hand_pose'],
                                        y['smplx_jaw_pose']],2)[idx_h[0],idx_h[1]] # [bs,nhmax]
            target['rotmat'] = roma.rotvec_to_rotmat(target['rotvec'])
            target['shape'] = y['smplx_shape'][idx_h[0],idx_h[1]]

        # Update with visibility indice
        _target = {}
        idx_vis = torch.where(visible_humans)[0]
        _target['idx'] = tuple([
            idx_h[0].to(self.device)[idx_vis],
            pk_idx[:,1].to(self.device)[idx_vis], 
            pk_idx[:,0].to(self.device)[idx_vis],
            torch.zeros_like(idx_h[0].to(self.device)[idx_vis]) # to match the size of the forward model
            ])
        _target['scores'] = scores # [bs,patch_size,patch_size]
        _target['K'] = y['K']
        for k, v in target.items():
            _target[k] = v[idx_vis] # discard unvisible humans due to olccusion

        return _target

    def fit(self, data_train, l_data_val):

        start_epoch = 0
        for epoch in range(start_epoch, self.args.max_epochs):
            
            # Training
            timer_end = time.time()
            self.train_n_iters(data_train)
            train_n_iters_time = time.time() - timer_end

            # Checkpointing
            model_state_dict = self.model.state_dict()
            l_x = []
            for k in model_state_dict.keys(): # discard smpl_layer
                    if 'smpl_layer_' in k:
                        l_x.append(k)
            for x in l_x:
                model_state_dict.pop(x)

            save_dict = {'epoch': self.current_epoch,
                        'iter': self.current_iter,
                        'model_state_dict': model_state_dict,
                        'args': self.args}
            torch.save(save_dict, os.path.join(self.args.ckpt_dir, f"{self.current_epoch:05d}.pt"))

            # Cleaning old ckpt
            epochs = []
            for x in os.listdir(self.args.ckpt_dir):
                if '.pt' in x:
                    epoch = int(x.split('.pt')[0])
                    epochs.append(epoch)
            epochs.sort()
            epochs_to_keep = epochs[-self.args.nb_max_ckpt:]
            for x in epochs:
                fn = os.path.join(self.args.ckpt_dir, f"{x:05d}.pt")
                if x not in epochs_to_keep:
                    try:
                        os.remove(fn)
                    except:
                        print('trying to remove')

            # Evaluating
            timer_end = time.time()
            for data_val in l_data_val:
                self.evaluate(data_val)
            evaluate_time = time.time() - timer_end

            # Flush metrcs to tensorboard
            self.writer.add_scalar(f"workload/train_n_iters", train_n_iters_time, self.current_epoch)
            self.writer.add_scalar(f"workload/evaluate", evaluate_time, self.current_epoch)
            self.writer.add_scalar(f"workload/ratio_trainVal", evaluate_time/(train_n_iters_time+evaluate_time), self.current_epoch)

            self.current_epoch += 1

        return 1
    
    def train_n_iters(self, data):
        print(f"\nTRAIN: ")
        self.model.train()

        meters = {k: AverageMeter(k) for k in ['workload/data', 'workload/batch', 'workload/ratio_data']}

        timer_end = time.time()
        for i, (x,y) in enumerate(tqdm(data)):
            
            # move tensor to device
            y = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in y.items()}
            data_time = time.time() - timer_end
            
            # preprare gt by computing mesh and 3d/2d joints
            gt = self.prepare_gt(y=y)

            # image to GPU
            x = x.to(self.device) # [bs,3,w,h]

            # Visu for debugging prepare_gt
            if 0:
                from PIL import Image
                from utils import render_meshes, denormalize_rgb
                print("VISU GT...")
                for k in tqdm(range(len(gt['scores']))):
                    # rgb
                    img_array = denormalize_rgb(x[k].cpu().numpy())

                    # hatmap primary kps
                    hm = gt['scores'][k].cpu().numpy()
                    hm = np.clip(hm + 0.1, 0, 1) # for visu purpose only
                    hm_array = np.asarray(Image.fromarray((hm*255).astype(np.uint8)).resize((img_array.shape[0],img_array.shape[1]), resample=Image.NEAREST)).reshape((img_array.shape[0],img_array.shape[1],1))
                    hm_array = (img_array * (hm_array / 255.)).astype(np.uint8)

                    # gt meshes
                    focal = gt['K'][k,[0,1],[0,1]].cpu().numpy()
                    princpt = gt['K'][k,[0,1],[-1,-1]].cpu().numpy()
                    gt_verts, gt_faces = [], []
                    for j in range(len(gt['v3d'])):
                        if gt['idx'][0][j] == k:
                            gt_verts.append(gt['v3d'][j].detach().cpu().numpy().reshape(-1,3))
                            gt_faces.append(self.smplx_neutral_11.faces)
                    gt_rend_array = render_meshes(img_array.copy(), 
                                                gt_verts, 
                                                gt_faces,
                                                {'focal': focal, 'princpt': princpt})

                    img = np.concatenate([img_array, hm_array, gt_rend_array], 1)
                    fn = f"img{k}.jpg"
                    Image.fromarray(img).save(fn)
                import ipdb;ipdb.set_trace()

            # Forward
            with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                pred = self.model(x, is_training=True, 
                                      idx=gt['idx'],
                                      K=gt['K'], 
                                      )

                # Loss
                loss, dict_loss = self.loss(pred, gt, epoch=self.current_epoch, img_size=self.args.img_size)

                # optim step
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                batch_time = time.time() - timer_end

                # meters
                meters['workload/data'].update(data_time)
                meters['workload/batch'].update(batch_time)
                meters['workload/ratio_data'].update(data_time/batch_time)
                for k, v in dict_loss.items():
                    k_name = f"loss/{k}"
                    if k_name not in meters:
                        meters[k_name] = AverageMeter(k_name)
                    meters[k_name].update(dict_loss[k].item())
                
                # Log
                if i % self.args.log_freq == 0:
                    print(f"EPOCH={self.current_epoch:03d} - i={i:05d}/{len(data):05d} - workload/ratio_data={meters['workload/ratio_data'].avg:.2f} - loss={meters['loss/total'].avg:.2f} - bce={meters['loss/bce'].avg:.2f} - v3d={meters['loss/v3d'].avg:.2f} - transl={meters['loss/transl'].avg:.2f}")

                    # write meters to tensorboard
                    for k, v in meters.items():
                        self.writer.add_scalar(f"{k}", v.avg, self.current_iter)

                    self.writer.flush() # https://github.com/pytorch/pytorch/issues/24234
                    sys.stdout.flush()

                self.current_iter += 1

                timer_end = time.time()

        return 1

    @torch.no_grad()
    def evaluate(self, data):
        print(f"\nEVAL: ")
        self.model.eval()

        meters = {k: AverageMeter(k) for k in ['pve', 'pa_pve', 'precision', 'recall', 'f1_score',
                                               'mpjpe', 'pa_mpjpe'
                                               ]}
        count, miss, fp = 0, 0, 0

        for i, (x,y) in enumerate(tqdm(data)):
            # move tensor to device
            y = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in y.items()}
            
            # preprare gt by computing mesh and 3d/2d joints
            gt = self.prepare_gt(y=y)

            # forward
            with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                x = x.to(self.device) # [bs,3,w,h]
                pred = self.model(x, is_training=False, K=gt['K'], 
                                  det_thresh=self.args.det_thresh, nms_kernel_size=self.args.nms_kernel_size)

            # match pred to gt - based on 2d bbox
            kp2d_gts = gt['j2d'].cpu().numpy()
            kp2d_preds = np.asarray([hum['j2d'].cpu().numpy()[:kp2d_gts.shape[1]] for hum in pred])
            bestMatch, falsePositives, misses = match_2d_greedy(kp2d_preds, kp2d_gts, np.ones_like(kp2d_gts[...,0]).astype(np.bool_))

            # detection metrics
            count += len(kp2d_gts)
            miss += len(misses)
            fp += len(falsePositives)

            # 3d metrics
            if len(bestMatch) > 0:
                for (pid, gid) in bestMatch:                    
                    # gt mesh centerex around pelvis
                    v3d = gt['v3d'][gid]
                    pelvis = gt['transl_pelvis'][gid].reshape(1,3)
                    v3d_ctx = v3d - pelvis

                    # pred mesh centerex around pelvis
                    v3d_hat = pred[pid]['v3d']
                    pelvis_hat = pred[pid]['transl_pelvis'].reshape(1,3)
                    v3d_hat_ctx = v3d_hat - pelvis_hat

                    # moving to smpl mesh for eval because gt are in smpl format
                    if v3d_ctx.shape[0] == 6890:
                       v3d_hat_ctx = (self.smplx2smpl_regressor @ v3d_hat_ctx)

                    # Per-Vertex Error
                    pve = ((torch.sqrt(((v3d_ctx - v3d_hat_ctx) ** 2).sum(-1))) * 1000).mean()
                    meters['pve'].update(pve.item())

                    # Procrustes-Aligned PVE
                    (R,t,s) = roma.rigid_points_registration(v3d_hat_ctx, v3d_ctx, compute_scaling=True)
                    pa_v3d_hat_ctx = s * (R.reshape(1,3,3) @ v3d_hat_ctx.reshape(-1,3,1)).reshape(-1,3) + t
                    pa_pve = ((torch.sqrt(((v3d_ctx - pa_v3d_hat_ctx) ** 2).sum(-1))) * 1000).mean()
                    meters['pa_pve'].update(pa_pve.item())

                    # MPJPE for 3DPW only
                    if data.dataset.name == '3dpw':
                        if i == 0:
                            # Can be download from https://github.com/nkolot/SPIN/blob/master/fetch_data.sh#L6C58-L6C78
                            self.J_regressor_h36m = torch.Tensor(np.load('models/smpl/J_regressor_h36m.npy')).to(self.device)
                            # https://github.com/nkolot/SPIN/blob/2476c436013055be5cb3905e4e4ecfa86966fac3/constants.py#L93C1-L95C31
                            self.H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
                            self.H36M_TO_J14 = self.H36M_TO_J17[:14]

                        # H36m joints
                        h36m = self.J_regressor_h36m @ v3d_ctx
                        h36m_hat = self.J_regressor_h36m @ v3d_hat_ctx

                        # center around h36m-pelvis
                        h36m_ctx = h36m - h36m[[0]]
                        h36m_hat_ctx = h36m_hat - h36m_hat[[0]]

                        # 14 joints only
                        h36m_ctx = h36m_ctx[self.H36M_TO_J14]
                        h36m_hat_ctx = h36m_hat_ctx[self.H36M_TO_J14]

                        # 17 joints only
                        # h36m_ctx = h36m_ctx[self.H36M_TO_J17]
                        # h36m_hat_ctx = h36m_hat_ctx[self.H36M_TO_J17]

                        # MPJPE
                        mpjpe = ((torch.sqrt(((h36m_ctx - h36m_hat_ctx) ** 2).sum(-1))) * 1000).mean()
                        meters['mpjpe'].update(mpjpe.item())

                        # PA-MPJPE
                        (R,t,s) = roma.rigid_points_registration(h36m_hat_ctx, h36m_ctx, compute_scaling=True)
                        pa_h36m_hat_ctx = s * (R.reshape(1,3,3) @ h36m_hat_ctx.reshape(-1,3,1)).reshape(-1,3) + t
                        pa_mpjpe = ((torch.sqrt(((h36m_ctx - pa_h36m_hat_ctx) ** 2).sum(-1))) * 1000).mean()
                        meters['pa_mpjpe'].update(pa_mpjpe.item())
            
            # log
            if i % self.args.log_freq == 0:
                precision, recall, f1_score = compute_prf1(count, miss, fp)
                if data.dataset.name == '3dpw':
                    print(f"i={i} - Recall={recall:.1f} - PVE={meters['pve'].avg:.1f} - PA-PVE={meters['pa_pve'].avg:.1f} - MPJPE={meters['mpjpe'].avg:.1f} - PA-MPJPE={meters['pa_mpjpe'].avg:.1f}")
                else:    
                    print(f"i={i} - Recall={recall:.1f} - PVE={meters['pve'].avg:.1f} - PA-PVE={meters['pa_pve'].avg:.1f}")
                sys.stdout.flush()

            # visu
            if self.args.visu_to_save > 0 and i < self.args.visu_to_save:
                # image
                img_array = denormalize_rgb(x[0].cpu().numpy())
                focal = gt['K'][0,[0,1],[0,1]].cpu().numpy()
                princpt = gt['K'][0,[0,1],[-1,-1]].cpu().numpy()

                # gt
                gt_verts, gt_faces = [], []
                for j in range(len(gt['v3d'])):
                    gt_verts.append(gt['v3d'][j].cpu().numpy().reshape(-1,3))
                    gt_faces.append(self.smplx_neutral_11.faces if gt['v3d'][j].shape[0] == 10475 else self.smpl_male_10.faces)
                gt_rend_array = render_meshes(img_array.copy(), 
                                                gt_verts, 
                                                gt_faces,
                                                {'focal': focal, 'princpt': princpt})
                
                # pred
                pred_verts, pred_faces = [], []
                for j in range(len(pred)):
                    pred_verts.append(pred[j]['v3d'].cpu().numpy().reshape(-1,3))
                    pred_faces.append(self.smplx_neutral_11.faces)
                pred_rend_array = render_meshes(img_array.copy(), 
                                                pred_verts, 
                                                pred_faces,
                                                {'focal': focal, 'princpt': princpt})

                img = np.concatenate([img_array, pred_rend_array, gt_rend_array], 1)
                # Image.fromarray(img).save('img.jpg');ipdb.set_trace() # debugging
                Image.fromarray(img).save(os.path.join(self.args.visu_dir, f"img_epoch{self.current_epoch:04d}_{data.dataset.name}_{i:04d}.jpg"))

        # final metrics
        print(f"***EVAL METRICS - {data.dataset.name}-{data.dataset.split}-{data.dataset.subsample}***")
        precision, recall, f1_score= compute_prf1(count, miss, fp)
        meters['precision'].update(precision)
        meters['recall'].update(recall)
        meters['f1_score'].update(f1_score)
        for k, v in meters.items():
            self.writer.add_scalar(f"{data.dataset.name}-{data.dataset.split}-{data.dataset.subsample}/{k}", v.avg, self.current_iter)
            print(f"    - {k}: {v.avg:.1f}")
        self.writer.flush() # https://github.com/pytorch/pytorch/issues/24234
        sys.stdout.flush()
        return meters['pve'].avg
    
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model(pretrained_backbone=1, **vars(args))
    model = model.to(device)

    # Load from a pretrained model
    if args.pretrained is not None and os.path.isfile(args.pretrained):
        print(f"Loading weights from {args.pretrained}")
        ckpt = torch.load(args.pretrained)
        log = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"{log}")

    l_val_data = []
    assert len(args.val_split) == len(args.val_data) == len(args.val_subsample)
    for i in range(len(args.val_data)):
        val_data = DataLoader(eval(args.val_data[i])(split=f"{args.val_split[i]}", 
                                                training=0, 
                                                img_size=args.img_size,
                                                subsample=args.val_subsample[i], # for fast evaluation on a subsampled part of the validation
                                                n=args.val_n[i], # for debugging purpose only
                                                ),
                            batch_size=1,
                            num_workers=0,
                            shuffle=False,
                            drop_last=False,
                            collate_fn=collate_fn,
                            )
        l_val_data.append(val_data)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    loss = Loss(args)

    trainer = Trainer(model=model, loss=loss, optimizer=optimizer, device=device, args=args)

    print()
    print(f"ARGS: {trainer.args}")
    print(f"LOG_DIR: {trainer.args.log_dir}")
    print()

    if args.eval_only:
        for val_data in l_val_data: 
            trainer.evaluate(val_data)
    else:
        train_data = DataLoader(eval(args.train_data)(split=f"{args.train_split}", 
                                                  training=1, 
                                                  img_size=args.img_size,
                                                  n_iter=args.batch_size * args.n_iters_per_epoch,
                                                  subsample=args.train_subsample,
                                                  extension=args.extension,
                                                  res=args.res,
                                                  n=args.train_n, # for debugging purpose only
                                                  crops=args.crops,
                                                  flip=args.flip,
                                                  ),
                          batch_size=args.batch_size,
                          num_workers=args.num_workers,
                          shuffle=True,
                          drop_last=True,
                          collate_fn=collate_fn,
                          )
        trainer.fit(train_data, l_val_data)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--train_data', type=str, default='BEDLAM')
    parser.add_argument('--train_split', type=str, default='training')
    parser.add_argument('--train_n', type=int, default=-1)
    parser.add_argument('--val_data', type=str, nargs='+', default=['BEDLAM', 'EHF', 'THREEDPW'])
    parser.add_argument('--val_split', type=str, nargs='+', default=['validation', 'test', 'test'])
    parser.add_argument('--val_n', type=int, nargs='+', default=[-1, -1, -1])
    parser.add_argument('--val_subsample', type=int, nargs='+', default=[25, 1, 20])
    parser.add_argument('--save_dir', type=str, default='logs')
    parser.add_argument('--name', type=str, default='trainval')
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--train_subsample', type=int, default=1)
    parser.add_argument('--num_workers', '-j', type=int, default=0)
    parser.add_argument('--img_size', type=int, default=336)
    parser.add_argument('--backbone', type=str, default='dinov2_vits14', choices=['dinov2_vitl14', 'dinov2_vitb14', 'dinov2_vits14'])
    parser.add_argument("--n_iters_per_epoch", "-iter", type=int, default=100)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--max_iter", type=int, default=10000)
    parser.add_argument("--nb_max_ckpt", type=int, default=10)    
    parser.add_argument('--amp', type=int, default=1, choices=[0,1], help="Use Automatic Mixed Precision for pretraining")
    parser.add_argument('--use_efficient_attention', type=int, default=1, choices=[0,1], help="Use Automatic Mixed Precision for pretraining")
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-6, help='learning rate (absolute lr)')
    parser.add_argument('--eval_only', type=int, default=0, choices=[0,1])
    parser.add_argument('--person_center', type=str, default='head', choices=['pelvis', 'head', 'nose'])
    parser.add_argument('--visu_to_save', type=int, default=0)
    parser.add_argument('--extension', type=str, default='png', choices=['png', 'jpg'])
    parser.add_argument('--res', type=int, default=None, choices=[None, 512, 1280])
    parser.add_argument('--num_betas', type=int, default=10, choices=[10, 11])
    parser.add_argument('--det_thresh', type=float, default=0.2)
    parser.add_argument('--nms_kernel_size', type=int, default=3)
    parser.add_argument('--crops', type=int, nargs='+', default=[0])
    parser.add_argument('--flip', type=int, default=1, choices=[0,1])
    parser.add_argument('--brightness', type=float, default=0.)
    parser.add_argument('--contrast', type=float, default=0.)
    parser.add_argument('--saturation', type=float, default=0.)
    parser.add_argument('--hue', type=float, default=0.)

    parser = Loss.add_specific_args(parser)
    args = parser.parse_args()
    args.max_epochs = args.max_iter // args.n_iters_per_epoch

    main(args)