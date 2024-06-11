# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ['EGL_DEVICE_ID'] = '0'

import warnings
import pickle
import torch
from utils.constants import SMPLX_DIR, EHF_DIR, ANNOT_DIR
import smplx
from tqdm import tqdm
import sys
import numpy as np
from PIL import Image, ImageOps, ImageFile
import random
from utils import normalize_rgb, denormalize_rgb
ImageFile.LOAD_TRUNCATED_IMAGES = True # to avoid "OSError: image file is truncated"
from torch.utils.data import Dataset
from plyfile import PlyData
import roma

class EHF(Dataset):
    def __init__(self,
                 split='test',
                 training=False,
                 img_size=512,
                 root_dir=EHF_DIR,
                 force_build_dataset=0,
                 *args, **kwargs
                 ):
        super().__init__()
        
        self.name = 'ehf'
        self.annotations_dir = ANNOT_DIR
        self.training = training
        self.img_size = img_size
        self.subsample = 1

        assert split in ['test']
        
        self.root_dir = root_dir
        self.split = split
        self.image_dir = self.root_dir
        self.annot_file = os.path.join(self.annotations_dir, f"{self.name}_{split}.pkl")
        self.force_build_dataset = force_build_dataset

        self.annots = None
        if self.force_build_dataset or not os.path.isfile(self.annot_file):
            self.annots = self.build_dataset()
        if self.annots is None:
            with open(self.annot_file, 'rb') as f:
                self.annots = pickle.load(f)

        self.imagenames = list(self.annots.keys())
        self.imagenames.sort()

    @torch.no_grad()
    def build_dataset(self):
        print(f"Bulding annotation file for: {self.name} - {self.split}")

        imagename2annot = {}

        # Camera parameters for the entire dataset
        cam_param = {'R': [-2.98747896, 0.01172457, -0.05704687]}
        cam_param['R'] = roma.rotvec_to_rotmat(torch.Tensor(cam_param['R'])).numpy()
        cam_param['t'] = np.array([-0.03609917, 0.43416458, 2.37101226])
        cam_param['K'] = np.array([[1498.22426237, 0, 790.263706],
                                   [0, 1498.22426237, 578.90334],
                                   [0, 0, 1]], dtype=np.float32)
        cam_param['focal'] = np.asarray([cam_param['K'][0,0],cam_param['K'][1,1]])
        cam_param['princpt'] = np.asarray([cam_param['K'][0,-1],cam_param['K'][1,-1]])

        fns = os.listdir(EHF_DIR)
        fns = [x for x in fns if '_align.ply' in x]
        fns.sort()

        for i, fn in enumerate(fns):
            # mesh
            mesh_gt_path = os.path.join(self.root_dir, fn)
            plydata = PlyData.read(mesh_gt_path)
            x = plydata['vertex']['x']
            y = plydata['vertex']['y']
            z = plydata['vertex']['z']
            mesh_gt = np.stack((x, y, z), 1)
            mesh_gt_cam = np.dot(cam_param['R'], mesh_gt.transpose(1, 0)).transpose(1, 0) + cam_param['t'].reshape(1, 3)

            persons = [{'smplx_vertices': mesh_gt_cam.reshape(-1,3)}]

            # get image w/o opening the file
            img_path = fn.replace('align.ply', 'img.png')
            with open(os.path.join(self.image_dir, img_path), 'rb') as f:
                f.seek(16)
                width, height = int.from_bytes(f.read(4), 'big'), int.from_bytes(f.read(4), 'big')

            # Append
            imagename2annot[img_path] = {
                    # Camera
                    'focal': cam_param['focal'].astype(np.float32).reshape(2),
                    'princpt': cam_param['princpt'].astype(np.float32).reshape(2),
                    'size': np.asarray([width, height]).astype(np.int32).reshape(2),
                    # Humans
                    'humans': persons
                }                

        # Saving
        os.makedirs(os.path.dirname(self.annot_file), exist_ok=True)
        print(f"Saving into {self.annot_file}")
        with open(self.annot_file, 'wb') as f:
            pickle.dump(imagename2annot, f, protocol=pickle.HIGHEST_PROTOCOL)

        return imagename2annot

    def __getitem__(self, idx):
        imagename = self.imagenames[idx]
        annot = self.annots[imagename].copy()
        annot['imagename'] = imagename

        # initial image
        img_path = os.path.join(self.   image_dir, imagename)
        img_pil = Image.open(img_path)

        # Resize image - squared output image for the moment
        real_width, real_height = annot['size']
        img_pil = ImageOps.contain(img_pil, (self.img_size,self.img_size)) # keep the same aspect ratio
        width, height = img_pil.size
        img_pil = ImageOps.pad(img_pil, size=(self.img_size,self.img_size)) # pad with zero on the smallest side
        img_array = np.asarray(img_pil) # [height,width,3]
        img_array = normalize_rgb(img_array)

        # Update principal point
        K = np.eye(3)
        K[[0,1],[-1,-1]] = self.img_size * (annot['princpt'].copy() / [real_width, real_height])

        # Update focal length
        if width > height: # because of the padding
            max_size_init = real_width
        else:
            max_size_init = real_height
        fovx = np.degrees(2 * np.arctan(max_size_init/ (2 * annot['focal'][0].copy())))
        fovy = np.degrees(2 * np.arctan(max_size_init/ (2 * annot['focal'][1].copy())))
        K[0,0] = self.img_size / (2 * np.tan(np.radians(fovx) /2))
        K[1,1] = self.img_size / (2 * np.tan(np.radians(fovy) /2))
        annot['K'] = K
        annot.pop('princpt')
        annot.pop('focal')

        return img_array, annot

    def __len__(self):
        return len(self.imagenames)
        
    def __repr__(self):
        return f"{self.name}: split={self.split} - N={len(self.imagenames)}"
    
def visualize(i=50, img_size=800):
    from utils import render_meshes, demo_color
    model_neutral = smplx.create(SMPLX_DIR, 'smplx', gender='neutral', num_betas=11, use_pca=False, flat_hand_mean=True)

    dataset = EHF(split='test', force_build_dataset=0, img_size=img_size)
    print(dataset)
    
    img_array, annot = dataset.__getitem__(i)

    img_array = denormalize_rgb(img_array)
    verts_list = []
    verts_list.append(annot['humans'][0]['smplx_vertices'])
    faces_list = [model_neutral.faces for _ in annot['humans']]
    _color = [demo_color[0] for _ in annot['humans']]
    pred_rend_array = render_meshes(img_array.copy(), 
                                            verts_list,
                                            faces_list,
                                            {'focal': annot['K'][[0,1],[0,1]],
                                             'princpt': annot['K'][[0,1],[-1,-1]]},
                                            alpha=0.7,
                                            color=_color)
    img_array = np.concatenate([img_array, np.asarray(pred_rend_array)], 1)

    fn = f"{dataset.name}_{dataset.split}_{i}.jpg"
    Image.fromarray(img_array).save(fn)
    print(f"open {fn}")
    return 1

def dataloader(img_size=512):
    from torch.utils.data import DataLoader
    from datasets.bedlam import collate_fn
    n_iter=1000
    dataset = EHF(split='test', img_size=img_size)
    print(dataset)
    dataloader = DataLoader(dataset,
                        batch_size=1,
                        num_workers=0,
                        shuffle=False,
                        drop_last=False,
                        collate_fn=collate_fn
                        )
    for ii, (x, y) in enumerate(tqdm(dataloader)):
        sys.stdout.flush()
        if ii == 100:
            print()
        if ii == n_iter:
            return

def create_annots():
    dataset = EHF(split='test', force_build_dataset=1)

    
if __name__ == "__main__":
    exec(sys.argv[1])