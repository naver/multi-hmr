# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ['EGL_DEVICE_ID'] = '0'

import warnings
import pickle
import torch
from utils.constants import SMPLX_DIR, BEDLAM_DIR, ANNOT_DIR
import smplx
from tqdm import tqdm
import sys
import numpy as np
from PIL import Image, ImageOps, ImageFile
import random
from utils import normalize_rgb, denormalize_rgb
ImageFile.LOAD_TRUNCATED_IMAGES = True # to avoid "OSError: image file is truncated"
from torch.utils.data import Dataset

class BEDLAM(Dataset):
    def __init__(self,
                 split='training',
                 training=False,
                 img_size=512,
                 root_dir=BEDLAM_DIR,
                 force_build_dataset=0,
                 n_iter=None,
                 subsample=1,
                 extension='png',
                 crops=[0],
                 flip=1,
                 res=None,
                 n=-1,
                 ):
        super().__init__()
        
        self.name = 'bedlam'
        self.annotations_dir = ANNOT_DIR
        self.training = training
        self.img_size = img_size
        self.n_iter = n_iter
        self.subsample = subsample
        self.crops = crops # 0 is the default
        self.flip = flip # 1 by default

        assert split in ['training', 'validation']
        
        self.root_dir = root_dir
        self.split = split
        self.image_dir = os.path.join(self.root_dir, f"{self.split}")
        self.annot_file = os.path.join(self.annotations_dir, f"{self.name}_{split}.pkl")
        self.force_build_dataset = force_build_dataset

        self.annots = None
        if self.force_build_dataset or not os.path.isfile(self.annot_file):
            self.annots = self.build_dataset()
        if self.annots is None:
            with open(self.annot_file, 'rb') as f:
                self.annots = pickle.load(f)

        # modif keys such that image extension are jpg
        assert extension in ['png', 'jpg']

        if extension == 'jpg':
            print(f"- Using jpg extension and res={res}")
            annots = {}
            for k, v in tqdm(self.annots.items()):
                sys.stdout.flush()
                if res == None:
                    name = k[:-3]+'jpg'
                else:
                    name = k[:-4]+f"_{res}.jpg"
                annots[name] = v.copy()
            del self.annots
            self.annots = annots

        self.imagenames = list(self.annots.keys())
        self.imagenames.sort()

        if n >= 0:
            self.imagenames = self.imagenames[:n]

        if self.subsample > 1:
            self.imagenames = [self.imagenames[k] for k in np.arange(0,len(self.imagenames),self.subsample).tolist()]

    def __len__(self):
        if self.training:
            return self.n_iter
        else:
            return len(self.imagenames)
        
    def __repr__(self):
        return f"{self.name}: split={self.split} - N={len(self.imagenames)}"
    
    @torch.no_grad()
    def build_dataset(self):
        print(f"Bulding annotation file for: {self.name} - {self.split}")

        imagename2annot = {}

        # Parametric 3D model
        model_neutral = smplx.create(SMPLX_DIR, 'smplx', gender='neutral', num_betas=11, use_pca=False, flat_hand_mean=True)

        # Annots files
        annot_dir = os.path.join(self.root_dir, f"all_npz_12_{self.split}")
        fns = os.listdir(annot_dir)
        fns.sort()
        for i_fn, fn in enumerate(tqdm(fns)):
            annot_x = np.load(os.path.join(self.root_dir, f"all_npz_12_{self.split}", fn))

            # Retrieving np.array once
            pose_cam_array = annot_x['pose_cam']
            K_array = annot_x['cam_int']
            H_array = annot_x['cam_ext']
            shape_array = annot_x['shape']
            imgname_array = annot_x['imgname']
            trans_cam_array = annot_x['trans_cam']

            l_imgname = list(set(imgname_array.tolist()))
            for _, imgname in enumerate(l_imgname):
                
                img_path = os.path.join(fn[:-4], 'png', imgname)
                if not os.path.exists(os.path.join(self.image_dir, img_path)):
                    continue
                
                # gte image w/o opening the file
                with open(os.path.join(self.image_dir, img_path), 'rb') as f:
                    f.seek(16)
                    width, height = int.from_bytes(f.read(4), 'big'), int.from_bytes(f.read(4), 'big')
                if 'closeup' in img_path:
                    width, height = height, width

                idxs = np.where(imgname == imgname_array)[0]

                persons = []
                for i in idxs:
                    sys.stdout.flush()

                    # SMPLX params
                    pose = pose_cam_array[i]
                    root_pose = pose[:3]
                    body_pose=pose[3:66]
                    jaw_pose=pose[66:69]
                    leye_pose=pose[69:72]
                    reye_pose=pose[72:75]
                    left_hand_pose=pose[75:120]
                    right_hand_pose=pose[120:165]
                    betas=shape_array[i]
                    transl = trans_cam_array[i] + H_array[i][:, 3][:3]

                    person = {
                        # SMPL GT in camera coordinates system
                        'smplx_root_pose': root_pose.reshape(1,3), # axis-angle
                        'smplx_body_pose': body_pose.reshape(21,3), # axis-angle
                        'smplx_jaw_pose': jaw_pose.reshape(1,3), # axis-angle
                        'smplx_leye_pose': leye_pose.reshape(1,3), # axis-angle
                        'smplx_reye_pose': reye_pose.reshape(1,3), # axis-angle
                        'smplx_left_hand_pose': left_hand_pose.reshape(15,3), # axis-angle
                        'smplx_right_hand_pose': right_hand_pose.reshape(15,3), # axis-angle
                        'smplx_shape': betas.reshape(11),
                        'smplx_gender': 'neutral',
                        'smplx_transl': transl.reshape(3),
                    }
                    persons.append(person)
                
                # Camera info
                K = K_array[i] # [3,3]
                focal = np.asarray([K[0,0], K[1,1]])
                princpt = np.asarray([K[0,-1], K[1,-1]])

                # Append
                imagename2annot[img_path] = {
                    # Camera
                    'focal': focal.astype(np.float32).reshape(2),
                    'princpt': princpt.astype(np.float32).reshape(2),
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
        if self.training:
            idx = random.choices(range(len(self.imagenames)))[0]
        else:
            pass
        imagename = self.imagenames[idx]
        annot = self.annots[imagename].copy()
        annot['imagename'] = imagename
        
        # find appropriate image_dir
        img_path = os.path.join(self.image_dir, imagename)

        # Original size
        real_width, real_height = annot['size']

        # Camera
        K = np.eye(3)
        princpt = annot['princpt'].copy()
        princpt_width, princpt_height = princpt[0]/real_width, princpt[1]/real_height # normalize between 0 and 1
        K[[0,1],[-1,-1]] = self.img_size * np.asarray([princpt_width, princpt_height]) # needs to be update if random cropping
        focal = annot['focal'].copy()
        K[[0,1],[0,1]] = focal  / (max([real_width, real_height]) / self.img_size) # update according to the new image size

        # preprocessing the image
        img_pil = Image.open(img_path)
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')

        # BEDLAM specifc to correct the rotation issue
        # https://github.com/pixelite1201/BEDLAM/blob/ebf8bb14a43de46cc74dca4c00c13e571b325726/visualize_ground_truth.py#L183
        if self.name == 'bedlam' and 'closeup' in imagename and self.split != 'test':
            img_pil = img_pil.rotate(-90, expand=True)

        # Flipping
        flip = self.flip and random.choice([0,1]) and self.training
        if flip:
            img_pil = ImageOps.mirror(img_pil)
            K[0,-1] = self.img_size - K[0,-1]

        # Crop
        crop = random.choice(self.crops) if self.training else 0
        if crop == 0:
            # extra zero-padding to make the image square
            img_pil = ImageOps.contain(img_pil, (self.img_size,self.img_size)) # keep the same aspect ratio
            img_pil = ImageOps.pad(img_pil, size=(self.img_size,self.img_size)) # pad with zero on the smallest side            
        else:
            raise NameError

        # PIL to np.array
        img_array = np.asarray(img_pil) # [height,width,3]
        img_array = normalize_rgb(img_array, imagenet_normalization=1)

        annot['K'] = K
        annot.pop('princpt')
        annot.pop('focal')

        # Humans
        _humans = annot['humans'].copy()
        annot.pop('humans')
        if self.training:
            humans = [hum for hum in _humans if hum['smplx_transl'][-1] > 0.01] # the person should be in front of the camera
        else:
            humans = [hum for hum in _humans]
        l_dist = [hum['smplx_transl'][-1] for hum in humans]
        indexed_lst = list(enumerate(l_dist))
        sorted_indexed = sorted(indexed_lst, key=lambda x: x[1], reverse=False)
        sorted_indices = [index for index, value in sorted_indexed]
        annot['humans'] = [humans[h_idx] for h_idx in sorted_indices]
        
        # Update smplx_gender - 0=neutral - 1=male - 2=female - kids?
        for hum in annot['humans']:
            hum['smplx_gender_id'] = np.asarray({'neutral': 0}[hum['smplx_gender']])

        # Update humans according to data-augment:
        if flip:
            humans = []
            for hum in annot['humans']:
                # pop
                _hum = hum.copy()
                for z in ['smplx_root_pose', 'smplx_body_pose', 'smplx_left_hand_pose', 'smplx_right_hand_pose', 'smplx_jaw_pose', 'smplx_transl', 'smplx_leye_pose', 'smplx_reye_pose']:
                    _hum.pop(z)

                # transl
                _hum['smplx_transl'] = hum['smplx_transl'].copy()
                _hum['smplx_transl'][0] = - hum['smplx_transl'][0]

                # root
                _pose = hum['smplx_root_pose'].copy()
                _pose[:, 1:3] *= -1
                _hum['smplx_root_pose'] = _pose

                # jaw
                _pose = hum['smplx_jaw_pose'].copy()
                _pose[:, 1:3] *= -1
                _hum['smplx_jaw_pose'] = _pose

                # body_pose
                _pose = hum['smplx_body_pose'].copy()
                orig_flip_pairs = ((0,1), (3,4), (6,7), (9,10), (12,13), (15,16), (17,18), (19,20))
                for pair in orig_flip_pairs:
                    _pose[pair[0], :], _pose[pair[1], :] = _pose[pair[1], :].copy(), _pose[pair[0], :].copy()
                _pose[:, 1:3] *= -1  # multiply -1 to y and z axis of axis-angle
                _hum['smplx_body_pose'] = _pose

                # hands
                lhand, rhand = hum['smplx_left_hand_pose'].copy(), hum['smplx_right_hand_pose'].copy()
                lhand[:, 1:3] *= -1
                rhand[:, 1:3] *= -1
                _hum['smplx_right_hand_pose'], _hum['smplx_left_hand_pose'] = lhand, rhand

                # eyes
                leye, reye = hum['smplx_leye_pose'].copy(), hum['smplx_reye_pose'].copy()
                leye[:, 1:3] *= -1
                reye[:, 1:3] *= -1
                _hum['smplx_reye_pose'], _hum['smplx_leye_pose'] = leye, reye

                humans.append(_hum)

            annot.pop('humans')
            annot['humans'] = humans

        return img_array, annot

def create_annots(splits=['validation', 'training']):
    for split in splits:
        dataset = BEDLAM(split=split, force_build_dataset=1)

def visualize(split='validation', i=1500, res=None, extension='png', training=0, img_size=800):
    # training - 52287 for a closeup
    from utils import render_meshes, demo_color
    model_neutral = smplx.create(SMPLX_DIR, 'smplx', gender='neutral', num_betas=11, use_pca=False, flat_hand_mean=True)

    dataset = BEDLAM(split=split, force_build_dataset=0,
                     res=res, extension=extension,
                     training=training,
                     img_size=img_size,
                     )
    print(dataset)
    
    img_array, annot = dataset.__getitem__(i)

    img_array = denormalize_rgb(img_array, imagenet_normalization=1)
    verts_list = []
    for person in annot['humans']:
        with torch.no_grad():
            verts = model_neutral(
                global_orient=torch.from_numpy(person['smplx_root_pose']).reshape(1,-1),
                body_pose=torch.from_numpy(person['smplx_body_pose']).reshape(1,-1),
                jaw_pose=torch.from_numpy(person['smplx_jaw_pose']).reshape(1,-1),
                leye_pose=torch.from_numpy(person['smplx_leye_pose']).reshape(1,-1),
                reye_pose=torch.from_numpy(person['smplx_reye_pose']).reshape(1,-1),
                left_hand_pose=torch.from_numpy(person['smplx_left_hand_pose']).reshape(1,-1),
                right_hand_pose=torch.from_numpy(person['smplx_right_hand_pose']).reshape(1,-1),
                betas=torch.from_numpy(person['smplx_shape']).reshape(1,-1),
                transl=torch.from_numpy(person['smplx_transl']).reshape(1,-1),
                ).vertices.cpu().numpy().reshape(-1,3)
        verts_list.append(verts)
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

    fn = f"{dataset.name}_{split}_{i}.jpg"
    Image.fromarray(img_array).save(fn)
    print(f"open {fn}")
    return 1

def collate_fn(x, *args, **kwargs):
    y = {}

    bs = len(x)

    # RGB image
    img_array = torch.from_numpy(np.stack([x[i][0] for i in range(bs)])).float() # [bs,3,W,H]
    y['imagename'] = np.stack([x[i][1]['imagename'] for i in range(bs)])
    
    # Camera
    y['K'] = torch.from_numpy(np.stack([x[i][1]['K'] for i in range(bs)])).float() # [bs,3,3]
    
    # Max number of persons
    y['n_humans'] = torch.from_numpy(np.stack([len(x[i][1]['humans']) for i in range(bs)])).float() # [bs]
    max_persons = int(max(y['n_humans']))

    # Validity index
    l_valid_humans = []
    for i in range(bs):
        n_humans = len(x[i][1]['humans'])
        valid_humans = np.concatenate([np.ones(n_humans), np.zeros(max_persons - n_humans)])
        l_valid_humans.append(valid_humans)
    y['valid_humans'] = torch.from_numpy(np.stack(l_valid_humans)).float() # [bs,max_persons]

    # Retrieve shapes of all keys - useful when no humans in a image
    all_keys = []
    key2shape = {}
    for i in range(bs):
        for h in range(len(x[i][1]['humans'])):
            keys = list(x[i][1]['humans'][h].keys())
            for k in keys:
                if isinstance(x[i][1]['humans'][h][k], np.ndarray):
                    all_keys.append(k)
                    key2shape[k] = x[i][1]['humans'][h][k].shape
                else:
                    pass

    all_keys = list(set(all_keys))

    # Humans
    for k in all_keys:
        l_values = []
        for i in range(bs):
            if len(x[i][1]['humans']) == 0:
                # no human in the image
                _shape = [0] + list(key2shape[k])
                value = np.zeros(_shape).astype(np.float32)
            else:
                value = np.stack([z[k] for z in x[i][1]['humans']])

            # zero pad
            shape = list(value.shape)
            shape[0] = max_persons - value.shape[0]
            value = np.concatenate([value, np.zeros(shape)])                    
            l_values.append(value)
        
        y_k = np.stack(l_values)
        y_k = torch.from_numpy(y_k).float() # [bs,max_persons,*D]
        
        y[k] = y_k

    return img_array, y

def dataloader(split='validation', batch_size=4, num_workers=0, shuffle=1, 
               extension='png', img_size=512, n=-1, res=None, n_iter=1000):
    from torch.utils.data import DataLoader
    dataset = BEDLAM(split=split, extension=extension, img_size=img_size, training=1, 
                     n=n, n_iter=n_iter*batch_size,
                     res=res)
    print(dataset)
    dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=shuffle,
                        drop_last=False,
                        collate_fn=collate_fn
                        )
    for ii, (x, y) in enumerate(tqdm(dataloader)):
        sys.stdout.flush()
        if ii == 100:
            print()
        if ii == n_iter:
            return

def create_jpeg(root_dir=BEDLAM_DIR, target_size=512):
    LOG_FREQ = 1000
    print(f"ROOT_DIR: {root_dir}")
    tot = 0
    for (dirpath, dirnames, filenames) in os.walk(root_dir):
        for fn in filenames:
            if fn[-4:] == '.png' and fn[0] != '.':
                img_path = os.path.join(dirpath, fn)
                img_pil = Image.open(img_path).convert('RGB')
                
                width, height = img_pil.size
                if target_size is not None:
                    if width > height:
                        percent = (target_size / float(img_pil.size[0]))
                        other_side = int((float(img_pil.size[1]) * float(percent)))
                        img_pil = img_pil.resize((target_size, other_side))
                    else:
                        percent = (target_size / float(img_pil.size[1]))
                        other_side = int((float(img_pil.size[0]) * float(percent)))
                        img_pil = img_pil.resize((other_side, target_size))
                    _img_path = os.path.join(dirpath, fn[:-4]+f"_{target_size}.jpg")
                else:
                    _img_path = os.path.join(dirpath, fn[:-4]+'.jpg')

                # img_pil.save('img.jpg')
                # ipdb.set_trace()

                img_pil.save(_img_path)
                tot += 1

                if tot % LOG_FREQ == 0:
                    print(f"Converted {tot} images so far")
                    sys.stdout.flush()


if __name__ == "__main__":
    exec(sys.argv[1])