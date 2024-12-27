# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import os 
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ['EGL_DEVICE_ID'] = '0'

import sys
from argparse import ArgumentParser
import random
import pickle as pkl
import numpy as np
from PIL import Image, ImageOps
import torch
from tqdm import tqdm
import time

from utils import normalize_rgb, render_meshes, get_focalLength_from_fieldOfView, demo_color as color, print_distance_on_image, render_side_views, create_scene, MEAN_PARAMS, CACHE_DIR_MULTIHMR, SMPLX_DIR
from model import Model
from pathlib import Path
import warnings

import json
from pathlib import Path

torch.cuda.empty_cache()

np.random.seed(seed=0)
random.seed(0)

def open_image(img_path, img_size, device=torch.device('cuda')):
    """ Open image at path, resize and pad """

    # Open and reshape
    img_pil = Image.open(img_path).convert('RGB')
    img_pil = ImageOps.contain(img_pil, (img_size,img_size)) # keep the same aspect ratio

    # Keep a copy for visualisations.
    img_pil_bis = ImageOps.pad(img_pil.copy(), size=(img_size,img_size), color=(255, 255, 255))
    img_pil = ImageOps.pad(img_pil, size=(img_size,img_size)) # pad with zero on the smallest side

    # Go to numpy 
    resize_img = np.asarray(img_pil)

    # Normalize and go to torch.
    resize_img = normalize_rgb(resize_img)
    x = torch.from_numpy(resize_img).unsqueeze(0).to(device)
    return x, img_pil_bis

def get_camera_parameters(img_size, fov=60, p_x=None, p_y=None, device=torch.device('cuda')):
    """ Given image size, fov and principal point coordinates, return K the camera parameter matrix"""
    K = torch.eye(3)
    # Get focal length.
    focal = get_focalLength_from_fieldOfView(fov=fov, img_size=img_size)
    K[0,0], K[1,1] = focal, focal

    # Set principal point
    if p_x is not None and p_y is not None:
            K[0,-1], K[1,-1] = p_x * img_size, p_y * img_size
    else:
            K[0,-1], K[1,-1] = img_size//2, img_size//2

    # Add batch dimension
    K = K.unsqueeze(0).to(device)
    return K

def load_model(model_name, device=torch.device('cuda')):
    """ Open a checkpoint, build Multi-HMR using saved arguments, load the model weigths. """
    # Model
    ckpt_path = os.path.join(CACHE_DIR_MULTIHMR, model_name+ '.pt')
    if not os.path.isfile(ckpt_path):
        os.makedirs(CACHE_DIR_MULTIHMR, exist_ok=True)
        print(f"{ckpt_path} not found...")
        print("It should be the first time you run the demo code")
        print("Downloading checkpoint from NAVER LABS Europe website...")
        
        try:
            os.system(f"wget -O {ckpt_path} https://download.europe.naverlabs.com/ComputerVision/MultiHMR/{model_name}.pt")
            print(f"Ckpt downloaded to {ckpt_path}")
        except:
            print("Please contact fabien.baradel@naverlabs.com or open an issue on the github repo")
            return 0

    # Load weights
    print("Loading model")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Get arguments saved in the checkpoint to rebuild the model
    kwargs = {}
    for k,v in vars(ckpt['args']).items():
            kwargs[k] = v

    # Build the model.
    kwargs['type'] = ckpt['args'].train_return_type
    kwargs['img_size'] = ckpt['args'].img_size[0]
    model = Model(**kwargs).to(device)

    # Load weights into model.
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print("Weights have been loaded")

    return model

def forward_model(model, input_image, camera_parameters,
                  det_thresh=0.3,
                  nms_kernel_size=1,
                 ):
        
    """ Make a forward pass on an input image and camera parameters. """
    
    # Forward the model.
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            humans = model(input_image, 
                           is_training=False, 
                           nms_kernel_size=int(nms_kernel_size),
                           det_thresh=det_thresh,
                           K=camera_parameters)

    return humans

def overlay_human_meshes(humans, K, model, img_pil, unique_color=False):

    # Color of humans seen in the image.
    _color = [color[0] for _ in range(len(humans))] if unique_color else color
    
    # Get focal and princpt for rendering.
    focal = np.asarray([K[0,0,0].cpu().numpy(),K[0,1,1].cpu().numpy()])
    princpt = np.asarray([K[0,0,-1].cpu().numpy(),K[0,1,-1].cpu().numpy()])

    # Get the vertices produced by the model.
    verts_list = [humans[j]['verts_smplx'].cpu().numpy() for j in range(len(humans))]
    faces_list = [model.smpl_layer['neutral'].bm_x.faces for j in range(len(humans))]

    # Render the meshes onto the image.
    pred_rend_array = render_meshes(np.asarray(img_pil), 
            verts_list,
            faces_list,
            {'focal': focal, 'princpt': princpt},
            alpha=1.0,
            color=_color)

    return pred_rend_array, _color

def tensor_to_list(tensor):
    return tensor.detach().cpu().numpy().tolist()

if __name__ == "__main__":
        parser = ArgumentParser()
        parser.add_argument("--model_name", type=str, default='multiHMR_896_L_synth')
        parser.add_argument("--img_folder", type=str, default='example_data')
        parser.add_argument("--out_folder", type=str, default='demo_out')
        parser.add_argument("--save_mesh", type=int, default=0, choices=[0,1])
        parser.add_argument("--extra_views", type=int, default=0, choices=[0,1])
        parser.add_argument("--det_thresh", type=float, default=0.3)
        parser.add_argument("--nms_kernel_size", type=float, default=3)
        parser.add_argument("--fov", type=float, default=60)
        parser.add_argument("--distance", type=int, default=0, choices=[0,1], help='add distance on the reprojected mesh')
        parser.add_argument("--unique_color", type=int, default=0, choices=[0,1], help='only one color for all humans')
        parser.add_argument("--export_json", type=int, default=1, choices=[0,1], help='export parameters as JSON')
        
        args = parser.parse_args()

        dict_args = vars(args)

        assert torch.cuda.is_available()

        # SMPL-X models
        smplx_fn = os.path.join(SMPLX_DIR, 'smplx', 'SMPLX_NEUTRAL.npz')
        if not os.path.isfile(smplx_fn):
            print(f"{smplx_fn} not found, please download SMPLX_NEUTRAL.npz file")
            print("To do so you need to create an account in https://smpl-x.is.tue.mpg.de")
            print("Then download 'SMPL-X-v1.1 (NPZ+PKL, 830MB) - Use thsi for SMPL-X Python codebase'")
            print(f"Extract the zip file and move SMPLX_NEUTRAL.npz to {smplx_fn}")
            print("Sorry for this incovenience but we do not have license for redustributing SMPLX model")
            assert NotImplementedError
        else:
             print('SMPLX found')
             
        # SMPL mean params download
        if not os.path.isfile(MEAN_PARAMS):
            print('Start to download the SMPL mean params')
            os.system(f"wget -O {MEAN_PARAMS}  https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smpl_mean_params.npz?versionId=CAEQHhiBgICN6M3V6xciIDU1MzUzNjZjZGNiOTQ3OWJiZTJmNThiZmY4NmMxMTM4")
            print('SMPL mean params have been succesfully downloaded')
        else:
            print('SMPL mean params is already here')

        # Input images
        suffixes = ('.jpg', '.jpeg', '.png', '.webp')
        l_img_path = [file for file in os.listdir(args.img_folder) if file.endswith(suffixes) and file[0] != '.']

        # Loading
        model = load_model(args.model_name)

        # Model name for saving results.
        model_name = os.path.basename(args.model_name)

        # All images
        os.makedirs(args.out_folder, exist_ok=True)
        l_duration = []
        for i, img_path in enumerate(tqdm(l_img_path)):
                
            # Path where the image + overlays of human meshes + optional views will be saved.
            save_fn = os.path.join(args.out_folder, f"{Path(img_path).stem}_{model_name}.png")

            # Get input in the right format for the model
            img_size = model.img_size
            x, img_pil_nopad = open_image(os.path.join(args.img_folder, img_path), img_size)

            # Get camera parameters
            p_x, p_y = None, None
            K = get_camera_parameters(model.img_size, fov=args.fov, p_x=p_x, p_y=p_y)

            # Make model predictions
            start = time.time()
            humans = forward_model(model, x, K,
                                   det_thresh=args.det_thresh,
                                   nms_kernel_size=args.nms_kernel_size)
            duration = time.time() - start
            l_duration.append(duration)

            if args.export_json:
                # Crear un diccionario para almacenar los parámetros
                params_dict = {
                    "image_width": img_pil_nopad.size[0],
                    "image_height": img_pil_nopad.size[1],
                    "camera_intrinsics": tensor_to_list(K[0]),
                    "humans": []
                }

                for human in humans:
                    human_params = {
                        "location": tensor_to_list(human['loc']),
                        "translation": tensor_to_list(human['transl']),
                        "translation_pelvis": tensor_to_list(human['transl_pelvis']),
                        "rotation_vector": tensor_to_list(human['pose']),
                        "expression": tensor_to_list(human['expression']),
                        "shape": tensor_to_list(human['shape']),
                        "joints_2d": tensor_to_list(human['j2d_smplx'])
                    }
                    params_dict["humans"].append(human_params)

                # Guardar el diccionario como JSON
                json_path = os.path.join(args.out_folder, f"{Path(img_path).stem}_{model_name}_params.json")
                with open(json_path, 'w') as f:
                    json.dump(params_dict, f, indent=2)
    
            # Superimpose predicted human meshes to the input image.
            img_array = np.asarray(img_pil_nopad)
            img_pil_visu= Image.fromarray(img_array)
            pred_rend_array, _color = overlay_human_meshes(humans, K, model, img_pil_visu, unique_color=args.unique_color)

            # Optionally add distance as an annotation to each mesh
            if args.distance:
                pred_rend_array = print_distance_on_image(pred_rend_array, humans, _color)

            # List of images too view side by side.
            l_img = [img_array, pred_rend_array]

            # More views
            if args.extra_views:
                # Render more side views of the meshes.
                pred_rend_array_bis, pred_rend_array_sideview, pred_rend_array_bev = render_side_views(img_array, _color, humans, model, K)

                # Concat
                _img1 = np.concatenate([img_array, pred_rend_array],1).astype(np.uint8)
                _img2 = np.concatenate([pred_rend_array_bis, pred_rend_array_sideview, pred_rend_array_bev],1).astype(np.uint8)
                _h = int(_img2.shape[0] * (_img1.shape[1]/_img2.shape[1]))
                _img2 = np.asarray(Image.fromarray(_img2).resize((_img1.shape[1], _h)))
                _img = np.concatenate([_img1, _img2],0).astype(np.uint8)
            else:
                 # Concatenate side by side
                _img = np.concatenate([img_array, pred_rend_array],1).astype(np.uint8)

            # Save to path.
            Image.fromarray(_img).save(save_fn)
            print(f"Avg Multi-HMR inference time={int(1000*np.median(np.asarray(l_duration[-1:])))}ms on a {torch.cuda.get_device_name()}")

            # Saving mesh
            if args.save_mesh:
                # npy file
                l_mesh = [hum['v3d'].cpu().numpy() for hum in humans]
                mesh_fn = save_fn+'.npy'
                np.save(mesh_fn, np.asarray(l_mesh), allow_pickle=True)
                x = np.load(mesh_fn, allow_pickle=True)

                # glb file
                l_mesh = [humans[j]['v3d'].detach().cpu().numpy() for j in range(len(humans))]
                l_face = [model.smpl_layer['neutral_10'].bm_x.faces for j in range(len(humans))]
                scene = create_scene(img_pil_visu, l_mesh, l_face, color=None, metallicFactor=0., roughnessFactor=0.5)
                scene_fn = save_fn+'.glb'
                scene.export(scene_fn)

        print('end')
