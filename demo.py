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
from multi_hmr_anny.multi_hmr import Multi_HMR as ModelAnny
from pathlib import Path

torch.cuda.empty_cache()

np.random.seed(seed=0)
random.seed(0)
import ipdb
import glob

def open_image(img_path, img_size, device=torch.device('cuda')):
    """ Open image at path, resize and pad """

    # Open and reshape
    img_pil = Image.open(img_path).convert('RGB')
    aspect_ratio = img_pil.width / img_pil.height
    
    # keep the original image with padding for visualisation
    img_pil_full = img_pil.copy()
    # img_pil_full = ImageOps.pad(img_pil_full.copy(), size=(max(img_pil_full.size),max(img_pil_full.size)), color=(255, 255, 255))

    # Resize while keeping aspect ratio
    img_pil = ImageOps.contain(img_pil, (img_size,img_size)) # keep the same aspect ratio

    # Keep a copy for visualisations.
    img_pil_bis = ImageOps.pad(img_pil.copy(), size=(img_size,img_size), color=(255, 255, 255))
    img_pil = ImageOps.pad(img_pil, size=(img_size,img_size)) # pad with zero on the smallest side

    # Go to numpy 
    resize_img = np.asarray(img_pil)

    # Normalize and go to torch.
    resize_img = normalize_rgb(resize_img)
    x = torch.from_numpy(resize_img).unsqueeze(0).to(device)
    return x, img_pil_full

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
            os.system(f"wget -O {ckpt_path} http://download.europe.naverlabs.com/multihmr/{model_name}.pt")
            print(f"Ckpt downloaded to {ckpt_path}")
        except:
            assert "Please contact fabien.baradel@naverlabs.com or open an issue on the github repo"

    # Load weights
    print("Loading model")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Get arguments saved in the checkpoint to rebuild the model
    kwargs = {}
    for k,v in vars(ckpt['args']).items():
            kwargs[k] = v

    # Build the model.
    if 'anny' in ckpt_path:
        model = ModelAnny(**kwargs).to(device)
    else:
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
            # print(model.backbone.encoder.patch_embed.proj.bias.dtype, input_image.dtype)
            humans = model(input_image, 
                           is_training=False, 
                           nms_kernel_size=int(nms_kernel_size),
                           det_thresh=det_thresh,
                           K=camera_parameters)
            # print(humans[0]['j2d_smplx'].dtype)

    return humans

def overlay_human_meshes(humans, faces, K, model, img_pil, unique_color=False, alpha=0.8, _color=None):

    # Color of humans seen in the image.
    if _color is None:
        _color = [color[0] for _ in range(len(humans))] if unique_color else color
    
    # Get focal and princpt for rendering.
    focal = np.asarray([K[0,0,0].cpu().numpy(),K[0,1,1].cpu().numpy()])
    princpt = np.asarray([K[0,0,-1].cpu().numpy(),K[0,1,-1].cpu().numpy()])

    # Get the vertices produced by the model.
    pred_rend_array = np.asarray(img_pil)
    if len(humans) > 0:
        try:
            name = 'verts_smplx' if 'verts_smplx' in humans[0] else 'v3d'
            verts_list = [humans[j][name].cpu().numpy() for j in range(len(humans))]
            faces_list = [faces for j in range(len(humans))]

            # Render the meshes onto the image.
            pred_rend_array = render_meshes(np.asarray(img_pil), 
                    verts_list,
                    faces_list,
                    {'focal': focal, 'princpt': princpt},
                    alpha=alpha,
                    color=_color)
        except Exception as e:
            print("Rendering error:", e)
            if len(humans) > 0:
                print(humans[0].keys())

    return pred_rend_array, _color

def _generate_rotated_frames(humans, faces, K, model, img, center, name, n_frames, angle_range, axis, unique_color, alpha, _color):
    frames = []
    for i in range(n_frames):
        angle = angle_range * i / (n_frames - 1)
        theta = np.deg2rad(angle)
        if axis == 'y':
            rotmat = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
        elif axis == 'x':
            rotmat = np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]
            ])
        else:
            raise ValueError("Axis must be 'x' or 'y'")
        _humans = []
        for k in range(len(humans)):
            x = (humans[k][name].cpu().numpy() - center) @ rotmat.T + center
            _human_k = {name: torch.tensor(x)}
            _humans.append(_human_k)
        frame, _ = overlay_human_meshes(_humans, faces, K, model, img, unique_color=unique_color, alpha=alpha, _color=_color)
        frames.append(frame.astype(np.uint8))
    return frames

def create_rotating_video(humans, faces, K, model, img_pil_visu, unique_color=False, alpha=0.8, fn='rotating.mp4', n_frames=20, angle_range=60):
    if len(humans) == 0:
        return None

    print("Generating rotating video...")

    central, _color = overlay_human_meshes(humans, faces, K, model, img_pil_visu, unique_color=unique_color, alpha=alpha, _color=None)
    white_img = Image.new(img_pil_visu.mode, img_pil_visu.size, (255, 255, 255))
    closest_idx = 0
    name = 'verts_smplx' if 'verts_smplx' in humans[0] else 'v3d'
    center = humans[closest_idx][name].mean(0).cpu().numpy()

    frames_central_to_right = _generate_rotated_frames(humans, faces, K, model, white_img, center, name, n_frames, angle_range, 'y', unique_color, alpha, _color)
    frames_right_to_central = frames_central_to_right[::-1][1:-1]
    frames_central_to_left = _generate_rotated_frames(humans, faces, K, model, white_img, center, name, n_frames, -angle_range, 'y', unique_color, alpha, _color)
    frames_left_to_central = frames_central_to_left[::-1][1:-1]
    frames_central_to_top = _generate_rotated_frames(humans, faces, K, model, white_img, center, name, n_frames, angle_range, 'x', unique_color, alpha, _color)
    frames_top_to_central = frames_central_to_top[::-1][1:-1]

    frames = [central.astype(np.uint8) for _ in range(n_frames//4)] + \
            frames_central_to_right + \
            frames_right_to_central + \
            [central.astype(np.uint8) for _ in range(n_frames//4)] + \
            frames_central_to_left + \
            frames_left_to_central + \
            [central.astype(np.uint8) for _ in range(n_frames//4)] + \
            frames_central_to_top + \
            frames_top_to_central + \
            [central.astype(np.uint8) for _ in range(n_frames//4)]

    # Create a video from frames
    import cv2
    height, width, layers = frames[0].shape
    fps = 10
    video_path = fn
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for frame in tqdm(frames, desc=f"Writing video"):
        img = frame[:, :, ::-1]  # Convert RGB to BGR
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        video.write(img)

    video.release()
    print(f"Saved video to {video_path}")

    return fn


if __name__ == "__main__":
        parser = ArgumentParser()
        parser.add_argument("--model_name", type=str, default='multiHMR_896_L_synth')
        parser.add_argument("--img_folder", type=str, default='example_data')
        parser.add_argument("--out_folder", type=str, default='demo_out')
        parser.add_argument("--save_mesh", type=int, default=0, choices=[0,1])
        parser.add_argument("--extra_views", type=int, default=0, choices=[0,1])
        parser.add_argument("--save_rotating_video", type=int, default=0, choices=[0,1])        
        parser.add_argument("--det_thresh", type=float, default=0.3)
        parser.add_argument("--nms_kernel_size", type=float, default=3)
        parser.add_argument("--fov", type=float, default=60)
        parser.add_argument("--distance", type=int, default=0, choices=[0,1], help='add distance on the reprojected mesh')
        parser.add_argument("--unique_color", type=int, default=0, choices=[0,1], help='only one color for all humans')
        parser.add_argument("--alpha", type=float, default=1.0, help='alpha blending value for rendering')
        
        args = parser.parse_args()

        dict_args = vars(args)

        assert torch.cuda.is_available()

        is_anny = 'anny' in args.model_name.lower()

        if not is_anny:
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
        if os.path.isfile(args.img_folder) and args.img_folder.lower().endswith(suffixes):
            l_img_path = [os.path.basename(args.img_folder)]
            args.img_folder = os.path.dirname(args.img_folder)
        else:
            l_img_path = []
            img_root = os.path.abspath(args.img_folder)
            for root, _, files in os.walk(img_root):
                for fname in files:
                    if fname.lower().endswith(suffixes) and not fname.startswith('.'):
                        abs_path = os.path.join(root, fname)
                        rel_path = os.path.relpath(abs_path, img_root)
                        l_img_path.append(rel_path)
            l_img_path.sort()

        # Loading
        model = load_model(args.model_name)

        if is_anny:
            faces = model.body_model.faces.cpu().numpy()
        else:
            faces = model.smpl_layer['neutral_10'].bm_x.faces

        # Model name for saving results.
        model_name = os.path.basename(args.model_name)

        # All images
        os.makedirs(args.out_folder, exist_ok=True)
        l_duration = []
        for i, img_path in enumerate(tqdm(l_img_path)):
            # Compose save_fn: out_folder/rel_path_no_ext_modelname.png
            save_fn = os.path.join(args.out_folder, f"{img_path}_{model_name}.png")
            # Ensure output subdirectories exist
            os.makedirs(os.path.dirname(save_fn), exist_ok=True)

            # Get input in the right format for the model
            img_size = model.img_size
            x, img_pil_visu = open_image(os.path.join(args.img_folder, img_path), img_size)

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

            # Update K for rendering at full resolution
            ratio = max(img_pil_visu.size) / x.shape[-1]
            K[0, 0, 2] = img_pil_visu.size[0] / 2.0
            K[0, 1, 2] = img_pil_visu.size[1] / 2.0
            K[0, [0, 1], [0, 1]] = ratio * K[0, [0, 1], [0, 1]]
             
            pred_rend_array, _color = overlay_human_meshes(humans, faces, K, model, img_pil_visu, unique_color=args.unique_color, alpha=args.alpha)

            # Optionally add distance as an annotation to each mesh
            if args.distance:
                pred_rend_array = print_distance_on_image(pred_rend_array, humans, _color)

            # List of images too view side by side.
            l_img = [np.asarray(img_pil_visu), pred_rend_array]
            _img = np.concatenate(l_img, 1).astype(np.uint8)

            # Save to path.
            Image.fromarray(_img).save(save_fn)
            print(f"Avg Multi-HMR inference time={int(1000*np.median(np.asarray(l_duration[-1:])))}ms on a {torch.cuda.get_device_name()} ---> {save_fn}")
            sys.stdout.flush()


            # video
            if args.save_rotating_video:
                create_rotating_video(humans, faces, K, model, img_pil_visu, unique_color=args.unique_color, alpha=args.alpha, fn=save_fn.replace('.png','_rotating.mp4'), n_frames=20, angle_range=60)



            # Saving mesh
            if args.save_mesh:
                # deprecated: only for backward compatibility
                # npy file
                l_mesh = [hum['verts_smplx'].cpu().numpy() for hum in humans]
                mesh_fn = save_fn+'.npy'
                np.save(mesh_fn, np.asarray(l_mesh), allow_pickle=True)
                x = np.load(mesh_fn, allow_pickle=True)

                # glb file
                l_mesh = [humans[j]['verts_smplx'].detach().cpu().numpy() for j in range(len(humans))]
                l_face = [faces for j in range(len(humans))]
                scene = create_scene(img_pil_visu, l_mesh, l_face, color=None, metallicFactor=0., roughnessFactor=0.5)
                scene_fn = save_fn+'.glb'
                scene.export(scene_fn)

        print('end')
