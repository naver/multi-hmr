import os
import subprocess
import argparse
import torch
import numpy as np
import random
import zipfile
import time
import json
from PIL import Image, ImageOps

from utils import MEAN_PARAMS, SMPLX_DIR, normalize_rgb
from demoJson import load_model, get_camera_parameters, forward_model, open_image
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

torch.cuda.empty_cache()

np.random.seed(seed=0)
random.seed(0)

def prepare_inference(args):
    smplx_fn = os.path.join(SMPLX_DIR, 'smplx', 'SMPLX_NEUTRAL.npz')
    if not os.path.isfile(smplx_fn):
        print(f"{smplx_fn} not found, please download SMPLX_NEUTRAL.npz file")
        print("To do so you need to create an account in https://smpl-x.is.tue.mpg.de")
        print("Then download 'SMPL-X-v1.1 (NPZ+PKL, 830MB) - Use this for SMPL-X Python codebase'")
        print(f"Extract the zip file and move SMPLX_NEUTRAL.npz to {smplx_fn}")
        print("Sorry for this inconvenience but we do not have license for redistributing SMPLX model")
        raise NotImplementedError
    else:
        print('SMPLX found')
            
    if not os.path.isfile(MEAN_PARAMS):
        print('Start to download the SMPL mean params')
        os.system(f"wget -O {MEAN_PARAMS} https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smpl_mean_params.npz?versionId=CAEQHhiBgICN6M3V6xciIDU1MzUzNjZjZGNiOTQ3OWJiZTJmNThiZmY4NmMxMTM4")
        print('SMPL mean params have been successfully downloaded')
    else:
        print('SMPL mean params is already here')

    model = load_model(args.model_name)
    return model

def extract_frames(args):
    vid = args.vid
    vid_name = os.path.splitext(args.vid)[0]
    frame_folder = os.path.join(args.img_folder, vid_name)
    video_path = os.path.join(args.vid_folder, vid)
    os.makedirs(frame_folder, exist_ok=True)

    command = ['ffmpeg', '-i', video_path]

    if args.init_sec > 0:
        command.extend(['-ss', str(args.init_sec)])
    if args.duration_sec > 0:
        command.extend(['-t', str(args.duration_sec)])
    
    command.append(f"{frame_folder}/frame%05d.jpg")

    subprocess.run(command, check=True)

    fps = 30

    return frame_folder, vid_name, fps

def process_frames(args, l_frame_paths, out_folder, model, model_name):
    l_duration = []
    start_process_frames = time.time()
    for i, frame_path in enumerate(tqdm(l_frame_paths)):
        save_file_name = os.path.join(out_folder, f"{Path(frame_path).stem}_{model_name}")
        input_path = frame_path

        if not os.path.isfile(input_path):
            print(f"File {input_path} does not exist. Skipping.")
            continue

        duration, humans, resized_dims, K = infer_img(input_path, model, args)
        l_duration.append(duration)

        if args.export_json:
            params_dict = {
                "frame_id": i,
                "resized_width": resized_dims[0],
                "resized_height": resized_dims[1],
                "checkpoint_resolution": model.img_size,
                "camera_intrinsics": tensor_to_list(K[0]),
                "humans": []
            }

            for human in humans:
                human_params = {
                    "location": tensor_to_list(human['loc']),
                    "translation": tensor_to_list(human['transl']),
                    "translation_pelvis": tensor_to_list(human['transl_pelvis']),
                    "rotation_vector": tensor_to_list(human['pose']),
                    "expression": tensor_to_list(human['expression'])
                }
                params_dict["humans"].append(human_params)

            json_path = os.path.join(out_folder, f"{Path(frame_path).stem}_{model_name}_params.json")
            with open(json_path, 'w') as f:
                json.dump(params_dict, f, indent=2)

        expand_if_1d = lambda x: np.expand_dims(x, axis=0) if isinstance(x, np.ndarray) and x.ndim == 1 else x
        for j, human in enumerate(humans):
            human_out = map_human(human)
            human_dict = {k: expand_if_1d(v) for k, v in human_out.items()}
            meta_fn = save_file_name + '_' + str(j) + '.npz'
            np.savez(meta_fn, **human_dict)

    print(f"Avg Multi-HMR inference time={int(1000*np.median(np.asarray(l_duration)))}ms on a {torch.cuda.get_device_name()}")
    print(f'Total process time={time.time() - start_process_frames}')

    output_zip = out_folder + '.zip'
    zip_npz_files(out_folder, output_zip)

def map_human(human):
    global_orient = human['rotmat'][0].cpu().numpy()
    body_pose = np.concatenate([human['rotmat'][i].cpu().numpy() for i in range(1, 22)], axis=0)
    left_hand_pose = np.concatenate([human['rotmat'][i].cpu().numpy() for i in range(22, 37)], axis=0)
    right_hand_pose = np.concatenate([human['rotmat'][i].cpu().numpy() for i in range(37, 52)], axis=0)
    jaw_pose = human['rotmat'][52].cpu().numpy()
    leye_pose = human['rotmat'][52].cpu().numpy()
    reye_pose = human['rotmat'][52].cpu().numpy()
    
    human_out = {
        'global_orient': global_orient,
        'body_pose': body_pose,
        'left_hand_pose': left_hand_pose,
        'right_hand_pose': right_hand_pose,
        'jaw_pose': jaw_pose,
        'leye_pose': leye_pose,
        'reye_pose': reye_pose,
        'betas': human['shape'].cpu().numpy(),
        'expression': human['expression'].cpu().numpy(),
        'transl': human['transl'].cpu().numpy()
    }
    return human_out

def infer_img(img_path, model, args):
    img_size = model.img_size
    x, img_pil_nopad, resized_dims = open_image(img_path, img_size)
    p_x, p_y = None, None
    K = get_camera_parameters(model.img_size, fov=args.fov, p_x=p_x, p_y=p_y)
    start = time.time()
    outputs = forward_model(model, x, K,
                            det_thresh=args.det_thresh,
                            nms_kernel_size=args.nms_kernel_size)
    duration = time.time() - start

    return duration, outputs, resized_dims, K

def zip_npz_files(folder_path, output_zip):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.npz'):
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, folder_path))

    print(f"All .npz files from {folder_path} have been zipped into {output_zip}")

def tensor_to_list(tensor):
    return tensor.detach().cpu().numpy().tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid', type=str, help='name of the video (with extension)')
    parser.add_argument('--img_folder', type=str, default='/mnt/input_img')
    parser.add_argument('--vid_folder', type=str, default='/mnt/input_vid')
    parser.add_argument('--out_folder', type=str, default='/mnt/output')
    parser.add_argument('--init_sec', type=int, default=0)
    parser.add_argument('--duration_sec', type=int, default=0)
    parser.add_argument("--model_name", type=str, default='multiHMR_896_L')
    parser.add_argument("--det_thresh", type=float, default=0.3)
    parser.add_argument("--nms_kernel_size", type=float, default=3)
    parser.add_argument("--fov", type=float, default=60)
    parser.add_argument("--distance", type=int, default=0, choices=[0,1], help='add distance on the reprojected mesh')
    parser.add_argument("--inference_id", type=str)
    parser.add_argument("--export_json", type=int, default=1, choices=[0,1], help='export parameters as JSON')
    args = parser.parse_args()

    dict_args = vars(args)

    assert torch.cuda.is_available()

    assert os.path.splitext(args.vid)[1] == '.mp4', 'Only mp4 format is supported'
    
    frame_folder, vid_name, fps = extract_frames(args)
    print(f'complete to extract {vid_name} / {fps} FPS at {frame_folder}')

    suffixes = ('.jpg', '.jpeg', '.png', '.webp')
    list_input_path = [os.path.join(frame_folder, file) for file in os.listdir(frame_folder) if file.endswith(suffixes) and file[0] != '.']
    assert len(list_input_path) > 0, 'No frames to infer'
    print(f'The number of images to infer: {len(list_input_path)}')

    if args.inference_id:
        inference_id = args.inference_id
    else:
        inference_id = vid_name
    out_folder = f'{args.out_folder}/{inference_id}'
    os.makedirs(out_folder, exist_ok=True)

    meta_data = {
        "fps": fps
    }
    meta_path = os.path.join(out_folder, 'meta.json')
    with open(meta_path, "w") as meta_file:
        json.dump(meta_data, meta_file)
    
    model = prepare_inference(args)
    print(f'complete to preparing {args.model_name} inference')

    process_frames(args, list_input_path, out_folder, model, args.model_name)
    print(f'complete to process {vid_name} at {out_folder}')