# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

"""
Server:
    CUDA_VISIBLE_DEVICES="0" python app.py

Laptop:
    ssh -N -L 8000:127.0.0.1:7860 my_server

Then open http://localhost:8000/
"""
import spaces
import os
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

from utils.constants import SMPLX_DIR, MEAN_PARAMS
from argparse import ArgumentParser
import torch
import gradio as gr
from PIL import Image, ImageOps
import numpy as np
from pathlib import Path

if torch.cuda.is_available() and torch.cuda.device_count()>0:
    device = torch.device('cuda:0')
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    device_name = torch.cuda.get_device_name(0)
    print(f"Device - GPU: {device_name}")
else:
    device = torch.device('cpu')
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    device_name = 'CPU'
    print("Device - CPU")

from demo import forward_model, get_camera_parameters, overlay_human_meshes, load_model as _load_model
from utils import normalize_rgb, demo_color as color, create_scene
import time

model = None
example_data_dir = 'example_data'
list_examples = os.listdir(example_data_dir)
list_examples_basename = [x for x in list_examples if x.endswith(('.jpg', 'jpeg', 'png')) and not x.startswith('._')]
list_examples = [[os.path.join(example_data_dir, x)] for x in list_examples_basename]
_list_examples_basename = [Path(x).stem for x in list_examples_basename]
tmp_data_dir = 'tmp_data'
model_name = 'none'

def download_smplx():
    os.makedirs(os.path.join(SMPLX_DIR, 'smplx'), exist_ok=True)
    smplx_fname = os.path.join(SMPLX_DIR, 'smplx', 'SMPLX_NEUTRAL.npz')

    if not os.path.isfile(smplx_fname):
        print('Start to download the SMPL-X model')
        if not ('SMPLX_LOGIN' in os.environ and 'SMPLX_PWD' in os.environ):
                raise ValueError('You need to set a secret for SMPLX_LOGIN and for SMPLX_PWD to run this space')
        fname = "models_smplx_v1_1.zip"
        username = os.environ['SMPLX_LOGIN'].replace('@','%40')
        password = os.environ['SMPLX_PWD']
        cmd = f"wget -O {fname} --save-cookies cookies.txt --keep-session-cookies --post-data 'username={username}&password={password}' \"https://download.is.tue.mpg.de/download.php?domain=smplx&sfile={fname}\""
        os.system(cmd)
        assert os.path.isfile(fname), "failed to download"
        os.system(f'unzip {fname}')
        os.system(f"cp models/smplx/SMPLX_NEUTRAL.npz {smplx_fname}")
        assert os.path.isfile(smplx_fname), "failed to find smplx file"
        print('SMPL-X has been succesfully downloaded')
    else:
         print('SMPL-X is already here')

    if not os.path.isfile(MEAN_PARAMS):
        print('Start to download the SMPL mean params')
        os.system(f"wget -O {MEAN_PARAMS}  https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smpl_mean_params.npz?versionId=CAEQHhiBgICN6M3V6xciIDU1MzUzNjZjZGNiOTQ3OWJiZTJmNThiZmY4NmMxMTM4")
        print('SMPL mean params have been succesfully downloaded')
    else:
         print('SMPL mean params is already here')

@spaces.GPU
def infer(fn, det_thresh, nms_kernel_size, fov):
    global device
    global model
    
    # Is it an image from example_data_dir ?
    basename = Path(os.path.basename(fn)).stem
    _basename = f"{basename}_{model_name}_thresh{int(det_thresh*100)}_nms{int(nms_kernel_size)}_fov{int(fov)}"
    is_known_image = (basename in _list_examples_basename) # only images from example_data
    
    # Filenames
    if not is_known_image:
        _basename = 'output' # such that we do not save all the uploaded results - not sure ?
    _glb_fn = f"{_basename}.glb"
    _rend_fn = f"{_basename}.png"
    glb_fn = os.path.join(tmp_data_dir, _glb_fn)
    rend_fn = os.path.join(tmp_data_dir, _rend_fn)
    os.makedirs(tmp_data_dir, exist_ok=True)

    # Already processed
    is_preprocessed = False
    if is_known_image:
        _tmp_data_dir_files = os.listdir(tmp_data_dir)
        is_preprocessed = (_glb_fn in _tmp_data_dir_files) and (_rend_fn in _tmp_data_dir_files) # already preprocessed

    is_known = is_known_image and is_preprocessed
    if not is_known:
        im = Image.open(fn)
        fov, p_x, p_y = fov, None, None # FOV=60 always here!
        img_size = model.img_size

        # Get camera information
        p_x, p_y = None, None
        K = get_camera_parameters(img_size, fov=fov, p_x=p_x, p_y=p_y, device=device)

        # Resise but keep aspect ratio
        img_pil = ImageOps.contain(im, (img_size,img_size)) # keep the same aspect ratio

        # Which side is too small/big
        width, height = img_pil.size
        pad = abs(width - height) // 2

        # Pad
        img_pil_bis = ImageOps.pad(img_pil.copy(), size=(img_size, img_size), color=(255, 255, 255))
        img_pil = ImageOps.pad(img_pil, size=(img_size, img_size)) # pad with zero on the smallest side

        # Numpy - normalize - torch.
        resize_img = normalize_rgb(np.asarray(img_pil))
        x = torch.from_numpy(resize_img).unsqueeze(0).to(device)

        img_array = np.asarray(img_pil_bis)
        img_pil_visu = Image.fromarray(img_array)

        start = time.time()
        humans = forward_model(model, x, K, det_thresh=det_thresh, nms_kernel_size=nms_kernel_size)
        print(f"Forward: {time.time() - start:.2f}sec")

        # Overlay
        start = time.time()
        pred_rend_array, _ = overlay_human_meshes(humans, K, model, img_pil_visu)
        rend_pil = Image.fromarray(pred_rend_array.astype(np.uint8))
        rend_pil.crop()
        if width > height:
            rend_pil = rend_pil.crop((0,pad,width,pad+height))
        else:
            rend_pil =rend_pil.crop((pad,0,pad+width,height))
        rend_pil.save(rend_fn)
        print(f"Rendering with pyrender: {time.time() - start:.2f}sec")

        # Save into glb
        start = time.time()
        l_mesh = [humans[j]['v3d'].detach().cpu().numpy() for j in range(len(humans))]
        l_face = [model.smpl_layer['neutral_10'].bm_x.faces for j in range(len(humans))]
        scene = create_scene(img_pil_visu, l_mesh, l_face, color=color, metallicFactor=0., roughnessFactor=0.5)
        scene.export(glb_fn)
        print(f"Exporting scene in glb: {time.time() - start:.2f}sec")
    else:
        print("We already have the predictions-visus stored somewhere...")
    
    out = [rend_fn, glb_fn]
    print(out)
    return out

     
if __name__ == "__main__":
        parser = ArgumentParser()
        parser.add_argument("--model_name", type=str, default='multiHMR_896_L_synth')
        
        args = parser.parse_args()

        # Info
        ### Description and style
        logo = r"""
        <center>
            <img src='https://europe.naverlabs.com/wp-content/uploads/2020/10/NLE_1_WHITE_264x60_opti.png' alt='Multi-HMR logo' style="width:250px; margin-bottom:10px">
        </center>
        """
        title = r"""
        <center>
            <h1 align="center">Multi-HMR: Regressing Whole-Body Human Meshes for Multiple Persons in a Single Shot</h1>
        </center>
        """

        description = f"""
        The demo is running on a {device_name}.
        <br>
        [<b>Demo code</b>] If you want to run Multi-HMR on several images please consider using the demo code available on [our Github repo](https://github.com/naver/multiHMR)
        """

        article = r"""
        ---
        📝 **Citation**
        <br>
        If our work is useful for your research, please consider citing:
        ```bibtex
        @inproceedings{multihmr2024,
            title={Multi-HMR: Regressing Whole-Body Human Meshes for Multiple Persons in a Single Shot},
            author={Baradel*, Fabien and 
                    Armando, Matthieu and 
                    Galaaoui, Salma and 
                    Br{\'e}gier, Romain and
                    Weinzaepfel, Philippe and 
                    Rogez, Gr{\'e}gory and 
                    Lucas*, Thomas},
            booktitle={arXiv},
            year={2024}
        }
        ```
        📋 **License**
        <br>
        CC BY-NC-SA 4.0 License. Please refer to the [LICENSE file](./Multi-HMR_License.txt) for details.
        <br>
        📧 **Contact**
        <br>
        If you have any questions, please feel free to send a message to <b>fabien.baradel@naverlabs.com</b> or open an issue on the [Github repo](https://github.com/naver/multi-hmr).
        """

        # Download SMPLX model and mean params
        download_smplx()

        # Loading the model
        model = _load_model(args.model_name, device=device)
        model_name = args.model_name
        
        # Gradio demo
        with gr.Blocks(title="Multi-HMR", css=".gradio-container") as demo:
            gr.Markdown(logo)
            gr.Markdown(title)
            gr.Markdown(description)

            with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(label="Input image", 
                                            #    type="pil", 
                                               type="filepath", 
                                               sources=['upload', 'clipboard'])
                    with gr.Column():
                        output_image = gr.Image(label="Reconstructions - Overlay", 
                                                #    type="pil", 
                                               type="filepath",
                                                )

            gr.HTML("""<br/>""")

            with gr.Row():
                with gr.Column():
                        alpha = -70 # longitudinal rotation in degree
                        beta = 70 # latitudinal rotation in degree
                        radius = 3. # distance to the 3D model
                        radius = None # distance to the 3D model
                        output_model3d = gr.Model3D(label="Reconstructions - 3D scene", 
                                                    camera_position=(alpha, beta, radius), 
                                                    clear_color=[1.0, 1.0, 1.0, 0.0])
                
            gr.HTML("""<br/>""")

            with gr.Row():
                    threshold = gr.Slider(0.1, 0.7, step=0.1, value=0.3, label='Detection Threshold')
                    nms = gr.Radio(label="NMS kernel size", choices=[1, 3, 5], value=3)
                    fov = gr.Radio(label="FOV", choices=[50, 60, 70], value=60)
                    send_btn = gr.Button("Infer")
                    send_btn.click(fn=infer, inputs=[input_image, threshold, nms, fov], outputs=[output_image, output_model3d])

            gr.Examples(list_examples, 
                        inputs=[input_image, 0.3, 3])

            gr.Markdown(article)

        demo.queue()  # <-- Sets up a queue with default parameters
        demo.launch(debug=False, share=False)


