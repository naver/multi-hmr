# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import torch
import numpy as np
import pyrender
import trimesh
import math
from scipy.spatial.transform import Rotation
from PIL import ImageFont, ImageDraw, Image

OPENCV_TO_OPENGL_CAMERA_CONVENTION = np.array([[1, 0, 0, 0],
                                               [0, -1, 0, 0],
                                               [0, 0, -1, 0],
                                               [0, 0, 0, 1]])

def geotrf( Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.
    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)
    
    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.
    
    Returns an array of projected 2d points.
    """
    assert Trf.ndim in (2,3)
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    ncol = ncol or pts.shape[-1]

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    if Trf.ndim == 3:
        assert len(Trf) == len(pts), 'batch size does not match'
    if Trf.ndim == 3 and pts.ndim > 3:
        # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
        pts = pts.reshape(pts.shape[0], -1, pts.shape[-1])
    elif Trf.ndim == 3 and pts.ndim == 2:
        # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
        pts = pts[:, None, :]

    if pts.shape[-1]+1 == Trf.shape[-1]:
        Trf = Trf.swapaxes(-1,-2) # transpose Trf
        pts = pts @ Trf[...,:-1,:] + Trf[...,-1:,:]
    elif pts.shape[-1] == Trf.shape[-1]:
        Trf = Trf.swapaxes(-1,-2) # transpose Trf
        pts = pts @ Trf
    else:
        pts = Trf @ pts.T
        if pts.ndim >= 2: pts = pts.swapaxes(-1,-2)
    if norm: 
        pts = pts / pts[...,-1:] # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1: pts *= norm

    return pts[...,:ncol].reshape(*output_reshape, ncol)

def create_scene(img_pil, l_mesh, l_face, color=None, metallicFactor=0., roughnessFactor=0.5, focal=600):
    
    scene = trimesh.Scene(
        lights=trimesh.scene.lighting.Light(intensity=3.0)
    )

    # Human meshes
    for i, mesh in enumerate(l_mesh):
        if color is None:
            _color = (np.random.choice(range(1,225))/255, np.random.choice(range(1,225))/255, np.random.choice(range(1,225))/255)
        else:
            if isinstance(color,list):
                _color = color[i]
            elif isinstance(color,tuple):
                _color = color
            else:
                raise NotImplementedError
        mesh = trimesh.Trimesh(mesh, l_face[i])
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=None, 
            material=trimesh.visual.material.PBRMaterial(
              metallicFactor=metallicFactor,
              roughnessFactor=roughnessFactor,
              alphaMode='OPAQUE',
              baseColorFactor=(_color[0], _color[1], _color[2], 1.0)
            ),
            image=None, 
            face_materials=None
        )
        scene.add_geometry(mesh)

    # Image
    H, W = img_pil.size[0], img_pil.size[1]
    screen_width = 0.3
    height = focal * screen_width / H
    width = screen_width * 0.5**0.5
    rot45 = np.eye(4)
    rot45[:3,:3] = Rotation.from_euler('z',np.deg2rad(45)).as_matrix()
    rot45[2,3] = -height # set the tip of the cone = optical center
    aspect_ratio = np.eye(4)
    aspect_ratio[0,0] = W/H
    transform = OPENCV_TO_OPENGL_CAMERA_CONVENTION @ aspect_ratio @ rot45
    cam = trimesh.creation.cone(width, height, sections=4, transform=transform)
    # cam.apply_transform(transform)
    # import ipdb
    # ipdb.set_trace()

    # vertices = geotrf(transform, cam.vertices[[4,5,1,3]])
    vertices = cam.vertices[[4,5,1,3]]
    faces = np.array([[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]])
    img = trimesh.Trimesh(vertices=vertices, faces=faces)
    uv_coords = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    # img_pil = Image.fromarray((255. * np.ones((20,20,3))).astype(np.uint8)) # white only!
    material = trimesh.visual.texture.SimpleMaterial(image=img_pil,
                                                     diffuse=[255,255,255,0], 
                                                     ambient=[255,255,255,0], 
                                                     specular=[255,255,255,0], 
                                                     glossiness=1.0)
    img.visual = trimesh.visual.TextureVisuals(uv=uv_coords, image=img_pil) #, material=material)
    # _main_color = [255,255,255,0]
    # print(img.visual.material.ambient)
    # print(img.visual.material.diffuse)
    # print(img.visual.material.specular)
    # print(img.visual.material.main_color)

    # img.visual.material.ambient = _main_color
    # img.visual.material.diffuse = _main_color
    # img.visual.material.specular = _main_color

    # img.visual.material.main_color = _main_color
    # img.visual.material.glossiness = _main_color
    scene.add_geometry(img)

    # this is the camera mesh
    rot2 = np.eye(4)
    rot2[:3,:3] = Rotation.from_euler('z',np.deg2rad(2)).as_matrix()
    # import ipdb
    # ipdb.set_trace()
    # vertices = cam.vertices
    # print(rot2)
    vertices = np.r_[cam.vertices, 0.95*cam.vertices, geotrf(rot2, cam.vertices)]
    # vertices = np.r_[cam.vertices, 0.95*cam.vertices, 1.05*cam.vertices]
    faces = []
    for face in cam.faces:
        if 0 in face: continue
        a,b,c = face
        a2,b2,c2 = face + len(cam.vertices)
        a3,b3,c3 = face + 2*len(cam.vertices)

        # add 3 pseudo-edges
        faces.append((a,b,b2))
        faces.append((a,a2,c))
        faces.append((c2,b,c))

        faces.append((a,b,b3))
        faces.append((a,a3,c))
        faces.append((c3,b,c))

    # no culling
    faces += [(c,b,a) for a,b,c in faces]

    cam = trimesh.Trimesh(vertices=vertices, faces=faces)
    cam.visual.face_colors[:,:3] = (255, 0, 0)
    scene.add_geometry(cam)
    
    # OpenCV to OpenGL
    rot = np.eye(4)
    cams2world = np.eye(4)
    rot[:3,:3] = Rotation.from_euler('y',np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world @ OPENCV_TO_OPENGL_CAMERA_CONVENTION @ rot))

    return scene

def render_meshes(img, l_mesh, l_face, cam_param, color=None, alpha=1.0, 
                  show_camera=False,
                  intensity=3.0,
                  metallicFactor=0., roughnessFactor=0.5, smooth=True,
                  ):
    """
    Rendering multiple mesh and project then in the initial image.
    Args:
        - img: np.array [w,h,3]
        - l_mesh: np.array list of [v,3]
        - l_face: np.array list of [f,3]
        - cam_param: info about the camera intrinsics (focal, princpt) and (R,t) is possible
    Return:
        - img: np.array [w,h,3]
    """
    # scene
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))

    # mesh
    for i, mesh in enumerate(l_mesh):
        if color is None:
            _color = (np.random.choice(range(1,225))/255, np.random.choice(range(1,225))/255, np.random.choice(range(1,225))/255)
        else:
            if isinstance(color,list):
                _color = color[i]
            elif isinstance(color,tuple):
                _color = color
            else:
                raise NotImplementedError
        mesh = trimesh.Trimesh(mesh, l_face[i])
        
        # import ipdb
        # ipdb.set_trace()

        # mesh.visual = trimesh.visual.TextureVisuals(
        #     uv=None, 
        #     material=trimesh.visual.material.PBRMaterial(
        #         metallicFactor=metallicFactor,
        #     roughnessFactor=roughnessFactor,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(_color[0], _color[1], _color[2], 1.0)
        #     ),
        #     image=None, 
        #     face_materials=None
        # )
        # print('saving')
        # mesh.export('human.obj')
        # mesh = trimesh.load('human.obj')
        # print('loading')
        # mesh = pyrender.Mesh.from_trimesh(mesh, smooth=smooth)

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=metallicFactor,
            roughnessFactor=roughnessFactor,
            alphaMode='OPAQUE',
            baseColorFactor=(_color[0], _color[1], _color[2], 1.0))
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=smooth)
        scene.add(mesh, f"mesh_{i}")

    # Adding coordinate system at (0,0,2) for the moment
    # Using lines defined in pyramid https://docs.pyvista.org/version/stable/api/utilities/_autosummary/pyvista.Pyramid.html
    if show_camera:
        import pyvista

        def get_faces(x):
            return x.faces.astype(np.uint32).reshape((x.n_faces, 4))[:, 1:]
        
        # Camera = Box + Cone (or Cylinder?)
        material_cam = pyrender.MetallicRoughnessMaterial(metallicFactor=metallicFactor, roughnessFactor=roughnessFactor, alphaMode='OPAQUE', baseColorFactor=(0.5,0.5,0.5))
        height = 0.2
        radius = 0.1
        cone = pyvista.Cone(center=(0.0, 0.0, -height/2), direction=(0.0, 0.0, -1.0), height=height, radius=radius).extract_surface().triangulate()
        verts = cone.points
        mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(verts, get_faces(cone)), material=material_cam, smooth=smooth)
        scene.add(mesh, f"cone")

        size = 0.1
        box = pyvista.Box(bounds=(-size, size, 
                                  -size, size, 
                                  verts[:,-1].min() - 3*size, verts[:,-1].min())).extract_surface().triangulate()
        verts = box.points
        mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(verts, get_faces(box)), material=material_cam, smooth=smooth)
        scene.add(mesh, f"box")
        

        # Coordinate system
        # https://docs.pyvista.org/version/stable/api/utilities/_autosummary/pyvista.Arrow.html
        l_color = [(1,0,0,1.0), (0,1,0,1.0), (0,0,1,1.0)]
        l_direction = [(1,0,0), (0,1,0), (0,0,1)]
        scale = 0.2
        pose3d = [2*scale, 0.0, -scale]
        for i in range(len(l_color)):
            arrow = pyvista.Arrow(direction=l_direction[i], scale=scale)
            arrow = arrow.extract_surface().triangulate()
            verts = arrow.points + np.asarray([pose3d])
            faces = arrow.faces.astype(np.uint32).reshape((arrow.n_faces, 4))[:, 1:]
            mesh = trimesh.Trimesh(verts, faces)
            material = pyrender.MetallicRoughnessMaterial(metallicFactor=metallicFactor, roughnessFactor=roughnessFactor, alphaMode='OPAQUE', baseColorFactor=l_color[i])
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=smooth)
            scene.add(mesh, f"arrow_{i}")
    
    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera_pose = np.eye(4)
    if 'R' in cam_param.keys():
        camera_pose[:3, :3] = cam_param['R']
    if 't' in cam_param.keys():
        camera_pose[:3, 3] = cam_param['t']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    
    # camera
    camera_pose = OPENCV_TO_OPENGL_CAMERA_CONVENTION @ camera_pose
    camera_pose = np.linalg.inv(camera_pose)
    scene.add(camera, pose=camera_pose)
 
    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)
   
    # light
    light = pyrender.DirectionalLight(intensity=intensity)
    scene.add(light, pose=camera_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:,:,:3].astype(np.float32)
    fg = (depth > 0)[:,:,None].astype(np.float32)

    # Simple smoothing of the mask
    bg_blending_radius = 1
    bg_blending_kernel = 2.0 * torch.ones((1, 1, 2 * bg_blending_radius + 1, 2 * bg_blending_radius + 1)) / (2 * bg_blending_radius + 1) ** 2
    bg_blending_bias =  -torch.ones(1)
    fg = fg.reshape((fg.shape[0],fg.shape[1]))
    fg = torch.from_numpy(fg).unsqueeze(0)
    fg = torch.clamp_min(torch.nn.functional.conv2d(fg, weight=bg_blending_kernel, bias=bg_blending_bias, padding=bg_blending_radius) * fg, 0.0)
    fg = fg.permute(1,2,0).numpy()

    # Alpha-blending
    img = (fg * (alpha * rgb + (1.0-alpha) * img) + (1-fg) * img).astype(np.uint8)

    renderer.delete()

    return img.astype(np.uint8)

def length(v):
    return math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])

def cross(v0, v1):
    return [
        v0[1]*v1[2]-v1[1]*v0[2],
        v0[2]*v1[0]-v1[2]*v0[0],
        v0[0]*v1[1]-v1[0]*v0[1]]

def dot(v0, v1):
    return v0[0]*v1[0]+v0[1]*v1[1]+v0[2]*v1[2]

def normalize(v, eps=1e-13):
    l = length(v)
    return [v[0]/(l+eps), v[1]/(l+eps), v[2]/(l+eps)]

def lookAt(eye, target, *args, **kwargs):
    """
    eye is the point of view, target is the point which is looked at and up is the upwards direction.

    Input should be in OpenCV format - we transform arguments to OpenGL
    Do compute in OpenGL and then transform back to OpenCV

    """
    # Transform from OpenCV to OpenGL format
    # eye = [eye[0], -eye[1], -eye[2]]
    # target = [target[0], -target[1], -target[2]]
    up = [0,-1,0]

    eye, at, up = eye, target, up
    zaxis = normalize((at[0]-eye[0], at[1]-eye[1], at[2]-eye[2]))
    xaxis = normalize(cross(zaxis, up))
    yaxis = cross(xaxis, zaxis)

    zaxis = [-zaxis[0],-zaxis[1],-zaxis[2]]

    viewMatrix = np.asarray([
        [xaxis[0], xaxis[1], xaxis[2], -dot(xaxis, eye)],
        [yaxis[0], yaxis[1], yaxis[2], -dot(yaxis, eye)],
        [zaxis[0], zaxis[1], zaxis[2], -dot(zaxis, eye)],
        [0, 0, 0, 1]]
    ).reshape(4,4)

    # OpenGL to OpenCV
    viewMatrix = OPENCV_TO_OPENGL_CAMERA_CONVENTION @ viewMatrix
    
    return viewMatrix

def print_distance_on_image(pred_rend_array, humans, _color):
    # Add distance to the image.
    font = ImageFont.load_default()
    rend_pil = Image.fromarray(pred_rend_array)
    draw = ImageDraw.Draw(rend_pil)
    for i_hum, hum in enumerate(humans):
            # distance
            transl = hum['transl_pelvis'].cpu().numpy().reshape(3)
            dist_cam = np.sqrt(((transl[[0,2]])**2).sum()) # discarding Y axis
            # 2d - bbox
            bbox = get_bbox(hum['j2d_smplx'].cpu().numpy(), factor=1.35, output_format='x1y1x2y2')
            loc = [(bbox[0] + bbox[2]) / 2., bbox[1]]
            txt = f"{dist_cam:.2f}m"
            length = font.getlength(txt)
            loc[0] = loc[0] - length // 2
            fill = tuple((np.asarray(_color[i_hum]) * 255).astype(np.int32).tolist())
            draw.text((loc[0], loc[1]), txt, fill=fill, font=font)
    return np.asarray(rend_pil)

def get_bbox(points, factor=1., output_format='xywh'):
    """
    Args:
        - y: [k,2]
    Return:
        - bbox: [4] in a specific format
    """
    assert len(points.shape) == 2, f"Wrong shape, expected two-dimensional array. Got shape {points.shape}"
    assert points.shape[1] == 2
    x1, x2 = points[:,0].min(), points[:,0].max()
    y1, y2 = points[:,1].min(), points[:,1].max()
    cx, cy = (x2 + x1) / 2., (y2 + y1) / 2.
    sx, sy = np.abs(x2 - x1), np.abs(y2 - y1)
    sx, sy = int(factor * sx), int(factor * sy)
    x1, y1 = int(cx - sx / 2.), int(cy - sy / 2.)
    x2, y2 = int(cx + sx / 2.), int(cy + sy / 2.)
    if output_format == 'xywh':
        return [x1,y1,sx,sy]
    elif output_format == 'x1y1x2y2':
        return [x1,y1,x2,y2]
    else:
        raise NotImplementedError

def render_side_views(img_array, _color, humans, model, K):
    _bg = 255. # white

    # camera
    focal = np.asarray([K[0,0,0].cpu().numpy(),K[0,1,1].cpu().numpy()])
    princpt = np.asarray([K[0,0,-1].cpu().numpy(),K[0,1,-1].cpu().numpy()])

    # Get the vertices produced by the model.
    l_verts = [humans[j]['v3d'].cpu().numpy() for j in range(len(humans))]
    l_faces = [model.smpl_layer['neutral_10'].bm_x.faces for j in range(len(humans))]

    bg_array = 1 + 0.*img_array.copy()
    if len(humans) == 0:
        pred_rend_array_bis = _bg * bg_array.copy()
        pred_rend_array_sideview = _bg * bg_array.copy()
        pred_rend_array_bev = _bg * bg_array.copy()
    else:
        # Small displacement
        H_bis = lookAt(eye=[2.,-1,-2], target=[0,0,3])
        pred_rend_array_bis = render_meshes(_bg* bg_array.copy(), l_verts, l_faces, 
                                            {'focal': focal, 'princpt': princpt, 'R': H_bis[:3,:3], 't': H_bis[:3,3]}, 
                                            alpha=1.0, color=_color, show_camera=True)
        
        # Where to look at
        l_z = []
        for hum in humans:
            l_z.append(hum['transl_pelvis'].cpu().numpy().reshape(-1)[-1])
        target_z = np.median(np.asarray(l_z))

        # Sideview
        H_sideview = lookAt(eye=[2.2*target_z,0,target_z], target=[0,0,target_z])
        pred_rend_array_sideview = render_meshes(_bg*bg_array.copy(), l_verts, l_faces, 
                                                 {'focal': focal, 'princpt': princpt, 'R': H_sideview[:3,:3], 't': H_sideview[:3,3]},  
                                                 alpha=1.0, color=_color, show_camera=True)

        # Bird-Eye-View
        H_bev = lookAt(eye=[0.,-2*target_z,target_z-0.001], target=[0,0,target_z])
        pred_rend_array_bev = render_meshes(_bg* bg_array.copy(), l_verts, l_faces, 
                                            {'focal': focal, 'princpt': princpt, 'R': H_bev[:3,:3], 't': H_bev[:3,3]}, 
                                            alpha=1.0, color=_color, show_camera=True)
        
    return pred_rend_array_bis, pred_rend_array_sideview, pred_rend_array_bev