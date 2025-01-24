import torch
import glob
import numpy as np
import os
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    TexturesUV,
)
from smplx import SMPL


# Load AMASS poses from SMPLX-N .npz files and extract SMPL part
def load_amass_smpl_poses(amass_dir, num_poses=10):
    pose_list = []
    count = 0
    for path in glob.glob(os.path.join(amass_dir, "*/*.npz")):
        count += 1
        data = np.load(path)
        poses = data.get("poses", [])  # shape [N, 165] for SMPLX
        betas = data.get("betas", [])  # shape [10]
        for i in range(min(len(poses), 100)):
            smpl_pose = poses[i][:72]  # extract SMPL-compatible pose
            pose_list.append(
                {
                    "global_orient": torch.tensor(smpl_pose[:3], dtype=torch.float32),
                    "body_pose": torch.tensor(smpl_pose[3:], dtype=torch.float32),
                    "betas": torch.tensor(betas[:10], dtype=torch.float32),
                }
            )
            if len(pose_list) == num_poses:
                break
        if len(pose_list) == num_poses:
            break
    return pose_list


# SMPL + renderer
def load_smpl(model_folder, device="cpu"):
    model = SMPL(
        model_path=model_folder,
        gender="NEUTRAL",
        use_pca=False,
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_transl=False,
    ).to(device)

    faces = torch.tensor(model.faces.astype(np.int64), device=device)
    return model, faces


def make_matrix(R, rads, axis="y"):
    assert R.shape[0] == len(rads)
    T = R.permute((1, 2, 0))

    # x axis
    if axis == "x":
        T[1, 1, :] = torch.cos(rads)
        T[1, 2, :] = -torch.sin(rads)
        T[2, 1, :] = torch.sin(rads)
        T[2, 2, :] = torch.cos(rads)

    # y axis
    elif axis == "y":
        T[0, 0, :] = torch.cos(rads)
        T[0, 2, :] = torch.sin(rads)
        T[2, 0, :] = -torch.sin(rads)
        T[2, 2, :] = torch.cos(rads)
    # z axis
    else:
        T[0, 0, :] = torch.cos(rads)
        T[0, 1, :] = -torch.sin(rads)
        T[1, 0, :] = torch.sin(rads)
        T[1, 1, :] = torch.cos(rads)

    T = T.permute(2, 0, 1)
    return T


# return batch rotation matrices: Bx3x3
def rotation_matrix(x_degs, y_degs, z_degs):
    B = x_degs.shape[0]
    # Convert degrees to radians
    x_rads = torch.deg2rad(x_degs)
    y_rads = torch.deg2rad(y_degs)
    z_rads = torch.deg2rad(z_degs)

    Rx = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)
    Ry = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)
    Rz = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)

    # Rotation matrix around X-axis
    Rx = make_matrix(Rx, x_rads, axis="x")

    # Rotation matrix around Y-axis
    Ry = make_matrix(Ry, y_rads, axis="y")

    # Rotation matrix around Z-axis
    Rz = make_matrix(Rz, z_rads, axis="z")

    # Final combined rotation matrix: R = Rz @ Ry @ Rx
    R = Rz @ Ry @ Rx
    return R


def get_random_cams(frames, device="cpu"):
    focal_len = 3.3
    tx = 0.0
    ty = 0.3
    tz = 3.0
    angles = torch.linspace(0, 359, frames, dtype=torch.float32)
    # ro_y = angles[torch.multinomial(angles, frames, replacement=True)]
    ro_y = angles
    ro_x = torch.zeros_like(ro_y)
    ro_z = torch.zeros_like(ro_y)
    Ro = rotation_matrix(ro_x, ro_y, ro_z)
    Tr = torch.tensor([tx, ty, tz]).unsqueeze(0).repeat(frames, 1)
    cameras = PerspectiveCameras(
        focal_length=((focal_len, focal_len),),
        R=Ro.to(device),
        T=Tr.to(device),
        device=device,
    )
    return cameras


def setup_renderer(frames=30, img_size=(512, 512), device="cpu"):
    cameras = get_random_cams(frames, device=device)
    raster_settings = RasterizationSettings(
        image_size=img_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    lights = PointLights(location=[[0.0, 2.0, 2.0]] * frames, device=device)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )
    return renderer


def load_uv_from_obj(obj_path, device="cpu"):
    verts, faces, aux = load_obj(obj_path, load_textures=True)
    verts_uvs = aux.verts_uvs.to(device)  # [N_uv_verts, 2]
    faces_uvs = faces.textures_idx.to(device)  # [F, 3]
    return verts_uvs, faces_uvs
