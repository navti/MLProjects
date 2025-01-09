# build_i2uv_dataset.py

import os
import random
import glob
from PIL import Image
from datasets import Dataset
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from datasets.arrow_writer import ArrowWriter
from datasets import load_dataset, Dataset, DatasetInfo, Features, Value, Image
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
from dataset import HuggingFaceATLAS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
PROMPTS_FILE = os.path.abspath("./data/atlas_prompt_50k.txt")
ATLAS_DATASET_DIR = os.path.expanduser("~/huggingface/hf_datasets")
SMPL_MODEL_PATH = os.path.expanduser("~/datasets/smpl/models")
SMPL_TEMPLATE_OBJ_PATH = os.path.expanduser(
    "~/datasets/smpl/models/smpl_uv_template.obj"
)
AMASS_DIR = os.path.expanduser("~/datasets/amass/CMU")
ATLAS_LARGE_DIR = os.path.expanduser("/mnt/ssdp3/datasets/atlas_large")
ATLAS_LARGE_TMP_DIR = os.path.expanduser("~/huggingface/hf_datasets/atlas_tmp")
BATCH_SIZE = 16

POSES_PER_TEXTURE = 5
IMAGE_SIZE = 1024
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 768
RADIUS = 2.7
FOCAL_LENGTH = 500.0


# Transforms
to_tensor = T.ToTensor()
to_pil = T.ToPILImage()
resize = T.Resize((IMAGE_SIZE, IMAGE_SIZE))


# load prompts file
def load_prompts(prompts_file):
    with open(prompts_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


# Load AMASS poses from SMPLX-N .npz files and extract SMPL part
def load_amass_smpl_poses(amass_dir):
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
    return pose_list


# SMPL + renderer
def load_smpl(model_folder):
    model = SMPL(
        model_path=model_folder,
        gender="NEUTRAL",
        use_pca=False,
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_transl=False,
    ).to(DEVICE)

    faces = torch.tensor(model.faces.astype(np.int64), device=DEVICE)
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


def get_random_cams(B):
    focal_len = 3.3
    tx = 0.0
    ty = 0.2
    tz = 3.0
    angles = torch.arange(0, 360, 60, dtype=torch.float32)
    ro_y = angles[torch.multinomial(angles, B, replacement=True)]
    ro_x = torch.zeros_like(ro_y)
    ro_z = torch.zeros_like(ro_y)
    Ro = rotation_matrix(ro_x, ro_y, ro_z)
    Tr = torch.tensor([tx, ty, tz]).unsqueeze(0).repeat(B, 1)
    cameras = PerspectiveCameras(
        focal_length=((focal_len, focal_len),),
        R=Ro.to(DEVICE),
        T=Tr.to(DEVICE),
        device=DEVICE,
    )
    return cameras


def setup_renderer(B):
    cameras = get_random_cams(B)
    raster_settings = RasterizationSettings(
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    lights = PointLights(location=[[0.0, 2.0, 2.0]] * B, device=DEVICE)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=DEVICE, cameras=cameras, lights=lights),
    )
    return renderer


def load_uv_from_obj(obj_path):
    verts, faces, aux = load_obj(obj_path, load_textures=True)
    verts_uvs = aux.verts_uvs.to(DEVICE)  # [N_uv_verts, 2]
    faces_uvs = faces.textures_idx.to(DEVICE)  # [F, 3]
    return verts_uvs, faces_uvs


# Main
if __name__ == "__main__":
    # features of atlas large dataset
    B = BATCH_SIZE
    features = Features(
        {
            "image": Image(),  # RGB image (rendered)
            "texture": Image(),  # UV texture
            "prompt": Value("string"),  # New: prompt string
        }
    )

    os.makedirs(ATLAS_LARGE_TMP_DIR, exist_ok=True)
    writer = ArrowWriter(path=ATLAS_LARGE_TMP_DIR + "/data.arrow", features=features)

    print("Loading ATLAS dataset...")
    # Dataset and dataloader
    atlas_dataset = HuggingFaceATLAS(
        split="train",
        prompts_file=PROMPTS_FILE,
        cache_dir=ATLAS_DATASET_DIR,
        resize_dim=IMAGE_SIZE,
    )
    atlas_loader = DataLoader(atlas_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Loading AMASS poses...")
    pose_list = load_amass_smpl_poses(AMASS_DIR)
    print(f"Loaded {len(pose_list)} poses.")

    print("Preparing SMPL and renderer...")
    smpl_model, faces = load_smpl(SMPL_MODEL_PATH)
    faces = faces.unsqueeze(0).repeat(B, 1, 1)
    verts_uvs, faces_uvs = load_uv_from_obj(SMPL_TEMPLATE_OBJ_PATH)
    vdim, fdim = verts_uvs.ndim, faces_uvs.ndim
    verts_uvs = verts_uvs.unsqueeze(0).repeat(B, *([1] * vdim))
    faces_uvs = faces_uvs.unsqueeze(0).repeat(B, *([1] * fdim))
    for batch in tqdm(atlas_loader):
        prompts = batch["prompt"]
        uv_imgs = batch["uv_image"]
        # Apply same texture across a few random poses
        for p in random.sample(pose_list, k=POSES_PER_TEXTURE):
            renderer = setup_renderer(B)
            gdim, pdim, bdim = (
                p["global_orient"].ndim,
                p["body_pose"].ndim,
                p["betas"].ndim,
            )
            global_orient = p["global_orient"].unsqueeze(0).repeat(B, *([1] * gdim))
            body_pose = p["body_pose"].unsqueeze(0).repeat(B, *([1] * pdim))
            betas = p["betas"].unsqueeze(0).repeat(B, *([1] * bdim))
            global_orient = torch.zeros(B, 3)
            smpl_output = smpl_model(
                global_orient=global_orient.to(DEVICE),
                body_pose=body_pose.to(DEVICE),
                betas=betas.to(DEVICE),
                pose2rot=True,
            )
            verts = smpl_output.vertices
            tex = uv_imgs.to(DEVICE)
            tex = resize(tex)
            tex = tex[:, :3, :, :]
            tex = tex.permute(0, 2, 3, 1)
            textures = TexturesUV(
                maps=tex,
                verts_uvs=verts_uvs,
                faces_uvs=faces_uvs,
            )
            mesh = Meshes(verts=verts, faces=faces, textures=textures)
            rendered = renderer(mesh)[..., :3]  # [B,H,W,3]
            rendered = (rendered.clamp(0, 1) * 255).byte().cpu()
            for r_img, uv_img, prompt in zip(rendered, uv_imgs, prompts):
                uv_img = to_pil(uv_img)
                rendered_img = to_pil(r_img.permute(2, 0, 1))
                writer.write(
                    {
                        "image": rendered_img,
                        "texture": uv_img,
                        "prompt": prompt,
                    }
                )
        # break

    print(f"Saving tmp dataset to: {ATLAS_LARGE_TMP_DIR}")
    writer.finalize()
    print("Saved")
    print(f"Now creating final dataset in {ATLAS_LARGE_DIR}")
    # Path to arrow file (created by ArrowWriter)
    arrow_file = os.path.join(ATLAS_LARGE_TMP_DIR, "data.arrow")

    # Wrap Arrow file into Dataset
    dataset = Dataset.from_file(arrow_file)
    os.makedirs(ATLAS_LARGE_DIR, exist_ok=True)
    # Save as a complete Hugging Face Dataset
    dataset.save_to_disk(ATLAS_LARGE_DIR)
    print("Done")
