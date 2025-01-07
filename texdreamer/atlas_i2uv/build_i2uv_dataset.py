# build_i2uv_dataset.py

import os
import random
import glob
from PIL import Image
from datasets import Dataset
import numpy as np
import torch
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
import math

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
POSES_PER_TEXTURE = 5

IMAGE_SIZE = 1024
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


def rotation_matrix(x_deg=0.0, y_deg=0.0, z_deg=0.0):
    # Convert degrees to radians
    x_rad = math.radians(x_deg)
    y_rad = math.radians(y_deg)
    z_rad = math.radians(z_deg)

    # Rotation matrix around X-axis
    Rx = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(x_rad), -math.sin(x_rad)],
            [0.0, math.sin(x_rad), math.cos(x_rad)],
        ],
        dtype=torch.float32,
    )

    # Rotation matrix around Y-axis
    Ry = torch.tensor(
        [
            [math.cos(y_rad), 0.0, math.sin(y_rad)],
            [0.0, 1.0, 0.0],
            [-math.sin(y_rad), 0.0, math.cos(y_rad)],
        ],
        dtype=torch.float32,
    )

    # Rotation matrix around Z-axis
    Rz = torch.tensor(
        [
            [math.cos(z_rad), -math.sin(z_rad), 0.0],
            [math.sin(z_rad), math.cos(z_rad), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    # Final combined rotation matrix: R = Rz @ Ry @ Rx
    R = Rz @ Ry @ Rx
    return R


def get_random_cam():
    focal_len = 3.3
    tx = 0.0
    ty = 0.3
    tz = 3.0
    Tr = torch.tensor([tx, ty, tz])
    ro_angles = list(range(0, 360, 60))
    ro_x, ro_z = 0, 0
    ro_y = random.sample(ro_angles, k=1)[0]
    Ro = rotation_matrix(ro_x, ro_y, ro_z)
    cameras = PerspectiveCameras(
        focal_length=((focal_len, focal_len),),
        R=Ro.unsqueeze(0).to(DEVICE),
        T=Tr.unsqueeze(0).to(DEVICE),
        device=DEVICE,
    )
    return cameras


def setup_renderer():
    cameras = get_random_cam()
    raster_settings = RasterizationSettings(
        image_size=IMAGE_SIZE,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    lights = PointLights(location=[[0.0, 2.0, 2.0]], device=DEVICE)
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
    ds = load_dataset("ggxxii/ATLAS", split="train", cache_dir=ATLAS_DATASET_DIR)
    atlas = ds.select(range(10))

    print("Loading AMASS poses...")
    pose_list = load_amass_smpl_poses(AMASS_DIR)
    print(f"Loaded {len(pose_list)} poses.")

    print("Preparing SMPL and renderer...")
    smpl_model, faces = load_smpl(SMPL_MODEL_PATH)
    verts_uvs, faces_uvs = load_uv_from_obj(SMPL_TEMPLATE_OBJ_PATH)

    prompts = load_prompts(prompts_file=PROMPTS_FILE)
    for idx in tqdm(range(len(atlas))):
        uv_img = atlas[idx]["image"]  # .convert("RGB")
        assert uv_img.mode == "RGBA"

        # Apply same texture across a few random poses
        for p in random.sample(pose_list, k=POSES_PER_TEXTURE):
            renderer = setup_renderer()
            smpl_output = smpl_model(
                # global_orient=p["global_orient"].unsqueeze(0).to(DEVICE),
                body_pose=p["body_pose"].unsqueeze(0).to(DEVICE),
                betas=p["betas"].unsqueeze(0).to(DEVICE),
                pose2rot=True,
            )
            verts = smpl_output.vertices[0]

            tex = to_tensor(uv_img).unsqueeze(0).to(DEVICE)
            tex = resize(tex)
            tex = tex[:, :3, :, :]
            tex = tex.permute(0, 2, 3, 1)
            textures = TexturesUV(
                maps=tex,
                verts_uvs=verts_uvs.unsqueeze(0),
                faces_uvs=faces_uvs.unsqueeze(0),
            )
            mesh = Meshes(verts=[verts], faces=[faces], textures=textures)

            rendered = renderer(mesh)[0, ..., :3]  # [H,W,3]
            rendered = (rendered.clamp(0, 1) * 255).byte().cpu()
            rendered_img = to_pil(rendered.permute(2, 0, 1))
            writer.write(
                {
                    "image": rendered_img,
                    "texture": uv_img,
                    "prompt": prompts[idx],
                }
            )

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
