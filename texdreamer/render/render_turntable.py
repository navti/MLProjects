# render_turntable.py

import os
import random
import torch
from torch import nn
from torchvision import transforms as T
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV
from PIL import Image
from utils import *
from smplx import SMPL


class TurnTableRender(nn.Module):
    def __init__(
        self,
        device="cpu",
        smpl_model_path="~/datasets/smpl/models",
        smpl_template_obj_path="~/datasets/smpl/models/smpl_uv_template.obj",
        amass_dir="~/datasets/amass/CMU",
        render_out_dir="render/out",
        render_img_size=(512, 384),  # (height, width)
        uv_img_resize=(1024, 1024),  # (height, width)
        radius=2.7,
        focal_length=500.0,
    ):
        super(TurnTableRender, self).__init__()
        self.device = device

        # Config
        self.smpl_model_path = os.path.expanduser(smpl_model_path)
        self.smpl_template_obj_path = os.path.expanduser(smpl_template_obj_path)
        self.amass_dir = os.path.expanduser(amass_dir)
        self.render_out_dir = os.path.abspath(render_out_dir)
        self.render_img_size = render_img_size
        self.radius = radius
        self.focal_length = focal_length

        # Transforms
        self.to_tensor = T.ToTensor()
        self.to_pil = T.ToPILImage()
        self.resize = T.Resize(uv_img_resize)
        print("Loading AMASS poses...")
        self.pose_list = load_amass_smpl_poses(self.amass_dir, num_poses=1000)
        print(f"Loaded {len(self.pose_list)} poses.")
        print("Preparing SMPL and renderer...")
        self.smpl_model, self.faces = load_smpl(self.smpl_model_path, self.device)
        self.verts_uvs, self.faces_uvs = load_uv_from_obj(
            self.smpl_template_obj_path, self.device
        )

    def forward(self, uv_img_path, render_path=None, n_frames=30, duration=50):
        uv = Image.open(uv_img_path)
        uv_imgs = self.to_tensor(self.resize(uv)).unsqueeze(0).repeat(n_frames, 1, 1, 1)
        faces = self.faces.unsqueeze(0).repeat(n_frames, 1, 1)
        vdim, fdim = self.verts_uvs.ndim, self.faces_uvs.ndim
        verts_uvs = self.verts_uvs.unsqueeze(0).repeat(n_frames, *([1] * vdim))
        faces_uvs = self.faces_uvs.unsqueeze(0).repeat(n_frames, *([1] * fdim))
        renderer = setup_renderer(
            n_frames, img_size=self.render_img_size, device=self.device
        )
        tex = uv_imgs.to(self.device)
        tex = tex[:, :3, :, :]
        tex = tex.permute(0, 2, 3, 1)
        textures = TexturesUV(
            maps=tex,
            verts_uvs=verts_uvs,
            faces_uvs=faces_uvs,
        )
        # sample 10 random poses, and then pick one
        p = random.sample(self.pose_list, k=100)[0]
        gdim, pdim, bdim = (
            p["global_orient"].ndim,
            p["body_pose"].ndim,
            p["betas"].ndim,
        )
        # global_orient = p["global_orient"].unsqueeze(0).repeat(B, *([1] * gdim))
        body_pose = p["body_pose"].unsqueeze(0).repeat(n_frames, *([1] * pdim))
        betas = p["betas"].unsqueeze(0).repeat(n_frames, *([1] * bdim))
        global_orient = torch.zeros(n_frames, 3)
        smpl_output = self.smpl_model(
            global_orient=global_orient.to(self.device),
            body_pose=body_pose.to(self.device),
            betas=betas.to(self.device),
            pose2rot=True,
        )
        verts = smpl_output.vertices
        mesh = Meshes(verts=verts, faces=faces, textures=textures)
        rendered = renderer(mesh)[..., :3]  # [B,H,W,3]
        frames = [self.to_pil(img) for img in rendered.permute(0, 3, 1, 2).cpu()]
        if render_path is not None:
            self.render_out_dir = render_path
        os.makedirs(self.render_out_dir, exist_ok=True)
        gif_path = f"{self.render_out_dir}/render.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )
        print("GIF saved to:", gif_path)
        return frames


# main
if __name__ == "__main__":
    # Save as GIF if path provided
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    renderer = TurnTableRender(device=device)
    uv_path = os.path.abspath("../texdreamer/generated_uv.png")
    frames = renderer(uv_path, n_frames=60)
