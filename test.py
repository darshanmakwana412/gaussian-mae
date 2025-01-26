import torch
from gsplat import rasterization

# Initialize a 3D Gaussian :
N = 1000
mean = torch.randn(N, 3, device="cuda", dtype=torch.float32)
quat = torch.randn(N, 4, device="cuda", dtype=torch.float32)
color = torch.rand((N, 3), device="cuda", dtype=torch.float32)
opac = torch.rand((N,), device="cuda", dtype=torch.float32)
scale = torch.rand((N, 3), device="cuda", dtype=torch.float32)
view = torch.eye(4, device="cuda", dtype=torch.float32)[None]

f = 50
K = torch.tensor(
    [[[f, 0.0, 128.0],
      [0.0, f, 128.0],
      [0.0, 0.0, 1.0]]], device="cuda"
)  # camera intrinsics
rgb_image, alpha, metadata = rasterization(
    mean, quat, scale, opac, color, view, K, 256, 256, camera_model="ortho", rasterize_mode="antialiased"
)

from torchvision.utils import save_image
print(rgb_image.max(), rgb_image.min())
save_image(rgb_image[0].permute(2, 0, 1), "test.jpg")