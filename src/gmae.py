import torch
from torch import nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange
from einops import rearrange, repeat

from gsplat import rasterization

from torchvision.utils import save_image, make_grid

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)
    
class GaussianMAE(nn.Module):
    def __init__(
        self,

        image_size = 256,
        channels = 3,
        patch_size = 4,
        masking_ratio = 0.75,
        num_gaussians = 512,

        # encoder configs
        encoder_dim = 512,
        encoder_depth = 8,
        encoder_heads = 8,
        encoder_dim_head = 64,

        # decoder configs
        decoder_dim = 512,
        decoder_depth = 8,
        decoder_heads = 8,
        decoder_dim_head = 64,
    ):
        super().__init__()
        assert 0 < masking_ratio < 1, 'masking_ratio must be between 0 and 1'
        self.masking_ratio = masking_ratio

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.image_width = image_width
        self.image_height = image_height
        self.num_gaussians = num_gaussians

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        # patchify
        self.to_patch = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)
        self.patch_to_emb = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, encoder_dim),
            nn.LayerNorm(encoder_dim),
        )

        # Encoder
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, encoder_dim))

        self.encoder = Transformer(
            dim = encoder_dim, 
            depth = encoder_depth,
            heads = encoder_heads, 
            dim_head = encoder_dim_head,
            mlp_dim = encoder_dim * 4,
        )

        # Decoder
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(num_gaussians, decoder_dim))

        self.decoder = Transformer(
            dim = decoder_dim,
            depth = decoder_depth,
            heads = decoder_heads,
            dim_head = decoder_dim_head,
            mlp_dim = decoder_dim * 4
        )
        self.decoder_pos_emb = nn.Embedding(num_gaussians, decoder_dim)

        # gaussian head
        self.to_gaussians = nn.Linear(decoder_dim, 14)

    def forward(self, img):
        device = img.device
        dtype = img.dtype

        # 1. Patchify the image
        patches = self.to_patch(img)  
        patch_tokens = self.patch_to_emb(patches)
        batch, num_patches, *_ = patches.shape

        # 2. Add positional embeddings
        patch_tokens += self.pos_embedding[:, :num_patches]

        # 3. Mask the patched tokens
        num_to_mask = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, patch_tokens.shape[1], device=device).argsort(dim=-1)

        masked_indices = rand_indices[:, :num_to_mask]
        unmasked_indices = rand_indices[:, num_to_mask:]

        batch_range = torch.arange(batch, device=device)[:, None]
        unmasked_tokens = patch_tokens[batch_range, unmasked_indices]

        # 4. Pass the unmasked tokens through the encoder
        encoded_tokens = self.encoder(unmasked_tokens)
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # 5. Prepare the full set of tokens for the decoder
        # all_decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device, dtype=dtype)
        # all_decoder_tokens[batch_range, unmasked_indices] = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # mask_tokens = repeat(self.mask_token, 'n d -> b n d', b=batch)
        mask_tokens = torch.randn(batch, self.num_gaussians, self.decoder_dim).to(device, dtype)
        mask_tokens = mask_tokens + self.decoder_pos_emb.weight

        # all_decoder_tokens[batch_range, masked_indices] = mask_tokens

        decoder_tokens = torch.cat((decoder_tokens, mask_tokens), dim=1)

        # 7. Pass the full set of tokens through the decoder
        decoded_tokens = self.decoder(decoder_tokens)

        # 8. Project the masked tokens to pixel space
        masked_decoded_tokens = decoded_tokens[:, encoded_tokens.shape[1]:]
        gaussian_params = self.to_gaussians(masked_decoded_tokens)
        return gaussian_params
    
    def decode_gaussian_params(self, gaussian_params, c: float = 1.0):
        means, quats, scales, opacities, colors = torch.split(
            gaussian_params,
            [3, 4, 3, 1, 3], 
            dim=-1
        )

        means = F.tanh(means)
        means = means - means.mean(dim=0)
        quats = F.sigmoid(quats)
        scales = c * F.sigmoid(scales)
        opacities = F.sigmoid(opacities).squeeze(-1)
        colors = F.sigmoid(colors)

        return means, quats, scales, opacities, colors
    
    def rasterize(self, means, quats, scales, opacities, colors, focal_length: float = 175.0):
        device = means.device
        dtype = means.dtype

        viewmats = torch.eye(4, device=device, dtype=dtype)[None]
        viewmats[:, 2, 3] = 5
        Ks = torch.tensor(
            [[[focal_length, 0.0, self.image_width // 2],
            [0.0, focal_length, self.image_height // 2],
            [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )

        # https://github.com/nerfstudio-project/gsplat/blob/0880d2b471e6650d458aa09fe2b2834531f6e93b/gsplat/rendering.py#L28-L54
        rgb_image, alpha, metadata = rasterization(
            means, quats, scales, opacities, colors,
            viewmats, Ks, self.image_width, self.image_height,
            camera_model="ortho", rasterize_mode="classic",
            backgrounds=torch.tensor([[1.0, 1.0, 1.0]], device=device, dtype=dtype)
        )

        return rgb_image, alpha, metadata

if __name__ == "__main__":
    dtype = torch.float32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gmae = GaussianMAE(
        image_size = 256,
        channels = 3,
        patch_size = 4,
        masking_ratio = 0.75,
        num_gaussians=512,

        # encoder configs
        encoder_dim = 384,
        encoder_depth = 6,
        encoder_heads = 6,
        encoder_dim_head = 64,

        # decoder configs
        decoder_dim = 384,
        decoder_depth = 6,
        decoder_heads = 6,
        decoder_dim_head = 64
    ).to(device, dtype)

    # imgs = torch.randn(4, 3, 256, 256).to(device, dtype)
    imgs = torch.randn(4, 3, 256, 256).to(device, dtype) * 2 - 1

    # Replicates the memory efficient implementation from https://arxiv.org/abs/2404.19737
    gaussian_params_shared = gmae(imgs)
    gaussian_params = gaussian_params_shared.detach()
    gaussian_params.requires_grad = True

    images = []
    loss = 0
    for gaussian_param, img in zip(gaussian_params, imgs):
        (
            means,
            quats,
            scales,
            opacities,
            colors,
        ) = gmae.decode_gaussian_params(gaussian_param, c=0.2)

        rgb_image, alpha, metadata = gmae.rasterize(
            means,
            quats,
            scales,
            opacities,
            colors,
            focal_length=200
        )
        rgb_image = rgb_image[0].permute(2, 0, 1)
        F.mse_loss(rgb_image, img).backward()

        images.append(rgb_image.clone().detach().cpu())

    gaussian_params_shared.backward(gradient=gaussian_params.grad)

    grid = make_grid(images, nrow=2, padding=2, normalize=True)
    save_image(grid, "output.png")