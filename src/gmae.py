import torch
from torch import nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange
from einops import rearrange, repeat

from gsplat import rasterization

from torchvision.utils import save_image

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
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))

        self.decoder = Transformer(
            dim = decoder_dim,
            depth = decoder_depth,
            heads = decoder_heads,
            dim_head = decoder_dim_head,
            mlp_dim = decoder_dim * 4
        )
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)

        # gaussian head
        self.to_gaussians = nn.Linear(decoder_dim, 14)

    def forward(self, img, c: float = 1.0, focal_length = 50):
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
        all_decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device, dtype=dtype)
        all_decoder_tokens[batch_range, unmasked_indices] = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_to_mask)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        all_decoder_tokens[batch_range, masked_indices] = mask_tokens

        # 7. Pass the full set of tokens through the decoder
        decoded_tokens = self.decoder(all_decoder_tokens)

        # 8. Project the masked tokens to pixel space
        masked_decoded_tokens = decoded_tokens[batch_range, masked_indices]
        gaussian_params = self.to_gaussians(masked_decoded_tokens)

        mean, quat, scale, color, opacity = torch.split(
            gaussian_params[0],
            [3, 4, 3, 3, 1], 
            dim=-1
        )
        mean = mean - mean.mean(dim=0)
        mean[:, -1] += 2.0

        quat = quat / torch.norm(quat, dim=-1, keepdim=True)
        scale = c * F.sigmoid(scale)
        color = F.sigmoid(color)
        opacity = F.sigmoid(opacity).squeeze(-1)

        self.view = torch.eye(4, device=device, dtype=dtype)[None]
        self.K = torch.tensor(
            [[[focal_length, 0.0, self.image_width // 2],
            [0.0, focal_length, self.image_height // 2],
            [0.0, 0.0, 1.0]]], device=device, dtype=dtype
        )

        rgb_image, alpha, metadata = rasterization(
            mean, quat, scale, opacity, color,
            self.view, self.K, self.image_width, self.image_height,
            camera_model="ortho", rasterize_mode="antialiased"
        )

        # 9. Reconstruction loss
        pred_patches = self.to_patch(rgb_image.permute(0, 3, 1, 2))[0] * 5
        pred_patches = pred_patches - pred_patches.mean()
        recon_loss = F.mse_loss(pred_patches[masked_indices], patches[0, masked_indices])

        return rgb_image, alpha, metadata, recon_loss
    
if __name__ == "__main__":
    dtype = torch.float32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gmae = GaussianMAE(
        image_size = 256,
        channels = 3,
        patch_size = 4,
        masking_ratio = 0.75,

        # encoder configs
        encoder_dim = 512,
        encoder_depth = 8,
        encoder_heads = 8,
        encoder_dim_head = 64,

        # decoder configs
        decoder_dim = 512,
        decoder_depth = 8,
        decoder_heads = 8,
        decoder_dim_head = 64
    ).to(device, dtype)

    imgs = torch.randn(1, 3, 256, 256).to(device, dtype)

    rgb_image, alpha, metadata, recon_loss = gmae(imgs, c=0.1, focal_length=175)
    # rgb_image has a range of [0, 1]

    save_image(rgb_image[0].permute(2, 0, 1), "test.jpg")