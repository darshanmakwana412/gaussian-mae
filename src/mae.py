import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from einops import rearrange, repeat

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
    
class MAE(nn.Module):
    def __init__(
        self,
        # encoder configs
        image_size,
        patch_size,
        encoder_dim,
        encoder_depth,
        encoder_heads,
        pool = 'cls',
        channels = 3,
        encoder_dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,

        # decoder configs
        decoder_dim = 512,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64
    ):
        super().__init__()
        assert 0 < masking_ratio < 1, 'masking_ratio must be between 0 and 1'
        self.masking_ratio = masking_ratio

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls or mean'

        # patchify
        self.to_patch = rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)
        self.patch_to_emb = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, encoder_dim),
            nn.LayerNorm(encoder_dim),
        )

        # Encoder
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, encoder_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, encoder_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.encoder = Transformer(
            dim = encoder_dim, 
            depth = encoder_depth,
            heads = encoder_heads, 
            dim_head = encoder_dim_head,
            mlp_dim = encoder_dim * 4,
            dropout = dropout
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

        # "pixel_values_per_patch" = dimension of each patch before embedding
        pixel_values_per_patch = patch_dim
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, img):
        device = img.device

        # 1. Patchify the image
        patches = self.to_patch(img)  
        tokens = self.patch_to_emb(patches)
        batch, num_patches, *_ = patches.shape

        # 2. Add positional embeddings
        if self.pool == "cls":
            b, n, _ = tokens.shape
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            tokens = torch.cat((cls_tokens, tokens), dim=1)
            tokens = tokens + self.pos_embedding[:, :(n + 1)]
        else:
            tokens = tokens + self.pos_embedding[:, 1:(num_patches + 1)]
        tokens = self.dropout(tokens)

        # 3. Mask the patched tokens
        num_to_mask = int(self.masking_ratio * num_patches)
        total_tokens_for_masking = tokens.shape[1] - 1 if self.pool == "cls" else tokens.shape[1]
        rand_indices = torch.rand(batch, total_tokens_for_masking, device=device).argsort(dim=-1)

        masked_indices = rand_indices[:, :num_to_mask]
        unmasked_indices = rand_indices[:, num_to_mask:]
        if self.pool == "cls":
            patch_tokens = tokens[:, 1:]
        else:
            patch_tokens = tokens

        batch_range = torch.arange(batch, device=device)[:, None]
        unmasked_tokens = patch_tokens[batch_range, unmasked_indices]

        masked_patches = patches[batch_range, masked_indices]

        if self.pool == "cls":
            cls_tok = tokens[:, 0].unsqueeze(1)
            tokens_for_encoder = torch.cat((cls_tok, unmasked_tokens), dim=1)
        else:
            tokens_for_encoder = unmasked_tokens

        # 4. Pass the unmasked tokens through the encoder
        encoded_tokens = self.encoder(tokens_for_encoder)
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # 5. Prepare the full set of tokens for the decoder
        all_decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)

        
        if self.pool == "cls":
            unmasked_decoder_tokens = decoder_tokens[:, 1:]  
        else:
            unmasked_decoder_tokens = decoder_tokens

        # Add decoder position embeddings to the unmasked tokens
        unmasked_decoder_tokens = (
            unmasked_decoder_tokens + self.decoder_pos_emb(unmasked_indices)
        )

        # Place unmasked tokens into full set
        all_decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_to_mask)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # Insert mask tokens
        all_decoder_tokens[batch_range, masked_indices] = mask_tokens

        # 7. Pass the full set of tokens through the decoder
        decoded_tokens = self.decoder(all_decoder_tokens)

        # 8. Project the masked tokens to pixel space
        masked_decoded_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(masked_decoded_tokens)

        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)

        return recon_loss
