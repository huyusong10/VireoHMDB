import torch
from torch import nn, einsum
from einops import rearrange

class AbsPosEmb(nn.Module):
    def __init__(
        self,
        fmap_size,
        dim_head
    ):
        super().__init__()
        frames, height, width = fmap_size
        scale = dim_head ** -0.5
        self.frames = nn.Parameter(torch.randn(frames, dim_head) * scale)
        self.height = nn.Parameter(torch.randn(height, dim_head) * scale)
        self.width = nn.Parameter(torch.randn(width, dim_head) * scale)

    def forward(self, q):
        emb = rearrange(self.frames, 'f d -> f () () d') + rearrange(self.height, 'h d -> () h () d') + rearrange(self.width, 'w d -> () () w d')
        emb = rearrange(emb, 'f h w d -> (f h w) d')
        logits = einsum('b h i d, j d -> b h i j', q, emb)
        return logits

class ABS_IMPL(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        heads = 4,
        dim_head = 128,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qkv = nn.Conv3d(dim, inner_dim * 3, 1, bias = False)

        self.pos_emb = AbsPosEmb(fmap_size, dim_head)

    def forward(self, fmap):
        heads, b, c, f, h, w = self.heads, *fmap.shape

        q, k, v = self.to_qkv(fmap).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y z -> b h (x y z) d', h = heads), (q, k, v))

        q *= self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim += self.pos_emb(q)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y z) d -> b (h d) x y z', x=f, y=h, z=w)
        return out

def get_mhsa(name='abs'):
    
    if name=='abs':
        return ABS_IMPL
