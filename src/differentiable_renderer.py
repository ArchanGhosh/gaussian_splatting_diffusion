import torch
import torch.nn as nn

from src.config import DEVICE, SPLATS_PER_CELL




class DifferentiableSplatRenderer(nn.Module):
    def __init__(self, img_size=128, grid_size=32, splats_per_cell=SPLATS_PER_CELL, device=DEVICE):
        super().__init__()
        self.img_size = img_size
        self.grid_size = grid_size
        self.splats_per_cell = splats_per_cell
        self.device = device
        self.tile_size = 32

    def get_splat_params(self, splat_grid, B):
        GH, GW = self.grid_size, self.grid_size
        # Reshape to isolate splats
        splats = splat_grid.permute(0, 2, 3, 1).reshape(B, GH, GW, self.splats_per_cell, 9).reshape(B, -1, 9)

        # Grid Coordinates
        stride_h = 2.0 / GH
        stride_w = 2.0 / GW
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1 + stride_h/2, 1 - stride_h/2, GH, device=self.device),
            torch.linspace(-1 + stride_w/2, 1 - stride_w/2, GW, device=self.device),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2).expand(-1, -1, self.splats_per_cell, -1).reshape(1, -1, 2)

        # Activation Functions
        offsets = torch.tanh(splats[..., 0:2]) * torch.tensor([stride_w, stride_h], device=self.device) * 1.5
        centers = base_grid + offsets
        scales = torch.exp(splats[..., 2:4]) * 0.04
        thetas = splats[..., 4]
        colors = torch.sigmoid(splats[..., 5:8])
        opacities = torch.sigmoid(splats[..., 8:9])

        return centers, scales, thetas, colors, opacities

    def render_tile(self, h_start, w_start, centers, scales, thetas, colors, opacities):
        h_end = h_start + self.tile_size
        w_end = w_start + self.tile_size

        y_range = torch.linspace(-1, 1, self.img_size, device=self.device)[h_start:h_end]
        x_range = torch.linspace(-1, 1, self.img_size, device=self.device)[w_start:w_end]
        y, x = torch.meshgrid(y_range, x_range, indexing='ij')
        pixel_coords = torch.stack([x, y], dim=-1).unsqueeze(0).unsqueeze(0)

        # Vectorized Gaussian Evaluation
        delta = pixel_coords - centers.unsqueeze(1).unsqueeze(1)
        cos_t = torch.cos(thetas); sin_t = torch.sin(thetas)
        scale_x_inv2 = 1.0 / (scales[:, 0] ** 2 + 1e-6)
        scale_y_inv2 = 1.0 / (scales[:, 1] ** 2 + 1e-6)

        a = cos_t**2 * scale_x_inv2 + sin_t**2 * scale_y_inv2
        b = -sin_t * cos_t * scale_x_inv2 + sin_t * cos_t * scale_y_inv2
        c = sin_t**2 * scale_x_inv2 + cos_t**2 * scale_y_inv2

        quad_form = (a.unsqueeze(1).unsqueeze(1) * delta[..., 0]**2 +
                     2 * b.unsqueeze(1).unsqueeze(1) * delta[..., 0] * delta[..., 1] +
                     c.unsqueeze(1).unsqueeze(1) * delta[..., 1]**2)

        weights = torch.exp(-0.5 * quad_form) * opacities.view(1, -1, 1, 1)

        denominator = weights.sum(dim=1, keepdim=True) + 1e-6
        numerator = (weights.unsqueeze(2) * colors.view(1, -1, 3, 1, 1)).sum(dim=1)

        return numerator / denominator

    def forward(self, splat_grid):
        B = splat_grid.shape[0]
        centers, scales, thetas, colors, opacities = self.get_splat_params(splat_grid, B)
        output_images = []
        for i in range(B):
            c_i, s_i, t_i, col_i, o_i = centers[i], scales[i], thetas[i], colors[i], opacities[i]
            tiles_rows = []
            for h in range(0, self.img_size, self.tile_size):
                tiles_cols = []
                for w in range(0, self.img_size, self.tile_size):
                    tiles_cols.append(self.render_tile(h, w, c_i, s_i, t_i, col_i, o_i))
                tiles_rows.append(torch.cat(tiles_cols, dim=3))
            output_images.append(torch.cat(tiles_rows, dim=2))
        return torch.cat(output_images, dim=0)