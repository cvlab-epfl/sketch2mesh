import torch
import torch.nn as nn
import numpy as np

class SoftThreshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold = 0.45, factor = 10.0):
        with torch.enable_grad():
            output = torch.sigmoid(factor*(input-threshold))
            ctx.save_for_backward(input, output)
        # binary thresholding
        return (input>threshold).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        input.retain_grad()
        output.backward(grad_output, retain_graph=True)
        return input.grad


class ContourRenderer(nn.Module):
    def __init__(self, silhouette_renderer, depth_renderer, image_renderer, max_depth=5, H=256, W=256):
        super().__init__()
        self.silhouette_renderer = silhouette_renderer
        self.depth_renderer = depth_renderer
        self.image_renderer = image_renderer

        self.max_depth = max_depth

        self.threshold = SoftThreshold()

        with torch.no_grad():
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                torch.cuda.set_device(self.device)
            else:
                self.device = torch.device("cpu")

            ## INIT SOBEL FILTER
            k_filter = 3
            filter = self.get_sobel_kernel(k_filter)

            self.filter_x = torch.nn.Conv2d(in_channels=1,
                                            out_channels=1,
                                            kernel_size=k_filter,
                                            padding=0,
                                            bias=False)
            self.filter_x.weight[:] = torch.tensor(filter, requires_grad = False)
            self.filter_x = self.filter_x.to(self.device)

            self.filter_y = torch.nn.Conv2d(in_channels=1,
                                            out_channels=1,
                                            kernel_size=k_filter,
                                            padding=0,
                                            bias=False)
            self.filter_y.weight[:] = torch.tensor(filter.T, requires_grad = False)
            self.filter_y = self.filter_y.to(self.device)

        # Pixel coordinates
        self.X, self.Y = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        self.X = (2*(0.5 + self.X.unsqueeze(0).unsqueeze(-1))/H - 1).float().cuda()
        self.Y = (2*(0.5 + self.Y.unsqueeze(0).unsqueeze(-1))/W - 1).float().cuda()

    def get_sobel_kernel(self, k=3):
        # get range
        range = np.linspace(-(k // 2), k // 2, k)
        # compute a grid the numerator and the axis-distances
        x, y = np.meshgrid(range, range)
        sobel_2D_numerator = x
        sobel_2D_denominator = (x ** 2 + y ** 2)
        sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
        sobel_2D = sobel_2D_numerator / sobel_2D_denominator
        return sobel_2D

    def depth_2_normal(self, depth, depth_unvalid, cameras):
        B, H, W, C = depth.shape

        grad_out = torch.zeros(B, H, W, 3).cuda()
        # Pixel coordinates
        xy_depth = torch.cat([self.X.repeat(B,1,1,1), self.Y.repeat(B,1,1,1), depth], 3).reshape(B,-1, 3)
        xyz_unproj = cameras.unproject_points(xy_depth, world_coordinates=False)

        # compute tangent vectors
        XYZ_camera = xyz_unproj.reshape(B, H, W, 3)
        vx = XYZ_camera[:,1:-1,2:,:]-XYZ_camera[:,1:-1,1:-1,:]
        vy = XYZ_camera[:,2:,1:-1,:]-XYZ_camera[:,1:-1,1:-1,:]

        # finally compute cross product
        normal = torch.cross(vx.reshape(-1, 3),vy.reshape(-1, 3))
        normal_norm = normal.norm(p=2, dim=1, keepdim=True)
        normal_normalized = normal.div(normal_norm)
        # reshape to image
        normal_out = normal_normalized.reshape(B, H-2, W-2, 3)
        grad_out[:,1:-1,1:-1,:] = (0.5 - 0.5*normal_out)

        # zero out +Inf
        grad_out[depth_unvalid] = 0.0

        return grad_out

    def buffer_2_contour_float64(self, buffer):
        # set the steps tensors
        B, C, H, W = buffer.shape
        grad = torch.zeros((B, 1, H, W)).to(self.device).double()
        padded_buffer = torch.nn.functional.pad(buffer, (1,1,1,1), mode='reflect')
        for c in range(C):
            grad_x = self.filter_x(padded_buffer[:, c:c+1])
            grad_y = self.filter_y(padded_buffer[:, c:c+1])
            grad_tensor = torch.stack((grad_x, grad_y),-1)
            grad_magnitude = torch.norm(grad_tensor, p =2, dim = -1)
            grad = grad + grad_magnitude

        return self.threshold.apply(1.0 - (torch.clamp(grad,0,1)))

    def buffer_2_contour(self, buffer):
        # set the steps tensors
        B, C, H, W = buffer.shape
        grad = torch.zeros((B, 1, H, W)).to(self.device)
        padded_buffer = torch.nn.functional.pad(buffer, (1,1,1,1), mode='reflect')
        for c in range(C):
            grad_x = self.filter_x(padded_buffer[:, c:c+1])
            grad_y = self.filter_y(padded_buffer[:, c:c+1])
            grad_tensor = torch.stack((grad_x, grad_y),-1)
            grad_magnitude = torch.norm(grad_tensor, p =2, dim = -1)
            grad = grad + grad_magnitude

        return self.threshold.apply(1.0 - (torch.clamp(grad,0,1)))



    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        # render silhouettes?
        # silhouette_ref = self.silhouette_renderer(meshes_world=meshes_world, **kwargs)
        # silhouette_out = silhouette_ref[..., 3]

        # now get depth out
        depth_ref = self.depth_renderer(meshes_world=meshes_world, **kwargs)
        depth_ref = depth_ref.zbuf[...,0].unsqueeze(-1)
        depth_unvalid = depth_ref<0
        depth_ref[depth_unvalid] = self.max_depth
        depth_out = depth_ref[..., 0]

        # post process depth to get normals, contours
        normals_out = self.depth_2_normal(depth_ref, depth_unvalid.squeeze(-1), kwargs['cameras']).permute(0,3,1,2)
        contours_out = self.buffer_2_contour(
                                torch.cat(( normals_out,
                                            depth_ref.permute(0,3,1,2))
                                    , 1)
                                )

        # textured image for visualization?
        # image_ref = self.image_renderer(meshes_world=meshes_world, **kwargs)
        # image_out = image_ref[..., 0:3].permute(0,3,1,2)

        return contours_out
