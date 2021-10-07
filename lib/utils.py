import torch
import pickle
import numpy as np
# pytorch3d differentiable renderer
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    SoftSilhouetteShader,
    TexturesVertex,
)
# Our own wrapper, to render contours
from lib.renderer_pytorch3D import ContourRenderer
# Image manipulation
from scipy.ndimage import rotate as rotate_scp
from scipy.ndimage.morphology import binary_dilation


def fourier_transform(x, L=5):
    cosines = torch.cat([torch.cos(2**l*3.1415*x) for l in range(L)], -1)
    sines = torch.cat([torch.sin(2**l*3.1415*x) for l in range(L)], -1)
    transformed_x = torch.cat((cosines,sines),-1)
    return transformed_x

def get_projection_torch3D(az, el, distance, focal_length=35, img_w=256, img_h=256, sensor_size_mm = 32., RCAM=False):
    """Calculate 4x3 3D to 2D projection matrix given viewpoint parameters."""
    # Calculate intrinsic matrix.
    f_u = focal_length * img_w  / sensor_size_mm
    f_v = focal_length * img_h  / sensor_size_mm
    u_0 = img_w / 2
    v_0 = img_h / 2
    K = np.matrix(((f_u, 0, u_0), (0, f_v, v_0), (0, 0, 1)))

    # Calculate rotation and translation matrices.
    sa = np.sin(np.radians(-az))
    ca = np.cos(np.radians(-az))
    # Edo's convention
    #sa = np.sin(np.radians(az+90))
    #ca = np.cos(np.radians(az+90))
    R_azimuth = np.transpose(np.matrix(((ca, 0, sa),
                                          (0, 1, 0),
                                          (-sa, 0, ca))))
    se = np.sin(np.radians(-el))
    ce = np.cos(np.radians(-el))
    R_elevation = np.transpose(np.matrix(((1, 0, 0),
                                          (0, ce, -se),
                                          (0, se, ce))))
    # fix up camera
    se = np.sin(np.radians(90))
    ce = np.cos(np.radians(90))
    if RCAM:
        R_cam = np.transpose(np.matrix(((ce, -se, 0),
                                            (se, ce, 0),
                                            (0, 0, 1))))
    else:
        R_cam = np.transpose(np.matrix(((1, 0, 0),
                                        (0, 1, 0),
                                        (0, 0, 1))))
    T_world2cam = np.transpose(np.matrix((0,
                                           0,
                                           distance)))
    RT = np.hstack((R_cam@R_azimuth@R_elevation, T_world2cam))

    return RT, K

def add_common_args(arg_parser):
    arg_parser.add_argument(
        "--debug",
        dest="debug",
        default=False,
        action="store_true",
        help="If set, debugging messages will be printed",
    )
    arg_parser.add_argument(
        "--quiet",
        "-q",
        dest="quiet",
        default=False,
        action="store_true",
        help="If set, only warnings will be printed",
    )
    arg_parser.add_argument(
        "--log",
        dest="logfile",
        default=None,
        help="If set, the log will be saved using the specified filename.",
    )


class AverageValueMeter(object):
    """
    Computes and stores the average and current value of a sequence of floats
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0
        self.min = np.inf
        self.max = -np.inf

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.min = min(self.min, val)
        self.max = max(self.max, val)


class ObjectMetricTracker():
    """
    Store metrics for one object along the course of 1 refinement
    """
    def __init__(self, metrics=['chd']):
        self.metrics = {}
        self.steps = {}
        self.best_metrics = {}
        for m in metrics:
            self.metrics[m] = []
            self.steps[m] = []
            self.best_metrics[m] = 1000000.

    def append(self, m, value, step):
        """
        Stores the latest value of metric m,
        returns true if a minimum is reached,
        false otherwise
        """
        if not m in self.metrics:
            self.metrics[m] = []
            self.steps[m] = []
            self.best_metrics[m] = 1000000.
        self.metrics[m].append(value)
        self.steps[m].append(step)
        if self.metrics[m][-1] < self.best_metrics[m]:
            self.best_metrics[m] = self.metrics[m][-1]
            return True
        return False
        
    def save(self, path):
        pickle.dump(self, open(path, 'wb'))


def myChamferDistance(x, y):  # for example, x = 2025,3 y = 2048,3
    #   compute chamfer distance between two point clouds x and y
    x_size = x.size()
    y_size = y.size()
    assert (x_size[1] == y_size[1]) # Same dimensionality of pts
    x = torch.unsqueeze(x, 0)  # x = 1,2025,3
    y = torch.unsqueeze(y, 1)  # y = 2048,1,3
    x = x.repeat(y_size[0], 1, 1)  # x = 2048,2025,3
    y = y.repeat(1, x_size[0], 1)  # y = 2048,2025,3
    x_y = x - y
    x_y = torch.pow(x_y, 2)  # x_y = 2048,2025,3
    x_y = torch.sum(x_y, 2, keepdim=True)  # x_y = 2048,2025,1
    x_y = torch.squeeze(x_y, 2)  # x_y = 2048,2025
    x_y_row, _ = torch.min(x_y, 0, keepdim=True)  # x_y_row = 1,2025
    x_y_col, _ = torch.min(x_y, 1, keepdim=True)  # x_y_col = 2048,1
    return x_y_row, x_y_col.squeeze(-1).unsqueeze(0)


def contours_pointcloud(verts, faces, contours, instanciated_renderer, cam):
    """
    Input:
        verts: [_, 3]
        faces: [_, 3]
        contours: [1, 1, 256, 256]
        instanciated_renderer: a Renderer, with .depth_renderer (MeshRasterizer) attribute
        cam: FoVPerspectiveCameras
    Returns: 2d coordinates of contour points, in the range [-1,1]^2
        Shape: Nx2

    TODO: this does not support batching yet... issue is at line
        f_inds = pix_to_face[contours[0] < 0.5]
    TODO: make code more efficient, since projection to screen space is 3 times in total
        (once previously to render contours, once manually here, once in the renderer to get the fragment)
    """
    # Pack verts+faces in a mesh structure:
    meshes = Meshes(verts.unsqueeze(0), faces.unsqueeze(0))
    # Project it to screen space
    meshes_screen = instanciated_renderer.depth_renderer.transform(meshes, cameras=cam)
    # Get the vertices coordinates of projected faces
    proj_faces = meshes_screen.verts_packed()[meshes_screen.faces_packed()] # (N_faces, 3, 3)

    # Render a fragment, for each pixel getting the face id and barycentric coords
    #with torch.no_grad():
    fragment = instanciated_renderer.depth_renderer(meshes, cameras=cam)
    pix_to_face, bary_coords = fragment.pix_to_face, fragment.bary_coords # long (batch, H, W, 1) and float (batch, H, W, 1, 3])

    # Keep only points from contours
    f_inds = pix_to_face[contours[0] < 0.5]     # index of all faces projected to contours (K, 1)
    weights = bary_coords[contours[0] < 0.5]    # barycentric coordinates (K, 1, 3)
    # Filter out points that fall outside the mesh (consequence of Sobel finite diff.: slightly bleeding contours)
    weights = weights[f_inds > 0]   # (L, 3), with L <= K
    f_inds = f_inds[f_inds>0]       # (L)
    # Coordinates of the points: perform barycentric interpolation
    pts = torch.bmm(weights.unsqueeze(1), proj_faces[f_inds]) # (L,1,3)
    pts = pts.squeeze(1)    # (L,3) - u,v,depth

    # Put in 2D uv coordinates
    coords_2d_uv = pts[:,:2]                        # remove depth component
    coords_2d_uv = torch.flip(coords_2d_uv, [1])    # swap x<->y axis
    coords_2d_uv = -coords_2d_uv        # flip both axis

    return coords_2d_uv


def pack_mesh_and_render(verts, faces, instanciated_renderer, cam, light):
    # Pack mesh and create fake texture
    meshes = Meshes(verts.unsqueeze(0), faces.unsqueeze(0))
    verts_shape = meshes.verts_packed().shape
    sphere_verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=verts.device, requires_grad=False)
    meshes.textures = TexturesVertex(verts_features=sphere_verts_rgb)
    # Render
    return instanciated_renderer(meshes, cameras=cam, lights=light)


def get_renderer_cameras_lights(R_cuda, t_cuda, image_size=256):
    device = R_cuda.device
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    cameras = FoVPerspectiveCameras(device=device, znear=0.001, zfar=3500, aspect_ratio=1.0, fov=60.0, R=R_cuda, T=t_cuda)
        
    sigma = 1e-5
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.000001, # To avoid bumps in the depth map > grainy sketches
        faces_per_pixel=1,
    )
    raster_settings_soft = RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1. / 1e-4 - 1.)*sigma,
        faces_per_pixel=25,
    )
    # silhouette renderer
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings_soft
        ),
        shader=SoftSilhouetteShader()
    )
    # depth renderer
    depth_renderer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
    # image renderer
    image_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
    )
    # assemble in single rendering function
    renderer_pytorch3D = ContourRenderer(silhouette_renderer, depth_renderer, image_renderer)

    return cameras, renderer_pytorch3D, lights


def find_min_max(contours, coords, imsize):
    ### Find min/max along x direction
    # Min x
    idx_0 = np.argmin(contours, axis=0)
    idx_0 = np.stack([idx_0, coords], axis=1)[idx_0 > 0]
    # Max x
    idx_1 = np.argmin(contours[::-1,:], axis=0)
    idx_1 = np.stack([idx_1, coords], axis=1)[idx_1 > 0]
    idx_1[:,0] = imsize - idx_1[:,0] - 1
    ### Find min/max along y direction
    # Min y
    idy_0 = np.argmin(contours, axis=1)
    idy_0 = np.stack([coords, idy_0], axis=1)[idy_0 > 0]
    # Max y
    idy_1 = np.argmin(contours[:,::-1], axis=1)
    idy_1 = np.stack([coords, idy_1], axis=1)[idy_1 > 0]
    idy_1[:,1] = imsize - idy_1[:,1] - 1
    ### Create output image
    img_pc = np.ones((imsize, imsize))
    idxy = np.concatenate((idx_0, idx_1, idy_0, idy_1))
    img_pc[idxy[:,0], idxy[:,1]] = 0.
    return img_pc


def filter_contours_exterior(contours, dilation=False,
        degs = [10, 20, 30, 35, 40, 45]):
    """
    Filter a contour (binary) image, to keep only the outmost pixels
    and remove what can be seen as 'interior' pixels - without assuming
    or needing that the outmost contour is closed

    Input:
        contours: shape (N*N)
        dilation: boolean, do we thicken the returned outmost contour? 
            Can be useful after Sobel filters, that gives slightly bleeding contours,
            to be alined in 3D space
        degs: in addition to rays perpendicular to the image borders,
            what are the orientations we shoot at?
    Output:
        shape (N*N)
    """
    imsize = contours.shape[0]
    coords = np.arange(0, imsize)
    img_pc = find_min_max(contours, coords, imsize)
    for orientation in [-1, 1]:
        for d in degs:
            rotated_im = rotate_scp(contours, d * orientation, order=0, reshape=False, cval=1.)
            filtered_rotated_im = find_min_max(rotated_im, coords, imsize)
            img_pc = np.minimum(img_pc, rotate_scp(filtered_rotated_im, - d * orientation, order=0, reshape=False, cval=1.))
    # Dilate if needed:
    if dilation:
        img_pc = 1 - binary_dilation(1-img_pc, iterations=2)
    # Take intersection with original image
    return 1 - (1 - img_pc) * (1 - contours)


def filter_contours_exterior_thick(contours, dilation=True,
    degs = [0]
        ):
    """
    Filter a contour (binary) image, to keep only the outmost pixels
    and remove what can be seen as 'interior' pixels - without assuming
    or needing that the outmost contour is closed
    Input:
        contours: shape (N*N)
        dilation: boolean, do we thicken the returned outmost contour?
            Can be useful after Sobel filters, that gives slightly bleeding contours,
            to be alined in 3D space
        degs: in addition to rays perpendicular to the image borders,
            what are the orientations we shoot at?
    Output:
        shape (N*N)
    """
    imsize = contours.shape[0]
    coords = np.arange(0, imsize)
    img_pc = find_min_max_np_inner(contours, coords, imsize)
    for orientation in [-1, 1]:
        for d in degs:
            rotated_im = rotate_scp(contours, d * orientation, order=0, reshape=False, cval=1.)
            filtered_rotated_im = find_min_max_np_inner(rotated_im, coords, imsize)
            img_pc = np.minimum(img_pc, rotate_scp(filtered_rotated_im, - d * orientation, order=0, reshape=False, cval=1.))
    # Dilate if needed:
    if dilation:
        img_pc = 1 - binary_dilation(1-img_pc, iterations=1)
    # Take intersection with original image
    return 1 - (1 - img_pc) * (1 - contours)



def are_contours_thick(contours):
    ### Find min/max along x direction
    # Min x
    idx_0 = np.argmin(contours, axis=0)
    contours_2 = contours.copy()
    for i in range(len(idx_0)):
        contours_2[:idx_0[i], i] = 0
    idx_0_inner = np.argmax(contours_2, axis=0) - 1 - 2
    diff_a = idx_0_inner - idx_0
    # Max x
    idx_1 = np.argmin(contours[::-1,:], axis=0)
    contours_2 = contours[::-1,:].copy()
    for i in range(len(idx_1)):
        contours_2[:idx_1[i], i] = 0
    idx_1_inner = np.argmax(contours_2, axis=0) - 1 - 2
    diff_b = idx_1_inner - idx_1
    ### Find min/max along y direction
    # Min y
    idy_0 = np.argmin(contours, axis=1)
    contours_2 = contours.copy()
    for i in range(len(idy_0)):
        contours_2[i, :idy_0[i]] = 0
    idy_0_inner = np.argmax(contours_2, axis=1) - 1 - 2
    diff_c = idy_0_inner - idy_0
    # Max y
    idy_1 = np.argmin(contours[:,::-1], axis=1)
    contours_2 = contours[:,::-1].copy()
    for i in range(len(idy_1)):
        contours_2[i, :idy_1[i]] = 0
    idy_1_inner = np.argmax(contours_2, axis=1) - 1 - 2
    diff_d = idy_1_inner - idy_1
    # Heuristic on difference between inner/outer contours
    thick_a = diff_a[(diff_a > -3)*(diff_a < 6)].mean() > 1.2
    thick_b = diff_b[(diff_b > -3)*(diff_b < 6)].mean() > 1.2
    thick_c = diff_c[(diff_c > -3)*(diff_c < 6)].mean() > 1.2
    thick_d = diff_d[(diff_d > -3)*(diff_d < 6)].mean() > 1.2
    return thick_a and thick_b and thick_c and thick_d


# Inner
def find_min_max_np_inner(contours, coords, imsize):
    ### Find min/max along x direction
    # Min x
    idx_0 = np.argmin(contours, axis=0)
    # UGLY LOOP: fill contours up to idx_0
    contours_2 = contours.copy()
    for i in range(len(idx_0)):
        contours_2[:idx_0[i], i] = 0
    idx_0_inner = np.argmax(contours_2, axis=0) - 1 - 2
    idx_0 = np.stack([idx_0_inner, coords], axis=1)[idx_0 > 0]
    #
    #
    # Max x
    idx_1 = np.argmin(contours[::-1,:], axis=0)
    # UGLY LOOP: fill contours up to idx_1
    contours_2 = contours[::-1,:].copy()
    for i in range(len(idx_1)):
        contours_2[:idx_1[i], i] = 0
    idx_1_inner = np.argmax(contours_2, axis=0) - 1 - 2
    idx_1 = np.stack([idx_1_inner, coords], axis=1)[idx_1 > 0]
    idx_1[:,0] = imsize - idx_1[:,0] - 1
    #
    #
    ### Find min/max along y direction
    # Min y
    idy_0 = np.argmin(contours, axis=1)
    # UGLY LOOP: fill contours up to idy_0
    contours_2 = contours.copy()
    for i in range(len(idy_0)):
        contours_2[i, :idy_0[i]] = 0
    idy_0_inner = np.argmax(contours_2, axis=1) - 1 - 2
    idy_0 = np.stack([coords, idy_0_inner], axis=1)[idy_0 > 0]
    #
    #
    # Max y
    idy_1 = np.argmin(contours[:,::-1], axis=1)
    # UGLY LOOP: fill contours up to idy_1
    contours_2 = contours[:,::-1].copy()
    for i in range(len(idy_1)):
        contours_2[i, :idy_1[i]] = 0
    idy_1_inner = np.argmax(contours_2, axis=1) - 1 - 2
    idy_1 = np.stack([coords, idy_1_inner], axis=1)[idy_1 > 0]
    idy_1[:,1] = imsize - idy_1[:,1] - 1
    #
    #
    ### Create output image
    img_pc = np.ones((imsize, imsize))
    idxy = np.concatenate((idx_0, idx_1, idy_0, idy_1))
    img_pc[idxy[:,0], idxy[:,1]] = 0.
    return img_pc

def filter_contours_input(contours):
    if are_contours_thick(contours):
        return filter_contours_exterior_thick(contours)
    else:
        return filter_contours_exterior(contours)
