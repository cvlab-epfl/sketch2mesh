import numpy as np
import plyfile
import torch
from lib.utils import fourier_transform
import skimage.measure

def write_verts_faces_to_file(verts, faces, ply_filename_out):
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(verts[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)

def convert_sdf_samples_to_mesh(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to verts, faces

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # return mesh
    return mesh_points, faces


def create_mesh_optim_faster(
    samples, indices, decoder, latent_vec, N=256, max_batch=32 ** 3, offset=None, scale=None, fourier = False, taylor = False
    , debug = False
):
    # Move tensors to cuda in case they are not already there
    samples = samples.cuda()
    if torch.is_tensor(indices):
        indices = indices.clone().long().cuda()
    else: # numpy case
        indices = torch.tensor(indices).long().cuda()
    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    # Update SDF values on the grid
    num_samples = indices.shape[0]
    with torch.no_grad():
        head = 0
        while head < num_samples:
            sample_subset = samples[indices[head : min(head + max_batch, num_samples)], 0:3].reshape(-1, 3)
            if fourier:
                sample_subset = fourier_transform(sample_subset)

            latent_inputs = latent_vec.expand(sample_subset.shape[0], -1)
            inputs = torch.cat([latent_inputs, sample_subset], 1)

            samples[indices[head : min(head + max_batch, num_samples)], 3] = decoder(inputs).squeeze(1)
            head += max_batch

        sdf_values = samples[:, 3]
        sdf_values = sdf_values.reshape(N, N, N)

    # Run MC
    verts, faces = convert_sdf_samples_to_mesh(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        offset,
        scale,
    )
    # Convert to cuda: this is actually the slowest part...
    verts = torch.tensor(verts.astype(float), requires_grad = True, dtype=torch.float32, device=samples.device)
    faces = torch.tensor(faces.astype(float), requires_grad = False, dtype=torch.float32, device=samples.device)

    # Compute indices that will need to be recomputed at next iter
    with torch.no_grad():
        # first fetch bins that are activated
        voxel_origin = torch.tensor([-1., -1., -1.]).cuda()
        k = ((verts[:, 2] -  voxel_origin[2])/voxel_size + 0.0001).long()
        j = ((verts[:, 1] -  voxel_origin[1])/voxel_size + 0.0001).long()
        i = ((verts[:, 0] -  voxel_origin[0])/voxel_size + 0.0001).long()
        # find points around
        next_samples = i*N*N + j*N + k
        next_samples_ip = torch.minimum(i+1, torch.tensor([N-1]).cuda())*N*N + j*N + k
        next_samples_jp = i*N*N + torch.minimum(j+1, torch.tensor([N-1]).cuda())*N + k
        next_samples_kp = i*N*N + j*N + torch.minimum(k+1, torch.tensor([N-1]).cuda())
        next_samples_im = torch.maximum(i-1, torch.zeros(1).cuda())*N*N + j*N + k
        next_samples_jm = i*N*N + torch.maximum(j-1, torch.zeros(1).cuda())*N + k
        next_samples_km = i*N*N + j*N + torch.maximum(k-1, torch.zeros(1).cuda())

        next_indices = torch.cat((next_samples,next_samples_ip, next_samples_jp,next_samples_kp,next_samples_im,next_samples_jm, next_samples_km))

    return verts, faces, samples, next_indices



def create_mesh_optim_perceptual_noskip(
    decoder, latent_vec,  N=256, max_batch=32 ** 3, offset=None, scale=None, fourier = False,
    isolevel = 0.
):

    decoder.eval()

    with torch.no_grad():
        # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
        voxel_origin = [-1, -1, -1]
        voxel_size = 2.0 / (N - 1)

        overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
        samples = torch.zeros(N ** 3, 4)

        # transform first 3 columns
        # to be the x, y, z index
        samples[:, 2] = overall_index % N
        samples[:, 1] = (overall_index.long() / N) % N
        samples[:, 0] = ((overall_index.long() / N) / N) % N

        # transform first 3 columns
        # to be the x, y, z coordinate
        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
        samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

        num_samples = N ** 3

        samples.requires_grad = False
        samples.pin_memory()

        head = 0
        while head < num_samples:
            if fourier:
                sample_subset = fourier_transform(samples[head : min(head + max_batch, num_samples), 0:3].cuda())
            else:
                sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

            batch_vecs = latent_vec.view(latent_vec.shape[0], 1, latent_vec.shape[1]).repeat(1, sample_subset.shape[0], 1)
            input = torch.cat([batch_vecs.reshape(-1, latent_vec.shape[1]), sample_subset.reshape(-1, sample_subset.shape[-1])], dim=1)
            samples[head : min(head + max_batch, num_samples), 3] = decoder(input).squeeze(-1).detach().cpu()

            head += max_batch

        sdf_values = samples[:, 3]
        sdf_values = sdf_values.reshape(N, N, N)

    verts, faces = convert_sdf_samples_to_mesh(
        sdf_values.data.cpu() - isolevel,
        voxel_origin,
        voxel_size,
        offset,
        scale,
    )

    
    # first fetch bins that are activated
    k = ((verts[:, 2] -  voxel_origin[2])/voxel_size).astype(int)
    j = ((verts[:, 1] -  voxel_origin[1])/voxel_size).astype(int)
    i = ((verts[:, 0] -  voxel_origin[0])/voxel_size).astype(int)
    # find points around
    next_samples = i*N*N + j*N + k
    next_samples_ip = np.minimum(i+1,N-1)*N*N + j*N + k
    next_samples_jp = i*N*N + np.minimum(j+1,N-1)*N + k
    next_samples_kp = i*N*N + j*N + np.minimum(k+1,N-1)
    next_samples_im = np.maximum(i-1,0)*N*N + j*N + k
    next_samples_jm = i*N*N + np.maximum(j-1,0)*N + k
    next_samples_km = i*N*N + j*N + np.maximum(k-1,0)

    next_indices = np.concatenate((next_samples,next_samples_ip, next_samples_jp,next_samples_kp,next_samples_im,next_samples_jm, next_samples_km))

    return verts, faces, samples, next_indices
