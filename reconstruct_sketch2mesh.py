import torch
import torch.utils.data as data_utils
import os
import json
import lib
import lib.workspace as ws
import imageio
import numpy as np
import shutil
import argparse
import trimesh
from lib.utils import (fourier_transform, add_common_args, ObjectMetricTracker,
                        myChamferDistance, contours_pointcloud, pack_mesh_and_render,
                        get_renderer_cameras_lights, filter_contours_exterior, filter_contours_input)
from tqdm import tqdm
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
import copy


def load_model_parameters(experiment_directory, checkpoint, encoder, decoder):
    filename_encoder = os.path.join(
        experiment_directory, ws.model_params_subdir, "encoder_" + checkpoint + ".pth"
    )
    encoder.load_state_dict(torch.load(filename_encoder)["model_state_dict"])
    filename_decoder = os.path.join(
        experiment_directory, ws.model_params_subdir, "decoder_" + checkpoint + ".pth"
    )
    decoder.load_state_dict(torch.load(filename_decoder)["model_state_dict"])
    return

# For parallelizing the processing loop accross N threads
def split(items, n, N):
    """
    Split the list items in N elements of same length, 
    and returns the n-th one
    """
    l = len(items) // N
    # The last one takes the rest
    if n == N:
        return items[(n-1) * l: ]
    return items[(n-1) * l: n * l]

def main_function(experiment_directory, sketch_style, n_iters, marching_cubes_resolution_in=128):
    ########################################################################################
    # Read specs
    specs = ws.load_experiment_specifications(experiment_directory)
    data_source = specs['DataSource']
    test_split_file = specs['TestSplit']
    arch_encoder = __import__('lib.models.' + specs['NetworkEncoder'], fromlist=['ResNet'])
    arch_decoder = __import__('lib.models.' + specs['NetworkDecoder'], fromlist=['Decoder'])
    latent_size = specs['CodeLength']

    ########################################################################################
    # Create + load network, and create output directories
    encoder = arch_encoder.ResNet(latent_size, specs['Depth'], norm_type = specs['NormType']).cuda()
    decoder = arch_decoder.Decoder(latent_size, **specs['NetworkSpecs']).cuda()
    encoder = torch.nn.DataParallel(encoder)
    decoder = torch.nn.DataParallel(decoder)

    optimization_dir = os.path.join(args.experiment_directory, ws.optimizations_subdir, 'latest')
    optimization_meshes_dir = os.path.join(optimization_dir, ws.optimizations_meshes_subdir)
    if not os.path.isdir(optimization_meshes_dir):
        os.makedirs(optimization_meshes_dir)

    load_model_parameters(experiment_directory, 'latest', encoder, decoder)
    encoder.eval()

    ############################################
    # DATASET CREATION
    ############################################
    with open(test_split_file, "r") as f:
        test_split = json.load(f)
    dataset_test = lib.data.Sketch2depth_norm(data_source, test_split, sketch_style)
    # Splitting data for this thread
    dataset_test.files_list = split(dataset_test.files_list, args.n, args.N)

    torch.manual_seed(12)
    loader_test = data_utils.DataLoader(
        dataset_test, batch_size=1, shuffle=False,
        num_workers=2, drop_last=False,
    )

    ############################################
    # LOOPING ON INSTANCES
    ############################################

    pbar1 = tqdm(total=len(dataset_test), position=0)
    for image, depth_gt, normal_gt, silhouette, extrinsic, name in loader_test:
        _ = pbar1.update()

        ########################################################################################
        # Save input image + GT depth, normals, silhouette, mesh
        instance_dir = os.path.join(optimization_meshes_dir, name[0])
        if not os.path.exists(instance_dir):
            os.makedirs(instance_dir)
        # Input sketch
        image_filename = os.path.join(instance_dir, 'input.png')
        image_export = 255*image[0].permute(1,2,0).numpy()
        imageio.imwrite(image_filename, image_export.astype(np.uint8))
        # Depth
        depth_filename = os.path.join(instance_dir, 'gt_depth.png')
        imageio.imwrite(depth_filename, (255. * depth_gt[0,0].numpy()).astype(np.uint8))
        # Normals
        normal_filename = os.path.join(instance_dir, 'gt_normal.png')
        imageio.imwrite(normal_filename, (255. * normal_gt[0,0].numpy()).astype(np.uint8))
        # Silhouette
        silh_filename = os.path.join(instance_dir, 'gt_silhouette.png')
        imageio.imwrite(silh_filename, (255. * silhouette[0,0].numpy()).astype(np.uint8))
        # GT mesh
        gt_mesh_filename = os.path.join('dataset/Meshes', name[0][11:], 'isosurf.obj')
        _ = shutil.copyfile(gt_mesh_filename, os.path.join(instance_dir, 'gt_mesh.obj'))

        ########################################################################################
        # Create renderer
        # pytorch3d cameras
        R_cuda = extrinsic[:, 0:3, 0:3].float().cuda().clone().detach()
        t_cuda = extrinsic[:, 0:3, 3].float().cuda().clone().detach()
        # Renderer components
        cameras, renderer_pytorch3D, lights = get_renderer_cameras_lights(R_cuda, t_cuda)

        ########################################################################################
        # Load GT mesh (for metrics computations)
        meshes_gt = trimesh.load(gt_mesh_filename)
        gt_vertices, gt_faces = torch.tensor(meshes_gt.vertices).float().cuda(), torch.tensor(meshes_gt.faces).long().cuda()
        meshes_gt = Meshes(gt_vertices.unsqueeze(0), gt_faces.unsqueeze(0))

        ########################################################################################
        # Create the target contours as 2D point cloud
        tgt_contours_out_flat = torch.tensor(filter_contours_input(image[0,0].detach().cpu().numpy())).cuda().reshape(-1)
        
        X, Y = torch.meshgrid(torch.arange(0, image[0,0].shape[0]).cuda(), torch.arange(0, image[0,0].shape[0]).cuda())
        grid_map = torch.cat([X[:,:,None], Y[:,:,None]], 2).float()
        grid_map = grid_map / (0.5 * image[0,0].shape[0]) - 1.
        grid_map_flat = grid_map.reshape(-1, 2)
        tgt_pc = grid_map_flat[tgt_contours_out_flat < 0.1]

        ########################################################################################
        # Reconstruct initial 3D mesh from sketch
        depth_gt, normal_gt = depth_gt.cuda(), normal_gt.cuda()
        # get latent code from image
        latent, _, _, _, _ = encoder(image)
        # get mesh from latent code
        verts, faces, samples, next_indices = lib.mesh.create_mesh_optim_perceptual_noskip(
            decoder, latent, N=marching_cubes_resolution_in, max_batch=int(2 ** 18), fourier=specs['NetworkSpecs']['fourier']
        )
        # store mesh
        mesh_filename = os.path.join(instance_dir, 'pred_mesh.ply')
        lib.mesh.write_verts_faces_to_file(verts, faces, mesh_filename)

        ########################################################################################
        # Render and store contour of the predicted mesh
        verts_dr = torch.tensor(verts[:, :].copy(), dtype=torch.float32, requires_grad = False).cuda()  # [num_vertices, XYZ]
        faces_dr = torch.tensor(faces[:, :].copy()).cuda()  # [num_faces, 3]
        out_contour = pack_mesh_and_render(verts_dr, faces_dr, renderer_pytorch3D, cameras, lights)
        # Save
        image_out_export = 255*out_contour[0,0].detach().cpu().numpy()
        image_out_filename = os.path.join(instance_dir, 'output.png')
        imageio.imwrite(image_out_filename, image_out_export.astype(np.uint8))


        ############################################
        # MAIN REFINEMENT LOOP
        ############################################
        # Metric tracker for this shape
        metrics = ObjectMetricTracker()

        # Optimization variables / optimizer:
        latent_for_optim = latent.clone().detach().requires_grad_(True)
        decoder_for_optim = copy.deepcopy(decoder)
        lr= 5e-5 # or 1e-4
        optimizer = torch.optim.Adam([latent_for_optim], lr=lr)
        decoder.eval()

        pbar2 = tqdm(total=n_iters + 1, desc=f'{name[0][20:26]}', leave=False, position=1)
        for e in range(n_iters + 1):
            _ = pbar2.update()
            optimizer.zero_grad()
            # First create mesh with Marching Cubes
            verts, faces, samples, next_indices = lib.mesh.create_mesh_optim_faster(
                samples, next_indices, decoder_for_optim, latent_for_optim, N=128, max_batch=int(2 ** 18), fourier=specs['NetworkSpecs']['fourier']
            )
            xyz_upstream = verts
            faces_upstream = faces
            
            """
            Render contours
            """
            with torch.no_grad():
                out_contour = pack_mesh_and_render(xyz_upstream, faces_upstream, renderer_pytorch3D, cameras, lights)
            bin_contours = out_contour

            """
            Compute 2D CHD loss
            """
            bin_contours_for_chd = torch.tensor(filter_contours_exterior(bin_contours[0,0].detach().cpu().numpy(), dilation=True)).cuda()[None, None]

            # Get 2D positions of contour points
            coords_2d_uv_contours = contours_pointcloud(xyz_upstream, faces_upstream, bin_contours_for_chd, renderer_pytorch3D, cameras)
            # Chamfer computation
            dist1, dist2 = myChamferDistance(coords_2d_uv_contours, tgt_pc)
            # Bidirectional, ie
            # contours of the generated shape attract their closest target point
            # AND
            # contours of the target attract their closest generated shape point
            loss_chd = 0.1*(dist1.mean() + dist2.mean())

            """
            MeshSDF backwards pass: from gradients on the vertices to the latent code
            """
            loss = loss_chd
            # store upstream gradients
            loss.backward()
            dL_dx_i = xyz_upstream.grad
            # + take care of NaN gradients possibly happening, TODO: find source of issues
            dL_dx_i[torch.isnan(dL_dx_i)] = 0

            # Use vertices to compute full backward pass
            optimizer.zero_grad()
            xyz = verts.clone().detach().requires_grad_(True)
            if specs['NetworkSpecs']['fourier']:
                xyz_mapped = fourier_transform(xyz)
            else:
                xyz_mapped = xyz
            latent_inputs = latent_for_optim.expand(xyz.shape[0], -1)
            inputs = torch.cat([latent_inputs, xyz_mapped], 1).cuda()
            # Compute normals
            pred_sdf = decoder_for_optim(inputs)
            loss_normals = torch.sum(pred_sdf)
            loss_normals.backward(retain_graph = True)
            normals = xyz.grad/torch.norm(xyz.grad, 2, 1).unsqueeze(-1)
            # + take care of NaN gradients possibly happening, TODO: find source of issues
            normals[torch.isnan(normals)] = 0

            # now assemble inflow derivative
            optimizer.zero_grad()
            dL_ds_i = -torch.matmul(dL_dx_i.unsqueeze(1), normals.unsqueeze(-1)).squeeze(-1)

            # finally assemble full backward pass
            loss_backward = torch.sum(dL_ds_i * pred_sdf)
            loss_backward = loss_backward

            # and backward here of all the previous loss terms
            loss_backward.backward()
            # + update params
            optimizer.step()

            """
            Save outputs + compute losses
            """
            if e == n_iters // 2:
                # Save optimized mesh, mid procedure
                mesh_filename = os.path.join(instance_dir, f'optim_{e}.ply')
                lib.mesh.write_verts_faces_to_file(verts, faces, mesh_filename)
            if e % 50 == 0:                  
                # Store contours for vis
                image_filename = os.path.join(instance_dir, f'{e:04d}.png')
                image_out_export = 255*bin_contours[0,0].detach().cpu().numpy()
                imageio.imwrite(image_filename, image_out_export.astype(np.uint8))

                # Compute 3D chamfer
                meshes_dr = Meshes(xyz_upstream.unsqueeze(0), faces_upstream.unsqueeze(0))
                meshes_gt_pts = sample_points_from_meshes(meshes_gt)
                meshes_dr_pts = sample_points_from_meshes(meshes_dr)
                metric3d, _ = chamfer_distance(meshes_gt_pts, meshes_dr_pts)
                metrics.append('chd', metric3d.detach().cpu().numpy().item(), e)

        pbar2.close()            
        
        # Save optimized mesh
        verts, faces, samples, next_indices = lib.mesh.create_mesh_optim_perceptual_noskip(
            decoder_for_optim, latent_for_optim, N=marching_cubes_resolution_in, max_batch=int(2 ** 18), fourier=specs['NetworkSpecs']['fourier']
        )

        # store mesh
        mesh_filename = os.path.join(instance_dir, 'optim_final.ply')
        lib.mesh.write_verts_faces_to_file(verts, faces, mesh_filename)

        # Store the metrics history for this shape
        metrics.save(os.path.join(instance_dir, 'metrics.pck'))

    pbar1.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Sketch2Mesh: reconstruct & refine meshes from input sketches')
    arg_parser.add_argument(
        '--experiment',
        '-e',
        dest='experiment_directory',
        required=True,
        help='The experiment directory. This directory should include '
        + 'experiment specifications in specs.json, and logging will be '
        + 'done in this directory as well.',
    )
    arg_parser.add_argument('--out_dir', default=ws.optimizations_meshes_subdir, type=str, 
        help='Reconstructed and refines shapes will be stored in this sub-directory'
    )
    arg_parser.add_argument('--sketch_style', default='fd', type=str, 
        help='Sketching style: [fd | suggestive | handdrawn (for cars only)]'
    )
    # Multiprocess: this thread is the n-th of N processes
    arg_parser.add_argument("--n", default=1, type=int, help="Thread rank")
    arg_parser.add_argument("--N", default=1, type=int, help="Total thread number")

    add_common_args(arg_parser)
    args = arg_parser.parse_args()
    ws.optimizations_meshes_subdir = args.out_dir

    if 'chair' in args.experiment_directory:
        n_iters = 500
    else:
        n_iters = 250

    print('=====================================')
    print(f'Generative model: {args.experiment_directory} .')
    print(f'Results saved in {ws.optimizations_meshes_subdir} .')
    print(f'Refinement using 2D CHD for {n_iters} steps')
    print(f'Using sketching style: {args.sketch_style}')
    print(f'This is thread {args.n} of {args.N}.')
    print('=====================================')

    main_function(args.experiment_directory, args.sketch_style, n_iters)
