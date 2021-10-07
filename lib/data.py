import numpy as np
import os
import torch
import torch.utils.data
import imageio
import lib.workspace as ws
from lib.utils import get_projection_torch3D

## List samples from a split json
def get_instance_filenames(split):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".npz"
                )
                npzfiles += [instance_filename]
    return npzfiles


def unpack_generic_image(filename):
    image = imageio.imread(filename).astype(float)/255.0
    return torch.tensor(image).float()


class Sketch2depth_norm(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        sketch_style
    ):
        self.data_source = data_source
        self.files_list =  get_instance_filenames(split)
        self.sketch_style = sketch_style

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        mesh_name = self.files_list[idx].split('.npz')[0]

        # At test time: fix view id to the first (random) one
        id = 0
        view_id = '{0:04d}'.format(id)

        # fetch image
        img_name = mesh_name
        image_filename = os.path.join(
            self.data_source, ws.sketches_subdir, img_name, view_id + '_' + self.sketch_style + '.png')
        image = unpack_generic_image(image_filename).unsqueeze(0)

        # fetch depth map
        depth_filename = os.path.join(
            self.data_source, ws.sketches_subdir, img_name, view_id + '_depth.png')
        depth = unpack_generic_image(depth_filename).unsqueeze(0)

        # fetch normal map
        normal_filename = os.path.join(
            self.data_source, ws.sketches_subdir, img_name, view_id + '_normal.png')
        normal = unpack_generic_image(normal_filename).unsqueeze(0)

        # fetch silhouette map
        silhouette_filename = os.path.join(
            self.data_source, ws.sketches_subdir, img_name, view_id + '_silhouette.png')
        silhouette = unpack_generic_image(silhouette_filename).unsqueeze(0)

        # fetch cameras
        camera_filename = os.path.join(
            self.data_source, ws.sketches_subdir, img_name, view_id + '.meta'
        )
        meta = np.loadtxt(camera_filename)

        RT1, K1 = get_projection_torch3D(-90-meta[0], meta[1], meta[2], img_w=meta[3], img_h=meta[3], RCAM=False)
        RT1, K1 = torch.tensor(RT1).float(), torch.tensor(K1).float()
        return image, depth, normal, silhouette, RT1, mesh_name
