import pickle
import numpy as np
import argparse
import glob
import tqdm

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--directory",
    "-d",
    dest="dir",
    required=True,
    help="directory containing optimization results in the form of a list of folders - one per instance",
)
args = arg_parser.parse_args()

#### Scan all instances that were optimized
metrics_files = glob.glob(args.dir + '/*/metrics.pck')
metrics_values = []

print('Reading metrics...')
for file in tqdm.tqdm(metrics_files):
  m = pickle.load(open(file, 'rb'))
  metrics_values.append(m.metrics['chd'])

metrics_values = np.array(metrics_values) * 1000

#### Print average values
print(f'Across {len(metrics_files)} shapes:')
print(f' - Average initial 3D Chamfer: {metrics_values.mean(axis=0)[0]}')
print(f' - After refinement: {metrics_values.mean(axis=0)[-1]}')
