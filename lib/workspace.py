import json
import os

model_params_subdir = "ModelParameters"
optimizations_subdir = "Optimizations"
optimizations_meshes_subdir = "output"
specifications_filename = "specs.json"
sketches_subdir = "Sketches"


def load_experiment_specifications(experiment_directory):
    filename = os.path.join(experiment_directory, specifications_filename)

    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include specifications file "
            + '"specs.json"'.format(experiment_directory)
        )

    return json.load(open(filename))
