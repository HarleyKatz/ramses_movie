import argparse
import configparser
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

from astropy.cosmology import FlatLambdaCDM
from matplotlib.colors import ListedColormap
from pathlib import Path
from scipy.io import FortranFile
from tqdm import tqdm

# Initialize the arguments
parser = argparse.ArgumentParser()
parser.add_argument("config_file_name")
args = parser.parse_args()
config_file = args.config_file_name # Get the config file name

# Check that the config file actually exists and is a file
f = Path(config_file)
if not f.is_file():
    print(f"{config_file} is not a file")
    exit()

# Initialize the config parser
config = configparser.ConfigParser()
config.read(args.config_file_name)

# Get the config params (in dictionary format)
params = config._sections

movie_path = params["GENERAL"]["movie_path"]  # Path to movie files output by ramses
save_location = params["GENERAL"]["output_path"]
min_frame = int(params["GENERAL"]["min_frame"]) # minimum frame to render
max_frame = int(params["GENERAL"]["max_frame"]) # maximum frame to render
mpi = config["GENERAL"].getboolean("mpi")

# If we are using mpi, initialize the MPI
if mpi:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

# Do something
quants = {}
for key in params:
    if key != "GENERAL":
        try:
            var_cmap = matplotlib.cm.get_cmap(params[key]["cmap"])
        except ValueError:
            print(f"colormap: {cmap} not available")
            exit()

        # Only linear for now
        alpha_method = params[key]["alpha_method"]
        if alpha_method == "none":
            s_cmap = var_cmap
        elif alpha_method == "linear":
            s_cmap = var_cmap(np.arange(var_cmap.N))
            s_cmap[:,-1] = np.linspace(float(params[key]["alpha_min"]), float(params[key]["alpha_max"]), var_cmap.N)
            s_cmap = ListedColormap(s_cmap)
        else:
            print(f"alpha method {alpha_method} not allowed")
            exit()

        quants[key] = {
                "name": key,
                "min": float(params[key]["min"]),
                "max": float(params[key]["max"]),
                "data": np.array([]),
                "cmap": s_cmap,
                "log": config[key].getboolean("log"),
                }

# Get the cosmology (Hard coded for sphinx???)
info = open(f"{movie_path}info_{str(1).zfill(5)}.txt").readlines()
H0 = float(info[11].split("=")[-1].strip())
omega_m = float(info[12].split("=")[-1].strip()) 
cosmo = FlatLambdaCDM(H0=H0, Om0=omega_m)

# Loop over the frames

# Only make one figure (saves memory)
plt.figure(figsize=(float(params["GENERAL"]["fig_x"]),float(params["GENERAL"]["fig_y"])))

if mpi:
    lr = np.arange(rank+min_frame,max_frame+1,size)
else:
    lr = np.arange(min_frame,max_frame+1)

for i in tqdm(lr):
    all_ok = True

    # Load in the data
    for q in quants:
        fname = f"{quants[q]['name']}_{str(i).zfill(5)}.map"
        ffile = FortranFile(f"{movie_path}{fname}")

        [time, fdw, fdh, fdd] = ffile.read_reals('d')
        [frame_nx, frame_ny] = ffile.read_ints()
        data = np.array(ffile.read_reals('f4'), dtype=np.float64)

        age_in_myr = cosmo.age(1./time - 1.).value * 1e3

        try: 
            quants[q]["data"] = data.reshape(frame_nx,frame_ny)
            l_min = quants[q]["data"][quants[q]["data"] > 0].min()
            l_max = quants[q]["data"][quants[q]["data"] > 0].max()
            quants[q]["data"][quants[q]["data"] < l_min] = 1e-5 * l_min
        except ValueError:
            all_ok = False

    if all_ok:
        plt.clf()
        for q in quants:
            vals = quants[q]["data"]

            if quants[q]["log"]:
                vals = np.log10(vals)

            plt.imshow(vals,vmin=quants[q]["min"],vmax=quants[q]["max"],cmap=quants[q]["cmap"])

        plt.savefig(f"{save_location}/img_{str(i).zfill(5)}.png",bbox_inches="tight",pad_inches=0,dpi=int(params["GENERAL"]["dpi"]))


