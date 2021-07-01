# Utilities for converting RAMSES movie outputs into awesome pngs

## Setup
First set up a python virtual environment:
e.g. `python -m venv env`

Next activate it:
`source ./env/bin/activate`

Install the requirements:
`pip install -r requirements.txt`

## How to use

- First set up a configuration file. See the example in the configs directory
- Make sure that the names of the params in the config match the names of the files in the movie directory
- Create an output directory for where the images should be stored
- Decide whether to use mpi or not

## How to launch

With MPI:
`mpirun -np 24 python movie.py ./configs/config.ini`

Without MPI:
`python movie.py ./configs/config.ini`

## Parameter definitions

### General
- `movie_path: <location of the RAMSES movie files>`
- `output_path: <location of where to output the PNGs>`
- `mpi: <Whether to use MPI (True or False)>`
- `min_frame: <index of the frame to start the movie>`
- `max_frame: <index of the frame to end the movie>`
- `fig_x: <width of the image in inches>`
- `fig_y: <height of the image in inches>`
- `dpi: <resolution of the image>`

### Field specific
- `name: <name of the field>`
- `log: <Whether to log scale the quantity>`
- `min: <minimum value for the color scale>`
- `max: <maximum value for the color scale>`
- `cmap: <name of the matplotlib colormap>`
- `alpha_method: <for custom color bars that fade (linear or none)>`
- `alpha_min: <minimum alpha for custom colorbars>`
- `alpha_max: <maximum alpha for color bars>` 

