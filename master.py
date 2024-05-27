import sys

import numpy as np
from mpi4py import MPI

from config import INPUT_IMAGE_PATH, OUTPUT_IMAGE_PATH, CANNY_THRESHOLD1, CANNY_THRESHOLD2
from utils import load_image, save_image, apply_canny


def master():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    image = load_image(sys.argv[1])
    height, width = image.shape
    num_fragment = size - 1
    fragment_height = height // num_fragment

    for i in range(1, size):
        start_row = (i - 1) * fragment_height
        end_row = start_row + fragment_height if i != num_fragment else height
        fragment = image[start_row:end_row, :]
        comm.send(fragment, dest=i, tag=11)

    result = np.zeros_like(image)
    for i in range(1, size):
        edge_fragment = comm.recv(source=i, tag=22)
        start_row = (i - 1) * fragment_height
        end_row = start_row + fragment_height if i != num_fragment else height
        result[start_row:end_row, :] = edge_fragment

    save_image(sys.argv[2], result)


def master_2():
    image = load_image(sys.argv[1])
    edges = apply_canny(image, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
    save_image(sys.argv[2], edges)
