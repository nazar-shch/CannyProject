import sys
import numpy as np
from mpi4py import MPI
from config import CANNY_THRESHOLD1, CANNY_THRESHOLD2
from utils import load_image, save_image, apply_canny


def master():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    image = load_image(sys.argv[1])
    height, width = image.shape
    num_fragment = size - 1
    fragment_height = height // num_fragment
    overlap = 10  # Number of rows to overlap between fragments

    # Send fragments to workers
    for i in range(1, size):
        start_row = max((i - 1) * fragment_height - overlap, 0)
        end_row = min(i * fragment_height + overlap, height)
        fragment = image[start_row:end_row, :]
        comm.send(fragment, dest=i, tag=11)

    result = np.zeros_like(image)
    # Receive edge-detected fragments from workers
    for i in range(1, size):
        edge_fragment = comm.recv(source=i, tag=22)
        start_row = (i - 1) * fragment_height
        end_row = start_row + fragment_height

        if i == 1:
            result[start_row:end_row, :] = edge_fragment[:fragment_height, :]
        elif i == num_fragment:
            result[start_row:, :] = edge_fragment[overlap:, :]
        else:
            result[start_row:end_row, :] = edge_fragment[overlap:overlap + fragment_height, :]

    save_image(sys.argv[2], result)


def master_2():
    image = load_image(sys.argv[1])
    edges = apply_canny(image, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
    save_image(sys.argv[2], edges)


if __name__ == "__main__":
    master()
