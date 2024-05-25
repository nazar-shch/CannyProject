from mpi4py import MPI

from config import CANNY_THRESHOLD1, CANNY_THRESHOLD2
from utils import apply_canny


def worker():
    comm = MPI.COMM_WORLD
    fragment = comm.recv(source=0, tag=11)
    edges = apply_canny(fragment, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
    comm.send(edges, dest=0, tag=22)
