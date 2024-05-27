import sys

from mpi4py import MPI

from master import master, master_2
from worker import worker


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0 and comm.Get_size() < 2:
        master_2()
        return

    if rank == 0:
        master()

    else:
        worker()


if __name__ == '__main__':
    main()
