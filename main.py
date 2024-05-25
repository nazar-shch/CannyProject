from mpi4py import MPI

from master import master
from worker import worker


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        master()

    else:
        worker()


if __name__ == '__main__':
    main()
