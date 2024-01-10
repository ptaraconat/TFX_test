from mpi4py import MPI

comm = MPI.COMM_WORLD
worker = comm.Get_rank()
size = comm.Get_size()

local_data = worker ** 3
data = comm.gather(local_data, root = 0)

print(worker, data)
print(worker, local_data)

if worker == 0 : 
	print('gather result: ', data)