from mpi4py import MPI

def reduce_func(a,b):
	return a+b

comm = MPI.COMM_WORLD
worker = comm.Get_rank()
size = comm.Get_size()

#data = comm.reduce(worker, op= reduce_func, root = 0)
data = comm.reduce(worker, op= MPI.SUM, root = 0)

print(worker, data)

if worker == 0 : 
	print('final result : ', data)