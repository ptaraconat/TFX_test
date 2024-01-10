from mpi4py import MPI

comm = MPI.COMM_WORLD
worker = comm.Get_rank()
size = comm.Get_size()

if worker == 0 : 
	data = [{'data for '+str(i) : i} for i in range(size)]
	print(data)
else : 
	data = None 
data = comm.bcast(data, root = 0)
print(worker, data)