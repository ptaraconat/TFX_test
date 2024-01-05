import os as os
import multiprocessing as mp
import time

def worker(number_list):
    time_start = time.time()
    returned_list = list()
    for number in number_list :
        returned_list.append(number ** 2.)
    time_end = time.time()
    print("worker time : ", time_end - time_start)

#the number of tasks based on SLURM environment variables
num_nodes = int(os.environ.get("SLURM_NNODES", 1))
cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
total_tasks = int(os.environ.get("SLURM_NTASKS", 1))
num_tasks = int(os.environ.get("SLURM_NTASKS", 1))

print('num nodes : ', num_nodes)
print('cpu per task : ', cpus_per_task)
print('total tasks : ' , total_tasks)
print('Num tasks : ', num_tasks)

n_processes = mp.cpu_count()
# set input array 
N = 100000
beg = 1
end = 11
delta = (end - beg)/(N-1)
array = [beg + i*delta for i in range(N) ]
# split array 
sublist_length = N//n_processes
sublists = [array[i * sublist_length: (i + 1) * sublist_length] for i in range(n_processes-1)]
sublists.append(array[(n_processes-1) * sublist_length:])
# Init processes 
processes = list()
for i in range(n_processes) :
    process = mp.Process(target = worker, args = (sublists[i],))
    processes.append(process)
# Run processes 
start_time = time.time()
for process in processes :
    process.start()
# Join processes 
for process in processes :
    process.join()
end_time = time.time()
# Disp total time 
print('total time : ', end_time - start_time)
# Run worker on entire array, without multi processing 
start_time = time.time()
worker(array)
end_time = time.time()
print('time without mp : ', end_time - start_time)
print('number of CPU available : ',mp.cpu_count())