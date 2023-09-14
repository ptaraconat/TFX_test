import time 
import multiprocessing as mp

def do_something():
    print('Sleeping 1second ...')
    time.sleep(1)
    print('Done Sleeping')

start = time.time()
do_something()
do_something()
end = time.time()
print('Script duration : ', end-start)

##################
print('Distributed Processes ')
# Create Processes 
p1 = mp.Process(target=do_something)
p2 = mp.Process(target=do_something)
# Run Processes 
start = time.time()
p1.start()
p2.start()
end = time.time()
print('Script duration : ', end-start) # Actually this value is not accurate. 

##############################################################################
# How to wait that both processes are finished to calculate the time spent ?? 
# use the join method 
print('Distributed Processes ')
# Create Processes 
p1 = mp.Process(target=do_something)
p2 = mp.Process(target=do_something)
# Run Processes 
start = time.time()
p1.start()
p2.start()
p1.join() #tells p1 to wait all running processes 
p2.join() #tells p2 to wait all running processes 
end = time.time()
print('Script duration : ', end-start) #

#############################################################################
######### More taks in parallel
start = time.time()
processes = []
for _ in range(10):
    p = mp.Process(target = do_something)
    p.start()
    processes.append(p)
for p in processes : 
    p.join()
end = time.time()
print('Script duration : ', end-start) #
