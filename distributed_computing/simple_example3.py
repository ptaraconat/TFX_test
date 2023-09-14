import time 
import multiprocessing as mp

def do_something(seconds):
    print(f'Sleeping {seconds} second ...')
    time.sleep(seconds)
    print('Done Sleeping')

start = time.time()
processes = []
for _ in range(10):
    p = mp.Process(target = do_something, args = [1.5])
    p.start()
    processes.append(p)
for p in processes : 
    p.join()
end = time.time()
print('Script duration : ', end-start) #