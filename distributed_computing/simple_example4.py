import time 
import concurrent.futures

def do_something(seconds):
    print(f'Sleeping {seconds} second ...')
    time.sleep(seconds)
    return f'Done Sleeping ... {seconds}'

start = time.time()
with concurrent.futures.ProcessPoolExecutor() as executor : 
    f1 = executor.submit(do_something, 1)
    f2 = executor.submit(do_something, 1)
    print(f1.result())
    print(f2.result())
end = time.time()
print('Script duration : ', end-start) #

#######################################
start = time.time()
with concurrent.futures.ProcessPoolExecutor() as executor :
    secs = [5,4,3,2,1] 
    results = [executor.submit(do_something, sec) for sec in secs]
    for f in concurrent.futures.as_completed(results): 
        print(f.result())
end = time.time()
print('Script duration : ', end-start) #

#######################################
start = time.time()
with concurrent.futures.ProcessPoolExecutor() as executor :
    secs = [5,4,3,2,1] 
    results = executor.map(do_something, secs)
    for result in results : 
        print(result) #here results are printed in the order they are executed. v

end = time.time()
print('Script duration : ', end-start) #