from threading import Thread, Lock
import time 

# storage is shared between threads 
database_value = 0 

def increase(lock):
    global database_value

    lock.acquire()
    local_copy = database_value
    # process data 
    local_copy += 1
    time.sleep(0.1)
    database_value = local_copy
    lock.release()
    # between lock.acquire and lock.release there is no switch between threats. 

def increase_bis(lock):
    global database_value

    with lock : 
        local_copy = database_value
        # process data 
        local_copy += 1
        time.sleep(0.1)
        database_value = local_copy
    # between lock.acquire and lock.release there is no switch between threats. 


if __name__ == "__main__" : 

    lock = Lock()
    print('start value', database_value)

    # declare threads
    thread1 = Thread(target = increase, args = (lock,))
    thread2 = Thread(target = increase, args = (lock,))
    # start threads
    thread1.start()
    thread2.start()
    #join threads
    thread1.join()
    thread2.join()

    print('end value', database_value)
    print('end main')
