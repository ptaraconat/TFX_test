import multiprocessing
from multiprocessing import Pool

n_cpu = multiprocessing.cpu_count()

print(n_cpu)

def f(x):
  return x

with Pool(processes=n_cpu) as pool:
    for i in pool.imap_unordered(f, range(10)):
       print(i)

    for i in pool.imap(f, range(10)):
       print(i)

