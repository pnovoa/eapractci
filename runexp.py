import os, os.path
import itertools
import multiprocessing
import sys
import glob

# This block of code enables us to call the script from command line.                                                                                
def execute(process):                                                             
    os.system(f'python3 {process}') 

if __name__ == '__main__':

    nproc = 5
    clean_output = True
    maxfes = 10000
    if len(sys.argv) > 1:
        nproc = int(sys.argv[1])
        if len(sys.argv) > 2:
            maxfes = int(sys.argv[2])
            if len(sys.argv) > 3:
                clean_output = sys.argv[3].lower() == 'true'

    if clean_output:
        filelist = glob.glob(os.path.join("gaout", "*"))
        for f in filelist:
            os.remove(f)
        # os.system(f'rm gaout/*')

    # Creating the tuple of all the processes
    python_script = ["runga.py"]
    alg_names = ['LAM']
    psizes = ['50']
    crossr = ['0.8']
    mutr = ['0.01']
    max_fes = [str(maxfes)]
    runs = list(map(str, range(1, 21)))

    all_processes = itertools.product(
        python_script,
        alg_names,
        psizes,
        crossr,
        mutr,
        max_fes,
        runs
    )

    all_processes = [" ".join(p) for p in all_processes]                                                                                                                    
    process_pool = multiprocessing.Pool(processes = nproc)                                                        
    process_pool.map(execute, all_processes)