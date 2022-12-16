import os, os.path
import itertools
import multiprocessing
import sys
import glob

# This block of code enables us to call the script from command line.                                                                                
def execute(process):                                                             
    os.system(f'python3.11 {process}') 

if __name__ == '__main__':

    nproc = 4
    clean_output = False

    if len(sys.argv) > 1:
        nproc = int(sys.argv[1])
        if len(sys.argv) > 2:
            clean_output = bool(sys.argv[2])

    if clean_output:
        filelist = glob.glob(os.path.join("gaout", "*"))
        for f in filelist:
            os.remove(f)
        # os.system(f'rm gaout/*')

    # Creating the tuple of all the processes
    python_script = ["gas_for_qaps.py"]
    alg_names = ["trad_ga"]
    psizes = ['50', '100', '200']
    max_fes = ['10000']
    runs = list(map(str, range(1, 3)))

    all_processes = itertools.product(
        python_script,
        alg_names,
        psizes,
        max_fes,
        runs
    )

    all_processes = [" ".join(p) for p in all_processes]                                                                                                                    
    process_pool = multiprocessing.Pool(processes = nproc)                                                        
    process_pool.map(execute, all_processes)