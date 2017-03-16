import os
from subprocess import call

os.getcwd()
os.chdir('/media/storage/CUDAClasses/CUDACLASS2017/JosephBrown/BigProject')

popsize = 100

for i in range(popsize):
    call(["./newstructure.run", "runningpop1"])


