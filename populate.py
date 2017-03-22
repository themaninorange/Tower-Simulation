import pandas as pd
import os
import random
import shutil
import numpy as np
from subprocess import call

os.getcwd()
os.chdir('/media/storage/CUDAClasses/CUDACLASS2017/JosephBrown/BigProject')
#os.chdir('/home/joseph/Dev/c_projects/CUDA/Energy Transfer Tower/')

pop = 6
folder = "runningpop1"

def firstInt(array): 
    #Source:
    #http://stackoverflow.com/questions/1586858/find-the-smallest-integer-not-in-a-list
    
    N = len(array)
    for cursor in range(N):
        target = array[cursor]
        while target < N  and target != array[target]:
            new_target = array[target]
            array[target] = target
            target = new_target
    
    # Pass 2, find first location where the index doesn't match the value
    for cursor in range(N):
        if array[cursor] != cursor:
            return cursor
    return N

def populate(folder, popsize):
    for i in range(popsize):
        call(["./newstructure.run", folder])
        print i

def readedges(folder, trial):
    with open(folder + "/" + trial + "/nodecon.txt") as f:
        lines = [line for line in f]
    numnodes = int(lines[0][7:-1])
    maxconns = int(lines[1][17:-1])
    nodei = []
    nodej = []
    edgek = []
    edgel = []
    
    for i in range(numnodes):
        values = lines[4 + i].split()
        nodei = nodei + [i]*len(values)
        nodej = nodej + [int(x) for x in values]
        edgek = edgek + [float(x) for x in lines[6 + numnodes + i].split()]
        edgel = edgel + [float(x) for x in lines[8 + 2*numnodes + i].split()]
    
    return pd.DataFrame({
        'nodei' : nodei,
        'nodej' : nodej,
        'edgek' : edgek,
        'edgel' : edgel }), numnodes, maxconns

def readpos(folder, trial):
    with open(folder + "/trial" + str(trial) + "/nodepos.txt") as f:
        lines = [line for line in f]
    posframe = pd.DataFrame([line.split() for line in lines[1:]])
    posframe.columns = ["px", "py", "pz", "anchor"]
    return posframe


def breed(folder, trial1, trial2):
    edges1, numnodes1, maxconns1 = readedges(folder, trial1)
    edges2, numnodes2, maxconns2 = readedges(folder, trial2)
    if numnodes1 != numnodes2:
        return
    if maxconns1 != maxconns2:
        return
    
    commonframe = pd.merge(edges1, edges2, how = 'inner', on=['nodei', 'nodej'])
    differentframe = pd.concat([edges1, edges2])
    differentframe = differentframe.reset_index(drop = True)
    edges_group = differentframe.groupby(list(differentframe.columns))
    idx = [x[0] for x in edges_group.groups.values() if len(x) ==1]
    differentframe = differentframe.reindex(idx)
    
    commonchoice1 = [i for i in range(commonframe.shape[0]) if random.random() < 0.5]
    commonchoice2 = [i for i in range(commonframe.shape[0]) if i not in commonchoice1]
    subcomm1 = commonframe.loc[commonchoice1][['edgek_x', 'edgel_x', 'nodei', 'nodej']]
    subcomm1.columns = ['edgek', 'edgel', 'nodei', 'nodej']
    subcomm2 = commonframe.loc[commonchoice2][['edgek_y', 'edgel_y', 'nodei', 'nodej']]
    subcomm2.columns = ['edgek', 'edgel', 'nodei', 'nodej']
    subdiff = differentframe.loc[random.sample(list(differentframe.index), int(np.mean((edges1.shape[0], edges2.shape[0])) - commonframe.shape[0]))]
    
    bigframe = pd.concat([subcomm1, subcomm2, subdiff])
    bigframe = bigframe.reset_index(drop = True)
    
    with open(folder + "/" + str(trial1) + "/nodepos.txt") as f:
        lines1 = [line for line in f]
    with open(folder + "/" + str(trial2) + "/nodepos.txt") as f:
        lines2 = [line for line in f]
    
    lines = [lines1[i] if random.random() < 0.5 else lines2[i] for i in range(len(lines1))]
    posframe = pd.DataFrame([[float(x) for x in line.split()[:-1]]+[line.split()[-1]] for line in lines[1:]])
    posframe.columns = ["px", "py", "pz", "anchor"]
    
    return bigframe, posframe, maxconns1


def writeUnstable(folder, nodecon, nodepos, maxconns):
    
    childfile = 'child' + str(firstInt([0] + [int(x[5:]) for x in os.listdir(folder + "/unstablechildren") if x[:5] == "child"]))
    os.mkdir(folder + "/unstablechildren/" + childfile)
    numnodes = nodepos.shape[0]
    
    filestring = "Nodes: " + str(numnodes) + \
    "\nMax Connections: " + str(maxconns) + \
    "\n\nbeami\n" + "\n".join(["\t"+"\t".join([str(x) for x in nodecon[nodecon['nodei'] == i]['nodej']]) for i in range(numnodes)]) + \
    "\n\nbeamk\n" + "\n".join(["\t"+"\t".join([str(x) for x in nodecon[nodecon['nodei'] == i]['edgek']]) for i in range(numnodes)]) + \
    "\n\nbeaml\n" + "\n".join(["\t"+"\t".join([str(x) for x in nodecon[nodecon['nodei'] == i]['edgel']]) for i in range(numnodes)])
    
    with open(folder + "/unstablechildren/" + childfile + "/nodecon.txt", "w") as f:
        f.write(filestring)

    filestring = "px\tpy\tpz\tanchor\n" + "\n".join(["\t".join([str(x) for x in nodepos.loc[i]]) for i in range(numnodes)])
    with open(folder + "/unstablechildren/" + childfile + "/nodepos.txt", "w") as f:
        f.write(filestring)

def mutate(nodecon, nodepos, conmuterate = 0.01, posmuterate = 0.1, conmutesize = 1.2, posmutesize = 0.2):
    posmute  = [i for i in range(nodepos.shape[0]) if random.random()<posmuterate]
    conmute  = [i for i in range(nodecon.shape[0]) if random.random()<conmuterate]
    conkill  = [i for i in range(nodecon.shape[0]) if random.random()<conmuterate and i not in conmute]
    
    for node in posmute:
        nodepos.loc[node,'px'] += (2*random.random() - 1)*posmutesize
        nodepos.loc[node,'py'] += (2*random.random() - 1)*posmutesize
        nodepos.loc[node,'pz'] += (2*random.random() - 1)*posmutesize
    
    for conn in conmute:
        nodei, nodej = nodecon.loc[conn][['nodei', 'nodej']]
        print("Mutating connection between %d and %d"%(nodei, nodej))
        print(nodecon[(nodecon.nodei == nodej) & (nodecon.nodej == nodei)])
        conj = nodecon[(nodecon.nodei == nodej) & (nodecon.nodej == nodei)].index[0]
        nodecon.loc[conn,'edgek'] *= conmutesize**(2*random.random()-1)
        nodecon.loc[conn,'edgel'] *= conmutesize**(2*random.random()-1)
        nodecon.loc[conj,'edgek'] *= conmutesize**(2*random.random()-1)
        nodecon.loc[conj,'edgel'] *= conmutesize**(2*random.random()-1)
    
    nodecon = nodecon[-(nodecon.nodei.isin(killnode) & nodecon.nodej.isin(killnode))]
    
    nodecon.reset_index(drop = True)
    
    return nodecon, nodepos

def stabilizeChildren(folder):
    for x in os.listdir(folder+'/unstablechildren'):
        call(['./newstructure.run', folder + '/unstablechildren/' + x, folder])
        shutil.rmtree(folder + '/unstablechildren/' + x)

def culling(folder):
    df = pd.read_csv(folder + '/summary.txt', sep = '\t', header = (0), lineterminator = '\n')
    dead = df.loc[df.goodness < df.goodness.astype(float).median()]
    alive = df.loc[-df.trial.isin(dead.trial)]
    for x in dead.trial:
        shutil.rmtree(folder + '/' + x)
    alive.to_csv(folder + '/summary.txt', sep='\t', header=True, index = False)

#populate the folder
populate(folder, pop)
#print(str(pop))
#print(folder)
#print(os.getcwd())

#On loop:
for i in range(100):
    
    #kill half
    culling(folder)
    
    #breed remaining half
    survivors = pd.read_csv(folder + '/summary.txt', sep = '\t', header = (0))
    goodnessp = np.array([float(x) for x in survivors.goodness])
    goodnessp /= sum(goodnessp)
        
    for j in range(pop - survivors.shape[0] + 2):
        
        #print([x for x in survivors.trial])
        
        breedingpair = np.random.choice([x for x in survivors.trial], 2, p = goodnessp)
        
        #print(breedingpair)
        
        nodecon, nodepos, maxconns = breed(folder, breedingpair[0], breedingpair[1])
        nodecon, nodepos = mutate(nodecon, nodepos)
        writeUnstable(folder, nodecon, nodepos, maxconns)
    
    stabilizeChildren(folder)









