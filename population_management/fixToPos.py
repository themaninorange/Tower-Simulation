import pandas as pd
import os
import random
import shutil
import numpy as np
import sys
from subprocess import call

os.getcwd()
os.chdir('/media/storage/CUDAClasses/CUDACLASS2017/JosephBrown/BigProject')
#os.chdir('/home/joseph/Dev/c_projects/CUDA/Energy Transfer Tower/')

if len(sys.argv) >= 2:
    folder = sys.argv[1]
else:
    folder = "dump"

if len(sys.argv) >= 3:
    trialFolder = sys.argv[2]
else:
    trialFolder = -1

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
    #genes = []
    
    for i in range(numnodes):
        values = lines[4 + i].split()
        nodei = nodei + [i]*len(values)
        nodej = nodej + [int(x) for x in values]
        edgek = edgek + [float(x) for x in lines[6 + numnodes + i].split()]
        edgel = edgel + [float(x) for x in lines[8 + 2*numnodes + i].split()]
    
    """
    if len(lines) >= 9 + 4*numnodes:
        for i in range(numnodes):
            genes = genes + [int(x) for x in lines[10 + 3*numnodes + i].split()]
    else:
    	genes = [0]*len()
    """
    
    return pd.DataFrame({
        'nodei' : nodei,
        'nodej' : nodej,
        'edgek' : edgek,
        'edgel' : edgel #,
        #'gene'  : genes 
        }), numnodes, maxconns

def readpos(folder, trial):
    with open(folder + "/" + str(trial) + "/nodepos.txt") as f:
        lines = [line for line in f]
    posframe = pd.DataFrame([line.split() for line in lines[1:]])
    posframe.columns = ["px", "py", "pz", "anchor"]
    posframe.ix[:, ["px", "py", "pz"]] = posframe.ix[:, ["px", "py", "pz"]].astype(float)
    return posframe

def readjuncs(folder, trial, edgeframe):
    with open(folder + "/" + str(trial) + "/juncval.txt") as f:
        lines = [line for line in f]
    
    numnodes = (len(lines) - 3)/2
    
    thetas = []
    omegas = []
    edgei = []
    edgej = []
    
    for i in range(numnodes):
        dungus = list(edgeframe.index[(edgeframe.nodei == i)])
        numBeams = len(dungus)
        
        thetas = thetas + [float(x) for x in lines[1 + i].split()]
        omegas = omegas + [float(x) for x in lines[3 + numnodes + i].split()]
        for j in dungus:
            edgei  = edgei  + [j]*numBeams
            edgej  = edgej  + dungus
    
    return pd.DataFrame({
        'edgei' : edgei,
        'edgej' : edgej,
        'theta' : thetas,
        'omega' : omegas #,
        #'gene'  : genes 
        })

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
    
    naivechoice   = [i for i in range(commonframe.shape[0]) if random.random() < 0.25]
                    # Want a one quarter chance of selecting an edge because it will probably be duplicated.
    print(naivechoice)
    commonconj    = [commonframe.index[(commonframe.nodei == commonframe.nodej[choice].item()) & \
                                       (commonframe.nodej == commonframe.nodei[choice].item())].item() for choice in naivechoice]
                    # Find the conjugate connection entry for each choice.
    print(commonconj)
    commonchoice1 = list(set(naivechoice + commonconj))
    
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

def writeUnstable2(folder, nodepos, nodecon, juncval, maxconns):
    
    def take(x):
        if x in taken:
            return True
        taken.append(x)
        return False
    
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

    stringlistthetas = []
    stringlistomegas = []
    for i in range(numnodes):
        
        indices = [idx for idx in juncval.index if i in juncval.center[idx]]
        dangle = juncval.loc[indices]
        
        tupTheta = [list(x) for x in zip([list(y) for y in juncval.nodes[indices]], juncval.theta[indices])]
        tupOmega = [list(x) for x in zip([list(y) for y in juncval.nodes[indices]], juncval.omega[indices])]
        
        dongle = [[[a for a in x[0] if a != i], x[1]] if len(x[0]) == 3 else [[a for a in x[0] if a != i]*2, x[1]] for x in tupTheta]
        dongle.sort(key=lambda thing: (thing[0][0], thing[0][1]))
        taken = []
        dramgus = ["%.3f"%x[1] for x in dongle if not take(x[0])]
        stringlistthetas.append('\t'.join(dramgus))
        
        dongle = [[[a for a in x[0] if a != i], x[1]] if len(x[0]) == 3 else [[a for a in x[0] if a != i]*2, x[1]] for x in tupOmega]
        dongle.sort(key=lambda thing: (thing[0][0], thing[0][1]))
        taken = []
        dramgus = ["%.3f"%x[1] for x in dongle if not take(x[0])]
        stringlistomegas.append('\t' + '\t'.join(dramgus))
    
    filestring = "defaultTheta\n" + '\n'.join(stringlistthetas) + '\n\nangleSpringConst\n' + '\n'.join(stringlistomegas)
    with open(folder + "/unstablechildren/" + childfile + "/juncval.txt", "w") as f:
        f.write(filestring)
    print("Genome written to %s"%childfile)

def mutate1(nodecon, nodepos, conmuterate = 0.01, posmuterate = 0.1, conmutesize = 1.2, posmutesize = 0.2):
    posmute  = [i for i in range(nodepos.shape[0]) if random.random()<posmuterate and nodepos.anchor[i] != 'T']
    conmute  = [i for i in range(nodecon.shape[0]) if random.random()<conmuterate]
    conkill  = [i for i in range(nodecon.shape[0]) if random.random()<conmuterate and i not in conmute]
    killpair = [(nodecon.nodei[conn].item(), nodecon.nodej[conn].item()) for conn in conkill]
    killlist = list(set(killpair + [(x[1], x[0]) for x in killpair]))
    
    for node in posmute:
    	nodepos.set_value(node, 'px', nodepos.loc[node,'px'].item() + (2*random.random() - 1)*posmutesize)
    	nodepos.set_value(node, 'py', nodepos.loc[node,'py'].item() + (2*random.random() - 1)*posmutesize)
    	nodepos.set_value(node, 'pz', nodepos.loc[node,'pz'].item() + (2*random.random() - 1)*posmutesize)
    
    for conn in conmute:
        nodei, nodej = nodecon.loc[conn][['nodei', 'nodej']]
        print("Mutating connection between %d and %d"%(nodei, nodej))
        print(nodecon[(nodecon.nodei == nodei) & (nodecon.nodej == nodej)])
        print(nodecon[(nodecon.nodei == nodej) & (nodecon.nodej == nodei)])
        if nodecon[(nodecon.nodei == nodej) & (nodecon.nodej == nodei)].shape[0] >0:
            conj = nodecon[(nodecon.nodei == nodej) & (nodecon.nodej == nodei)].index[0]
        else:
            conj = nodecon.shape[0]
            print(conj)
            nodecon.append(pd.DataFrame([0,0,nodej, nodei]), ignore_index = True)
            print(nodecon.shape[0])
        nodecon.set_value(conn, 'edgek', nodecon.loc[conn,'edgek'].item() * conmutesize**(2*random.random()-1))
        nodecon.set_value(conn, 'edgel', nodecon.loc[conn,'edgel'].item() * conmutesize**(2*random.random()-1))
        nodecon.set_value(conj, 'edgek', nodecon.loc[conj,'edgek'].item())
        nodecon.set_value(conj, 'edgel', nodecon.loc[conj,'edgel'].item())
    
    for pair in killlist:
        nodecon = nodecon[(nodecon.nodei != pair[0]) | (nodecon.nodej != pair[1])]
    
    nodecon.reset_index(drop = True)
    
    return nodecon, nodepos

def mutate2(nodecon, nodepos, conmuterate = 0.01, posmuterate = 0.1, conmutesize = 1.2, posmutesize = 0.2):
    posmute  = [i for i in range(nodepos.shape[0]) if random.random()<posmuterate and nodepos.anchor[i] != 'T']
    conmute  = [i for i in range(nodecon.shape[0]) if random.random()<conmuterate]
    conkill  = [i for i in range(nodecon.shape[0]) if random.random()<conmuterate and i not in conmute]
    killpair = [(nodecon.nodei[conn].item(), nodecon.nodej[conn].item()) for conn in conkill]
    killlist = list(set(killpair + [(x[1], x[0]) for x in killpair]))
    newpair  = [(i,j) for i in range(numnodes) for j in range(numnodes) if (i != j) and (np.random.rand() < conmuterate)]
    
    for node1 in posmute:
    	nodepos.set_value(node1, 'px', nodepos.loc[node1,'px'].item() + (2*random.random() - 1)*posmutesize)
    	nodepos.set_value(node1, 'py', nodepos.loc[node1,'py'].item() + (2*random.random() - 1)*posmutesize)
    	nodepos.set_value(node1, 'pz', nodepos.loc[node1,'pz'].item() + (2*random.random() - 1)*posmutesize)
    	
    	#Identify all connections involving the mutated node.  Change the lengths accordingly.
    	conframe1 = nodecon.loc[(nodecon.nodei == node1)]
    	conframe2 = nodecon.loc[(nodecon.nodej == node1)]
    	for i in conframe1.index:
    	    node2 = conframe1.nodej[i].item()
    	    conframe1.edgel[i] = np.sqrt((nodepos.px[node1] - nodepos.px[node2])**2 + \
    	                                 (nodepos.py[node1] - nodepos.py[node2])**2 + \
    	                                 (nodepos.pz[node1] - nodepos.pz[node2])**2)
    	for i in conframe2.index:
    	    node2 = conframe2.nodei[i].item()
    	    conframe2.edgel[i] = np.sqrt((nodepos.px[node1] - nodepos.px[node2])**2 + \
    	                                 (nodepos.py[node1] - nodepos.py[node2])**2 + \
    	                                 (nodepos.pz[node1] - nodepos.pz[node2])**2)
    
    
    for conn in conmute:
        nodei, nodej = nodecon.loc[conn][['nodei', 'nodej']]
        #print("Mutating connection between %d and %d"%(nodei, nodej))
        #print(nodecon[(nodecon.nodei == nodei) & (nodecon.nodej == nodej)])
        #print(nodecon[(nodecon.nodei == nodej) & (nodecon.nodej == nodei)])
        if nodecon[(nodecon.nodei == nodej) & (nodecon.nodej == nodei)].shape[0] >0:
            conj = nodecon[(nodecon.nodei == nodej) & (nodecon.nodej == nodei)].index[0]
        else:
            conj = nodecon.shape[0]
            nodecon.append(pd.DataFrame([0,0,nodej, nodei]), ignore_index = True)
        
        nodecon.set_value(conn, 'edgek', nodecon.loc[conn,'edgek'].item() * conmutesize**(2*random.random()-1))
        nodecon.set_value(conj, 'edgek', nodecon.loc[conn,'edgek'].item())
    
    for pair in killlist:
        nodecon = nodecon[(nodecon.nodei != pair[0]) | (nodecon.nodej != pair[1])]
    
    meank = np.mean(nodecon.edgek)
    
    for pair in newpair:
        if nodecon[(nodecon.nodei == pair[0]) & (nodecon.nodej == pair[1])].shape[0] == 0:
            tempdist = np.sqrt((nodepos.px[pair[0]] - nodepos.px[pair[1]])**2 + \
                               (nodepos.py[pair[0]] - nodepos.py[pair[1]])**2 + \
                               (nodepos.pz[pair[0]] - nodepos.pz[pair[1]])**2)
            nodecon.append(pd.DataFrame([[meank, tempdist, pair[0], pair[1]],
                                         [meank, tempdist, pair[1], pair[0]]]), ignore_index = True)
    
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

def fixLengths(nodecon, nodepos):
    newcons = nodecon.copy()
    newlengths = [lenVec(nodepos.ix[newcons.nodei[i], ["px", "py", "pz"]] - nodepos.ix[newcons.nodej[i], ["px", "py", "pz"]]) for i in newcons.index]
    newcons.ix[:, "edgel"] = newlengths
    return newcons

def lenVec(x):
    return np.sqrt(x.dot(x))

def getAngle(x, y):
    return np.arccos(x.dot(y)/(lenVec(x)*lenVec(y)))

def fixAngles(juncval, nodecon, nodepos):
    newjunc = juncval.copy()
    angleLegs = [(nodecon.ix[juncval.ix[i, "edgei"], ["nodei", "nodej"]], nodecon.ix[juncval.ix[i, "edgej"], ["nodei", "nodej"]]) for i in newjunc.index]
    newangles = [getAngle(nodepos.ix[angleLegs[i][0][0], ["px", "py", "pz"]] - nodepos.ix[angleLegs[i][0][1], ["px", "py", "pz"]], 
                          nodepos.ix[angleLegs[i][1][0], ["px", "py", "pz"]] - nodepos.ix[angleLegs[i][1][1], ["px", "py", "pz"]]) \
                          for i in range(len(angleLegs))]
    newjunc.ix[:, "theta"] = newangles
    return newjunc


#init stuff
if folder not in os.listdir('./'):
    os.mkdir(folder)

if 'unstablechildren' not in os.listdir('./' + folder):
    os.mkdir(folder + '/unstablechildren')

if 'summary.txt' not in os.listdir('./' + folder):
    with open(folder + '/summary.txt', 'w') as f:
        f.write('trial\tgoodness\n')

#####

nodecon, numnodes, maxconns = readedges(folder, trialFolder)
nodepos = readpos(folder, trialFolder)
if "juncval.txt" in os.listdir(folder + "/" + str(trial)):
    juncval = readjuncs(folder, trialFolder, nodecon)
else:
    juncval = randomJuncs(nodepos, nodecon)

legs = [(set(nodecon.ix[juncval.edgei[junc], ['nodei', 'nodej']]), set(nodecon.ix[juncval.edgej[junc], ['nodei', 'nodej']])) for junc in juncval.index]
juncval['nodes'] = [x[0].union(x[1]) for x in legs]
juncval['center'] = [x[0].intersection(x[1]) for x in legs]

nodecon = fixLengths(nodecon, nodepos)
juncval = fixAngles(juncval, nodecon, nodepos)

writeUnstable2(folder, nodepos, nodecon, juncval, maxconns)









