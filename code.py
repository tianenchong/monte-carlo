import numpy as np
import copy 
from math import exp
from matplotlib import pyplot as plt
from PIL import Image
from random import choice, uniform

# function finding 4 neighboring corners given a lattice site
def neighbourCoord(i,j):
    # left
    if i == 0:  # extreme left
        l = [9, j]
    else:
        l = [i-1, j]
    # right
    if i == 9:  # extreme right
        r = [0, j]
    else:
        r = [i+1, j]
    # up
    if j == 0: # extreme up
        u = [i, 9]
    else:
        u = [i, j-1]
    # down
    if j == 9: # extreme down
        d = [i, 0]
    else:
        d = [i, j+1]
        
    return l,r,u,d
    
# function finding all 8 neighbors given a lattice site  
def neighbourCoordAll(i,j):
    l,r,u,d = neighbourCoord(i,j)   # use same corners
    # up
    if j == 0:  # extreme up
        ur = [0 if i == 9 else i+1, 9] # extreme & normal right
        ul = [9 if i == 0 else i-1, 9]  # extreme & normal left
    else:       # normal up
        ur = [0 if i == 9 else i+1, j-1] # extreme & normal right
        ul = [9 if i == 0 else i-1, j-1] # extreme & normal left
    # down
    if j == 9:  # extreme down
        dr = [0 if i == 9 else i+1, 0] # extreme & normal right
        dl = [9 if i == 0 else i-1, 0]  # extreme & normal left
    else:  # normal down
        dr = [0 if i == 9 else i+1, j+1] # extreme & normal right
        dl = [9 if i == 0 else i-1, j+1]  # extreme & normal left
        
    return [l,r,u,d,ur,dr,dl,ul]

# function multiplying and summing spin with 4 neighboring corners given a lattice site
def neighbourMulSum(i,j,grid):
    l,r,u,d = neighbourCoord(i,j)
    return grid[i][j]*grid[l[0]][l[1]]+grid[i][j]*grid[r[0]][r[1]]+grid[i][j]*grid[u[0]][u[1]]+grid[i][j]*grid[d[0]][d[1]]
    
# function averaging spin of a local site (9 sites), given a center lattice site
def neighbourAllAvg(i,j,grid):
    neighbours = [l,r,u,d,ur,dr,dl,ul] = neighbourCoordAll(i,j)
    mtot = 0
    for n in neighbours:
        mtot = mtot + grid[n[0]][n[1]]
    return (grid[i][j]+mtot)/9
    
# function calculating total Hamiltonian of the system
def Htotal(grid):
    H = 0
    for i in range(10):
        for j in range(10):
            H = H + neighbourMulSum(i,j,grid)

    return H*(-1/2)
    
# function calculating total magnetization of the system
def Mtotal(grid):
    mag = 0
    for i in range(10):
        for j in range(10):
            mag = mag + grid[i][j]

    return mag*(1/100)
    
# function calculating probability distribution 
def probDist(grid):
    mLocalAvgs = [] 
    Ts = list(np.linspace(4,0.1,5,endpoint=True)) # generate 5 T values between 4 and 0.1
    # go through average spin of local site for all lattice sites
    for i in range(10):
        for j in range(10):
            mLocalAvgs.append(neighbourAllAvg(i,j,grid))
    # plot probability distribution histogram with 20 bins between -1 and 1
    plt.hist(mLocalAvgs,range=(-1,1),bins=np.linspace(-1,1,20,endpoint=True))
    plt.title("T = {}".format(Ts[count])) 
    plt.show()

# function rounding off new T values to match the original T values for question 4
def newTs(Ts, Tsq4):
    return [(int(x*10) if (x*10-int(x*10))<0.5 else int(x*10)+1)/10 for x in Tsq4] # explicit rounding workaround (rounding not working properly in Python)

# function generating Latex table data
def genReverseLatexTable(title,x,y):
    n_x = [i for i in x]
    n_y = [i for i in y]
    n_x.reverse()
    n_y.reverse()
    print(title)
    for i in range(len(n_x)):
        print("{:f} & {:f} \\\\".format(n_x[i],n_y[i]))

# variables initialization
run = 2 # independent simulation
ms = 500 # MC steps
ps = 10 # L^2 update moves per MC step
avHs = [] # average energy for all T, for all independent simulation 
avMs = [] # average magnetization for all T, for all independent simulation 
cvs = [] # heat capacity for all T, for all independent simulation 
suss = [] # susceptibility for all T, for all independent simulation 
Ts = [i/10 for i in range(1,41)] # initialize 40 T values between 4.0 to 0.1
Ts.reverse() # reverse T values to descending order
Tlen = len(Ts)
imageindex = 0
q4Ts = newTs(Ts, list(np.linspace(4,0.1,5,endpoint=True))) # 5 T values for q4 between 4.0 to 0.1 inclusive
halfms = int(ms/2)
lastm = ms - 1
lastp = ps - 1
lastrun = run - 1
grids = list(np.zeros((Tlen,10, 10), dtype='int')) # initialize grid to all 0
count = 0 # counter for probability distribution graph

for n in range(run): # for each independent simulation
    grid = [ [choice([-1,1]) for x in range(10)] for y in range(10) ] # generate a random grid
    h = Htotal(grid) # get total energy
    mag = Mtotal(grid) # get total magnetization
    avHTs = [] # average energy for all T, for this simulation
    avMTs = [] # average magnetization for all T, for this simulation
    cvTs = [] # heat capacity for all T, for this simulation
    susTs = [] # susceptibility for all T, for this simulation
    for T in Ts: # for each T
        if n == lastrun: # only save and show grid for last simulation
            # save current grid
            for j in range(10):
                for i in range(10):
                    grids[imageindex][j][i] = grid[j][i]
            imageindex = imageindex + 1
        
        htot = 0
        mtot = 0
        hsqtot = 0
        msqtot = 0
        for m in range(ms): # for each MC step
            for p in range(ps): # for each update move
                # test water
                gridtest = [ [grid[j][i] for i in range(10)] for j in range(10)] # copy grid
                randx = choice(range(10)) # generate random coordinate x
                randy = choice(range(10)) # generate random coordinate y
                Hbp = neighbourMulSum(randx,randy,gridtest) * -1.0 # energy before test spin flip
                gridtest[randx][randy]=-gridtest[randx][randy] # test spin flip
                Hap = neighbourMulSum(randx,randy,gridtest) * -1.0  # energy after test spin flip
                deltaH = Hap - Hbp # change in energy after test flip

                # actual spin flip if conditions are satisfied
                r = uniform(0,1) # generate uniform random value between 0 and 1
                if deltaH < 0 or r < exp(-deltaH/T): # change in energy is negative or probability is satisfied
                    grid[randx][randy]=-grid[randx][randy] # actual spin flip
                    h = h + deltaH # update energy value
                    mag = mag + grid[randx][randy]*2/100 # update magnetization value
                    
            if m >= halfms: # more than half of all MC steps then is considered thermal equilibrium
                htot = htot + h # sum all energy for averaging later
                mtot = mtot + mag # sum all magnetization for averaging later
                hsqtot = hsqtot + h**2 # sum all energy squared for averaging later
                msqtot = msqtot + mag**2 # sum all magnetization squared for averaging later
                if T in q4Ts and n == lastrun and m == lastm: # generate probability distribution only for specified T (one of those 5 values) matching any of the original 40 T values and only during last simulation and last MC step
                    probDist(grid)
                    count = count + 1
                    
        avHT = htot/halfms # average energy for this T
        avMT = mtot/halfms # average magnetization for this T
        avsqHT = avHT**2 # average energy squared for this T
        avsqMT = avMT**2 # average magnetization squared for this T
        sqavHT = hsqtot/halfms # squared energy average for this T
        sqavMT = msqtot/halfms # squared magnetization average for this T
        avHTs.append(avHT) # save average energy for this T
        avMTs.append(avMT) # save average magnetization for this T
        cvTs.append(1/(T**2)*(sqavHT - avsqHT)) # save heat capacity for this T
        susTs.append(1/T*(sqavMT - avsqMT)) # save susceptibility for this T
        
    avHs.append(avHTs) # save average energy for all T for this simulation
    avMs.append(avMTs) # save average magnetization for all T for this simulation
    cvs.append(cvTs) # save heat capacity for all T for this simulation
    suss.append(susTs) # save susceptibility for all T for this simulation
    
# set the layout for illustration of the system at all 40 T values to 8 by 5
rowsize = 8
colsize = 5
fig,ax=plt.subplots(nrows=rowsize,ncols=colsize,figsize=(20,20)) # figures formatting and cosmetics
fig.tight_layout(pad=3.0) # figures formatting and cosmetics

# show illustration of the system at all 40 T values
for i in range(Tlen):
    ax[int(i/5), i%5].get_yaxis().set_visible(False)
    ax[int(i/5), i%5].get_xaxis().set_visible(False)
    ax[int(i/5), i%5].set_title("T = {}".format(Ts[i]),fontsize=7)
    ax[int(i/5)][i%5].imshow(grids[i],cmap='gray',vmin=-1,vmax=1,aspect='equal',interpolation='nearest')
plt.show()


av2Hs = [] # average of the average energy for all T, for all simulation
avcvs = [] # average of heat capacity for all T, for all simulation
for j in range(Tlen):  # for each T as column
    avHsTot = 0
    cvsTot = 0
    for i in range(run):  # for each independent simulation as column
        avHsTot = avHsTot + avHs[i][j] # sum average energy from all independent simulations for this T for further averaging later
        cvsTot = cvsTot + cvs[i][j] # sum average heat capacity from all independent simulations for this T for further averaging later
    av2Hs.append(avHsTot/run) # further averaging for average energy for this T
    avcvs.append(cvsTot/run) # further averaging for average heat capacity for this T

plt.subplot(2, 1, 1) # plot layout setting
plt.plot(Ts,av2Hs, 'r-') # plot average energy for all T averaged from all simulation
plt.title('Graph of Average Energy and Heat Capacity vs Temperature') # title
plt.ylabel('Average Energy') # y label
genReverseLatexTable('Average Energy',Ts,av2Hs) # generate Latex dataset

plt.subplot(2, 1, 2) # plot layout setting
plt.plot(Ts,avcvs, 'b-') # plot heat capacity for all T averaged from all simulation
plt.xlabel('Temperature') # x label
plt.ylabel('Heat Capacity') # y label
genReverseLatexTable('Heat Capacity',Ts,avcvs) # generate Latex dataset
plt.show() # show the plot

plt.subplot(2, 1, 1) # plot layout setting
plt.plot(Ts,avMs[lastrun], 'r-') # only plot average magnetization for all T for last simulation
plt.title('Graph of Average Magnetization and Susceptibility vs Temperature') # title
plt.ylabel('Average Magnetization') # y label
genReverseLatexTable('Average Magnetization',Ts,avMs[lastrun]) # generate Latex dataset

plt.subplot(2, 1, 2) # plot layout setting
plt.plot(Ts,suss[lastrun], 'b-') # only plot susceptibility for all T for last simulation
plt.xlabel('Temperature') # x label
plt.ylabel('Susceptibility') # y label
genReverseLatexTable('Susceptibility',Ts,suss[lastrun]) # generate Latex dataset
plt.show() # show the plot

    
    
          
    
        


