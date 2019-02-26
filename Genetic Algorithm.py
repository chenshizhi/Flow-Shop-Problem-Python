
#####################################################
# algorithm for GA
#(Algo4:2-point crossover(C1) with shift mutation(SM))
#####################################################

import numpy as np
import pandas as pd
import random

# read the 5 datasets
car1 = pd.read_csv('car1.csv') 
car1 = np.array(car1)
car2 = pd.read_csv('car2.csv') 
car2 = np.array(car2)
car3 = pd.read_csv('car3.csv') 
car3 = np.array(car3)
car4 = pd.read_csv('car4.csv') 
car4 = np.array(car4)
car5 = pd.read_csv('car5.csv') 
car5 = np.array(car5)

#choose the car number
p_ij=car2

#define the NEH_SEQ 
NEH_seq=[4, 7, 5, 6, 2, 0, 3, 1]

#5 cars NEH_seq
#car1 [7, 0, 4, 8, 2, 10, 3, 6, 5, 1, 9]   
#car2 [4, 7, 5, 6, 2, 0, 3, 1]
#car3 [18, 7, 11, 15, 19, 4, 0, 9, 12, 2, 1, 17, 8, 6, 5, 10, 3, 14, 13, 16]
#car4 [17, 12, 9, 0, 16, 18, 8, 1, 11, 2, 7, 3, 4, 14, 10, 15, 5, 6, 19, 13]
#car5 [13, 19, 28, 4, 17, 10, 16, 12, 5, 8, 1, 0, 2, 20, 6, 22, 9, 23, 7, 3, 
#      15, 29, 25, 26, 14, 11, 24, 21, 18, 27]

nbm=len(p_ij)
nbj=len(p_ij[0])

print('Number of Machines:{0},Number of Jobs:{1}'.format(nbm,nbj))

#set the parameters
Npop = 30    # Number of population
Pc = 1       # Probability of crossover
Pm = 0.8     # Probability of mutation
D=0.95       #Threshold parameter
sig=0.99     #sigema=0.99

print('The parameters we chosen:\n''Population size:{0}\nCrossover probability:{1}\n\
Initial mutation probability:{2}\nThreshold parameter:{3}\n'.format(Npop,Pc,Pm,D))

#Number of evaluations
stopGeneration = 1000*nbj
#(it will take a long time to run the algorithm)
#(to test the algrithm we can set stopGeneration small)
#stopGeneration = 100

#calculate the makespan
def makespan_GA(current_seq, p_ij, nbm):
    c_ij = np.zeros((nbm, len(current_seq) + 1))
    for j in range(1, len(current_seq) + 1):
        c_ij[0][j] = c_ij[0][j - 1] + p_ij[0][current_seq[j - 1]]

    for i in range(1, nbm):
        for j in range(1, len(current_seq) + 1):
            c_ij[i][j] = max(c_ij[i - 1][j], c_ij[i][j - 1]) + p_ij[i][current_seq[j - 1]]
    return c_ij[nbm - 1][nbj]

#initialize the population and append the neh_seq to the initial population
def initialization(Npop):
    pop = []
    for i in range(Npop):
        p = list(np.random.permutation(nbj))
        while p in pop:
            p = list(np.random.permutation(nbj))
        pop.append(p)
    pop.append(NEH_seq)
    return pop

#select the population
def selection(pop):
    #popobj[0]=makespans ,popobj[1]=order from 0 to 30 ,totally 31
    popObj = []
    for i in range(len(pop)):
        popObj.append([makespan_GA(pop[i],p_ij,nbm),i])
#sort by makespan,so the plpobj[1]will not be in order     
    popObj.sort()
    distr = []
    distrInd = []
    
    for i in range(len(pop)):
        #append the makespan's order in the distrInd 
        #(while the makespan in pop are in order ) ascending in makespan index
        distrInd.append(popObj[i][1])
        #Select parent 1 using 2k/M(M+1) fitness_rank distribution
        prob = (2*(len(pop)-i)) / (len(pop) * (len(pop)+1))
        distr.append(prob)
    
    parents = []
    for i in range(len(pop)):
        #Select parent 2 using uniform distribution
        parents.append(list(np.random.choice(distrInd, 1, p=distr)))
        parents[i].append(np.random.choice(distrInd))
    return parents

#2-point crossover (C2)
def crossover(parents):
    pos = list(np.random.permutation(np.arange(nbj-1)+1)[:2])
    
    if pos[0] > pos[1]:
        t = pos[0]
        pos[0] = pos[1]
        pos[1] = t
    
    child = list(parents[0])
    
    for i in range(pos[0], pos[1]):
        child[i] = -1
    
    p = -1
    for i in range(pos[0], pos[1]):
        while True:
            p = p + 1
            if parents[1][p] not in child:
                child[i] = parents[1][p]
                break   
    return child

#shift mutation
def mutation(sol):
    pos = list(np.random.permutation(np.arange(nbj))[:2])
    
    if pos[0] > pos[1]:
        t = pos[0]
        pos[0] = pos[1]
        pos[1] = t
    
    remJob = sol[pos[1]]
    
    for i in range(pos[1], pos[0], -1):
        sol[i] = sol[i-1]
        
    sol[pos[0]] = remJob
    
    return sol

#Update the population
def elitistUpdate(oldPop, newPop):
    bestSolInd = 0
    bestSol = makespan_GA(oldPop[0],p_ij,nbm)
    
    for i in range(1, len(oldPop)):
        tempObj = makespan_GA(oldPop[i],p_ij,nbm)
        if tempObj < bestSol:
            bestSol = tempObj
            bestSolInd = i         
    rndInd = random.randint(0,len(newPop)-1)
    newPop[rndInd] = oldPop[bestSolInd]
    return newPop

# find the best solution
def findBestSolution(pop):
    bestObj = makespan_GA(pop[0],p_ij,nbm)
    avgObj = bestObj
    bestInd = 0
    for i in range(1, len(pop)):
        tObj = makespan_GA(pop[i],p_ij,nbm)
        avgObj = avgObj + tObj
        if tObj < bestObj:
            bestObj = tObj
            bestInd = i
            
    return bestInd, bestObj, avgObj/len(pop)

# Run the algorithm for 'stopGeneration'(1000*n) times generation
def Loop(Npop):
    
    Pm = 0.8
    # Creating the initial population
    population = initialization(Npop)
    for i in range(stopGeneration):
        # Selecting parents
        parents = selection(population)
        childs = []
    
        # Apply crossover
        for p in parents:
            r = random.random()
            y = random.random()
            if r < Pc:
                childs.append(crossover([population[p[0]], population[p[1]]]))
            else:
                if y < 0.5:
                   childs.append(population[p[0]])    
                else:
                   childs.append(population[p[1]])
    
        # Apply mutation 
        for c in childs:
            r = random.random()
            if r < Pm:
                c = mutation(c)
                # Update the population
        population_new = elitistUpdate(population, childs)
        # Results Time
        bestSol, minObj, avgObj = findBestSolution(population_new)
        mpset=[]
        mpset.append(minObj)
        if minObj/avgObj > D:
            Pm=sig*Pm
    return min(mpset)

# recode 30 seeds results for each run
def ran_seed(carnum):
    result=[]
    for times in range(30):
        np.random.seed(30*times)
        min_makespan=(Loop(Npop))
        result.append(min_makespan)
        print('seed {0}: makespan value {1}'.format(times+1,min_makespan))
    return result

#print the makespan table 
result=ran_seed(p_ij)
print('\nmakespan table:\n''mininum:{0}, maximum:{1}, mean:{2}, standard deviation:{3}'.\
format(np.min(result),np.max(result),np.mean(result),np.std(result)))













