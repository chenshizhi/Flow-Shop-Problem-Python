
#####################################################
# algorithm for random search
#####################################################

# Load the libraries needed
import time
import numpy as np
import pandas as pd
from copy import deepcopy
import random

#read 5 datasets
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

time_start=time.time() #recode the program start time

#calculate the makespan
#p_ij=dataset ,nbm=machine number, my_seq=current job sequence

def makespan(current_seq, p_ij, nbm):
    c_ij = np.zeros((nbm, len(current_seq) + 1))
    for j in range(1, len(current_seq) + 1):
        c_ij[0][j] = c_ij[0][j - 1] + p_ij[0][current_seq[j - 1]]

    for i in range(1, nbm):
        for j in range(1, len(current_seq) + 1):
            c_ij[i][j] = max(c_ij[i - 1][j], c_ij[i][j - 1]) + p_ij[i][current_seq[j - 1]]
    return current_seq,c_ij

#Apply the random search with 1000*n solution evaluation and calculate their makespan
def random_search(car_num):
    p_ij=car_num
    nbm=len(p_ij)
    nbj=len(p_ij[0])
    #original job sequence
    seq_origin=list(range(len(p_ij[0])))
    current_seq = []
    result_time=[]
    min_seq=[]
    loop_result={}
    for i in range(1000*nbj):
        seq = deepcopy(seq_origin)
        random.shuffle(seq)
        current_seq,c=makespan(seq,p_ij,nbm)
        mp=c[len(c)-1][len(c[0])-1]
        result_time.append(mp)
        loop_result[i]={'seq':current_seq,'makespan_values':mp}    
    min_mp=min(result_time)
    for loop_time in loop_result.values():
        if loop_time['makespan_values']==min_mp:
            min_seq.append(loop_time['seq'])          
    return min_seq,min_mp

#best_time has 30 makespan values , best_seq 30 seqs  
def main(n,car_num):
    best_time=[]
    best_seq=[]
    seed_result={}
    for times in range(n):
        np.random.seed(n*times)
        seq,min_mp = random_search(car_num)
#        best_seq.append(seq)
        best_time.append(min_mp)
        seed_result[times]={'seq':seq,'makespan_values':min_mp}
        #print 30 seeds best sequence and makespan for each 1000*n solution evaluations
        print('Seed {0} best sequence:{1} makespan values:{2}\n'.format(times+1,seq,min_mp))
    print('Makespan table:\n mininum:{0}, maximum:{1}, mean:{2}, standard deviation:{3}'.\
          format(np.min(best_time),np.max(best_time),np.mean(best_time),np.std(best_time)))
    for times in seed_result.values():
        if times['makespan_values']==np.min(best_time):
            best_seq.extend(times['seq'])
            
# remove the same best_seq,not-repeating
    NP_bestseq=[]
    [NP_bestseq.append(i) for i in best_seq if not i in NP_bestseq]
    NP_bestseq.sort()
    return best_time,NP_bestseq

# set 30 seeds 
def min_mp(car_num):
    n = 30
    best_time=main(n,car_num)
    return best_time

#chose the car number here
best_time=min_mp(car2) 

#recode the program running time
time_end=time.time() 
print(' Program Time Cost:',time_end-time_start,'s')

