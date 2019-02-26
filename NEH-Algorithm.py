
#####################################################
# algorithm for neh
#####################################################

# Load the libraries needed
import pandas as pd
import numpy as np

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

#chose the car number
p_ij=car2

nbm=len(p_ij)
nbj=len(p_ij[0])

#calculate makespan
def makespan_neh(current_seq, p_ij, nbm):
    c_ij = np.zeros((nbm, len(current_seq) + 1))
    for j in range(1, len(current_seq) + 1):
        c_ij[0][j] = c_ij[0][j - 1] + p_ij[0][current_seq[j - 1]]

    for i in range(1, nbm):
        for j in range(1, len(current_seq) + 1):
            c_ij[i][j] = max(c_ij[i - 1][j], c_ij[i][j - 1]) + p_ij[i][current_seq[j - 1]]
    return c_ij[nbm - 1][len(current_seq)]

#calculate the job's total processing time in all machines
def sum_processing_time(index_job, data, nb_machines):
    sum_p = 0
    for i in range(nb_machines):
        sum_p += data[i][index_job]
    return sum_p


#Sort the current sequence by job's total processing time(descending)
def order_neh(data, nb_machines, nb_jobs):
    my_seq = []
    for j in range(nb_jobs):
        my_seq.append(j)
    return sorted(my_seq,key=lambda x:sum_processing_time(x, data, nb_machines),reverse=True)

#insert the new job and obtain the new sequence
def insertion(sequence, index_position, value):
    new_seq = sequence[:]
    new_seq.insert(index_position, value)
    return new_seq

#run the neh
#calculate the new makespan after insert the new job
#Compare the makespan of each sequence, retaining the best sequence and makespan
def neh(data, nb_machines, nb_jobs):
    order_seq = order_neh(data, nb_machines, nb_jobs)
    seq_current = [order_seq[0]]
    #obtain partial and complete sequences [n(n+1)/2]-1 times
    for i in range(1, nb_jobs):
        min_cmax = float("inf")
        for j in range(0, i + 1):
            tmp_seq = insertion(seq_current, j, order_seq[i])
            cmax_tmp = makespan_neh(tmp_seq, data, nb_machines)
            print(tmp_seq, cmax_tmp)
            if min_cmax > cmax_tmp:
                best_seq = tmp_seq
                min_cmax = cmax_tmp
        seq_current = best_seq
    return seq_current,makespan_neh(seq_current, data, nb_machines)

# print the NEH seq and it's makespan
seq, cmax = neh(p_ij, nbm, nbj)
print('Number of Machines:{0},Number of Jobs:{1}'.format(nbm,nbj))
print("NEH sequence:", seq)
print("Makespan:",cmax)








