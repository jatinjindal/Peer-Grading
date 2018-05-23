import random
import numpy as np
import itertools

#finds the kendell tau distance
def kendral_dist(standard,arg):
	dist= 0
	for i in range(0,len(standard)):
		for j in range(i+1,len(standard)):
			if(arg[i]>arg[j]):
				dist = dist + 1
	return dist

#finds the ranking given by grader given the initial ranking 
def grading(initial_ranking,skills):
	perm = list(itertools.permutations(initial_ranking))
	prob = []
	total = 0
	for i in perm:
		prob.append(np.exp(-skills * kendral_dist(initial_ranking,i)))
		total = total + prob[-1]
	
	for i in range(0,len(prob)):
		prob[i] = prob[i]/total

	r = np.random.uniform(0, 1)
	s = 0
	for j in range(0,len(prob)):
		s += prob[j]
		if s >= r:
			break
	return perm[j]

#finds the accuracy of the grader using the probe assignments
def accuracy(finalranking, probe_index):
	probe = []
	original = []
	for i in finalranking:
		if i in probe_index:
			probe.append(i)
			original.append(i)
	original.sort()
	#print probe
	dist = kendral_dist(original,probe)
	x = len(probe)
	return 1 - (dist * 2.0)/float(x *(x-1))

#given the sample ranking computes the final aggregated overall ranking
def aggregate(Assignments,Sample_ranking,Accuracy):
	C = []
	final_ranking = []
	for  k in range(0,Assignments):
		C.append(k)
	for i in range(0,Assignments):
		x = []
		for d in C:
			x.append(0)
			count =0
			for g in range(0,Assignments):
				if d in Sample_ranking[g]:
					count = count + Accuracy[g]
					flag=0
					ind = (Sample_ranking[g]).index(d)
					for l in range(0,len(Sample_ranking[g])):
						if(Sample_ranking[g][l] in C and l>ind):
							x[-1] = x[-1] - Accuracy[g]
							flag=1
						elif(Sample_ranking[g][l] in C and l < ind):
							x[-1] = Accuracy[g] + x[-1]
							flag = 1
					if flag==0:
						count = count - Accuracy[g]
			#print count
			if count !=0:
				x[-1] = x[-1]/float(count)
	
		z = x.index(min(x))
		#	if C[z] in probe_index:
		#		z = C.index(probe_index[-1])
		#		del probe_index[-1]
	
		final_ranking.append(C[z])
		#print C[z]
		del C[z]
	return final_ranking

Assignments = 100
Probe_Assignments = 10
Normal_assignments = Assignments - Probe_Assignments
Sample_probe =4
Sample_normal =4

total_index = []
Skills = []
for i in range(0,Assignments):
	total_index.append(i)

for i in range(0,(Assignments/2)):
	Skills.append(3)
for i in range(0,(Assignments/2)):
	Skills.append(4)

#filters the index for probe assignments
probe_index = random.sample(range(0,Assignments),Probe_Assignments)
probe_index.sort()

#filters the index for normal assignments
normal_index = list(filter(lambda x: x not in probe_index, total_index))
normal_index.sort()

Sample_ranking = []
Accuracy = []

for k in range(0,Assignments):
	a = random.sample(range(0,Normal_assignments),Sample_normal)
	a = [normal_index[i] for i in a] #normal assignments given to kth grader
	b = random.sample(range(0,Probe_Assignments),Sample_probe)
	b = [probe_index[i] for i in b] #probe assignment given to kth grader
	initial_ranking = a
	initial_ranking.extend(b) #assignments given to kth grader
	initial_ranking.sort()
	Sample_ranking.append(grading(initial_ranking,Skills[k])) #ranking given by him
	Accuracy.append(accuracy(Sample_ranking[-1],probe_index)) #accuracy

Accuracy2 = np.random.gamma(10,0.1,Assignments) #accuracy taken by paper
#print Accuracy
#print Accuracy2

r1 = aggregate(Assignments,Sample_ranking,Accuracy) #resultant final ranking by our model
r2 = aggregate(Assignments,Sample_ranking,Accuracy2) #resultant final ranking by their model
#print r1
#print r2
#print total_index
e1 = kendral_dist(total_index,r1)
e2 = kendral_dist(total_index,r2)

print "error in our model %d" %e1
print "error in their model %d" %  e2
print ((e2-e1)/float(e1))*100
