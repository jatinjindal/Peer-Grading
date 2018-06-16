import random
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle

Probe_Assignments = 50
Assignments = 500
Normal_Assignments = Assignments - Probe_Assignments

Sample_probe =5
Sample_normal =5
Sample = Sample_probe + Sample_normal
	


Y1 = [] # mean of the errors for QUEST Algorithm
std1 = [] #standard deviation for QUEST
Y2 = [] # mean of the errors for Gibbs Algorithm 
std2 = [] # standard deviation for GIBBS
Y3 = [] # mean of the error using AVERAGE
std3 = [] # standard deviation using Average

alpha = [1,2,3,4,5]

for Alpha in alpha:
	#true data
	prior_mean = 1.0/2
	Gamma = 36
	prior_var = 1.0/Gamma
	true_score = np.random.normal(prior_mean, 1.0/Gamma, Assignments)

	#prior known bias and variance
	neta = 64
	bias = np.random.normal( 0, 1.0/neta , Assignments)
	shuffle(bias)

	Beta = 30
	reliability = np.random.gamma (Alpha , Beta, Assignments)
	shuffle(reliability)

	Error_quest = [] #contains the observation for the following iterations resp
	Error_gibs = []
	Error_spot = []
	for iteration in range(0,10): #for each case 10 observations are taken
		#-------------------------------------------------------------------
		# METHOD-1 (OUR ALGORITHM - SIMILAR TO QUEST)
		#REPORTED SCORE
		#submisssions - g,g+1,...g+Normal_Assignment-1 , Reported_score - marks given by grader to those assignments
		Reported_score =[]
		for k in range(0,Assignments):
			g = []
			for i in range(0,Sample_normal):
				j = (k+i)%Normal_Assignments
				x = np.random.normal( true_score[j] + bias[k], 1.0/reliability[k] , 1)
				g.extend(x)
			
				
			for i in range(0,Sample_probe):
				j = (k + i)%Probe_Assignments + Normal_Assignments
				x = np.random.normal( true_score[j] + bias[k], 1.0/reliability[k] , 1)
				g.extend(x)
			
			Reported_score.append(g)

		#--------------------------------------------------------------------
		#CALCULATING the expected bias and the expected reliability
		expected_bias = []
		expected_variance = []

		for grader in range(0,Assignments):
			test_data = Reported_score[grader][Sample_normal:]
			true_data = []
			for i in range(0,Sample_probe):
				j  = (grader + i)%Probe_Assignments + Normal_Assignments
				true_data.append(true_score[j])

			delta = []
			for i in range(0,Sample_probe):
				delta.append(test_data[i] - true_data[i])

			x = np.var(delta)
			expected_variance.append(x)
			x= np.mean(delta)
			expected_bias.append(x)

		#----------------------------------------------------------------
		#-----------------------Calculate Score-------------------------------
		Calculate_score = []
		for paper in range(0,Normal_Assignments):
			j = Sample_normal - 1
			temp = 0
			denom =0
			for i in range(0,Sample_normal):
				grader = paper-j
				if grader<0:
					grader = grader + Normal_Assignments
				s = Reported_score[grader][j]
				temp = temp + (s-expected_bias[grader])/expected_variance[grader]	
				denom = denom + 1.0/expected_variance[grader]
				j = j-1
			temp = temp + prior_mean/prior_var
			denom = denom + 1.0/prior_var
			temp = temp/denom
			Calculate_score.append(temp)


		error1 = 0
		for i in range(0,Normal_Assignments):
			error1 = error1 + (Calculate_score[i]- true_score[i])*(Calculate_score[i] - true_score[i])

		error1 = error1 / Normal_Assignments
		error1 = error1 **(0.5)
		
		Error_quest.append(error1)









		#-------------*******************-------------------------------
		#METHOD-2 (Gibbs Sampling)
		#Reported_score

		Reported_score =[]

		for k in range(0,Assignments):
			g = []
			for i in range(0,Sample):
				j = (k+i)%Assignments
				x = np.random.normal( true_score[j] + bias[k], 1.0/reliability[k] , 1)
				g.extend(x)
			
			Reported_score.append(g)

		#------------------------------------------------------

		#initialising unobserved variables
		exp_bias = []
		exp_reliabilty = []
		exp_score = []
		for i in range(0,Assignments):
			exp_score.append(0.5)
			exp_bias.append(0.05)
			exp_reliabilty.append(100)

		#-----------------------------------------------------

		for t in range(0,1000):
			
			for paper in range(0,Assignments):
				num = Gamma*prior_mean
				denom = Gamma
				for k in range(0,Sample):
					grader = (paper-k+Assignments)%Assignments
					num = num + exp_reliabilty[grader]*(exp_bias[grader] + Reported_score[grader][k])
					denom = denom + exp_reliabilty[grader]

				exp_score[paper] = random.normalvariate(num/denom, 1.0/denom)


			for grader in range(0,Assignments): #v = grader
				x = Alpha + Sample/2.0
				y = Beta
				for k in range(0,Sample):
					p = (grader+k)%Assignments # paper(p) checked by grader
					temp = Reported_score[grader][k] - (exp_score[p]+ exp_bias[grader])
					temp = temp * temp
					y =y + temp/2.0
				exp_reliabilty[grader] = random.gammavariate(x,y)


			for grader in range(0,Assignments):
				num =0
				denom = neta + Sample*	exp_reliabilty[grader]
				for k in range(0,Sample):
					p = (grader+k)%Assignments
					num = num+ Reported_score[grader][k] - exp_score[p]

				num = num * exp_reliabilty[grader]

				exp_bias[grader] = random.normalvariate(num/denom, 1.0/denom)



		error_gibs = 0
		for i in range(0,Assignments):
			error_gibs = error_gibs + (exp_score[i]-true_score[i])*(exp_score[i]-true_score[i])

		error_gibs  =error_gibs / Assignments
		error_gibs = error_gibs ** (0.5)
		#print "Error_GIBS"
		#print error_gibs
		Error_gibs.append(error_gibs)

		#print Error_gibs




		#----------------------------------------------------------------------
		# Method-3 Spot checking

		Calculate_score = []
		for paper in range(0,Normal_Assignments):
			j = Sample - 1
			num = 0
			for i in range(0,Sample):
				grader = paper - j
				if grader < 0:
					grader = grader+Normal_Assignments
				s = Reported_score[grader][j]
				num = num + s
				j = j -1
			Calculate_score.append(num/Sample)

		error2 = 0
		for i in range(0,Normal_Assignments):
			error2 = error2 + (true_score[i] - Calculate_score[i])*(true_score[i] - Calculate_score[i])

		error2 = error2/Normal_Assignments
		error2 = error2 ** (0.5)
		#print "Spot Checking"
		#print error2
		Error_spot.append(error2)

	
	x = np.mean(Error_spot)
	Y3.append(x)
	x = np.var(Error_spot)
	std3.append(x)

	x = np.mean(Error_gibs)
	Y2.append(x)
	x = np.var(Error_gibs)
	std2.append(x)
	
	x = np.mean(Error_quest)
	Y1.append(x)
	x = np.var(Error_quest)
	std1.append(x)
	




#plot the graph

std1 = np.sqrt(std1)
std2 = np.sqrt(std2)
std3 = np.sqrt(std3)
fig = plt.figure()
ax = fig.add_subplot(111)


## the data
Reliability = [1, 2, 3, 4, 5]
N = 5


## necessary variables
ind = np.arange(N)                # the x locations for the groups
width = 0.2                     # the width of the bars

## the bars
rects1 = ax.bar(ind, Y1, width,
                color='black',
                yerr=std1,
                error_kw=dict(elinewidth=2,ecolor='red'))

rects2 = ax.bar(ind+width, Y2, width,
                    color='red',
                    yerr=std2,
                    error_kw=dict(elinewidth=2,ecolor='black'))

rects3 = ax.bar(ind+2*width, Y3, width, color = 'blue', yerr = std3, error_kw=dict(elinewidth=2,ecolor='black'))
# axes and labels
ax.set_xlim(-2*width,len(ind)+2*width)
#ax.set_ylim(0,45)
ax.set_ylabel('RMS Error')
ax.set_yscale('log')
ax.set_title('Error vs Reliability')
xTickMarks = [str(i) for i in Reliability]
ax.set_xticks(ind+2*width)
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, rotation=45, fontsize=10)

## add a legend
ax.legend( (rects1[0], rects2[0], rects3[0]), ('Quest', 'Gibs','Spot') )

plt.show()