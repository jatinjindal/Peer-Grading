import random
import matplotlib.pyplot as plt
import numpy as np
from numpy import median
from random import shuffle


def Reported_Score_in_QUEST():
	Reported_score =[]
	for grader in range(0,Assignments):
		g = []
		for i in range(0,Sample_normal):
			paper = (grader+i)%Normal_Assignments
			x = np.random.normal( true_score[paper] + bias[grader], 1.0/reliability[grader] , 1)
			g.extend(x)
			
				
		for i in range(0,Sample_probe):
			paper = (grader + i)%Probe_Assignments + Normal_Assignments
			x = np.random.normal( true_score[paper] + bias[grader], 1.0/reliability[grader] , 1)
			g.extend(x)
			
		Reported_score.append(g)

	return Reported_score
def Expected_Quality_in_Quest():
	expected_bias = []
	expected_variance = []

	for grader in range(0,Assignments):
		test_data = Reported_score[grader][Sample_normal:]
		true_data = []
		for i in range(0,Sample_probe):
			paper  = (grader + i)%Probe_Assignments + Normal_Assignments
			true_data.append(true_score[paper])

		delta = []
		for i in range(0,Sample_probe):
			delta.append(test_data[i] - true_data[i])

		x = np.var(delta)
		expected_variance.append(x)
		x= np.mean(delta)
		expected_bias.append(x)
	return (expected_bias,expected_variance)
def Calculate_Score_in_Quest():
	Calculate_score = []
	for paper in range(0,Normal_Assignments):
		num = 0
		denom =0
		for i in range(0,Sample_normal):
			grader = (paper-i + Normal_Assignments) % Normal_Assignments
			
			s = Reported_score[grader][i]
			num = num + (s-expected_bias[grader])/expected_variance[grader]	
			denom = denom + 1.0/expected_variance[grader]
			
		num = num + prior_mean/prior_var
		denom = denom + 1.0/prior_var
		temp = num/denom
		Calculate_score.append(temp)
	
	return Calculate_score
def Error():
	error = 0
	for i in range(0,Normal_Assignments):
			error = error + abs((Calculate_score[i]/true_score[i])- 1)

	error = error / Normal_Assignments
	error = error*100
	#error = error **(0.5)
	return error
def Reported_Score_in_Gibbs():
	Reported_score =[]
	for grader in range(0,Assignments):
		g = []
		for i in range(0,Sample):
			paper = (grader+i)%Assignments
			x = np.random.normal( true_score[paper] + bias[grader], 1.0/reliability[grader] , 1)
			g.extend(x)
		Reported_score.append(g)
	return Reported_score
def Calculated_Score_in_Gibbs():
	Sample_Observations = []
	Remaining = Total_Gibs_Iterations - Initial_Cut_off
	for t in range(0,Total_Gibs_Iterations):
		
		for paper in range(0,Assignments):
			num = Gamma*prior_mean
			denom = Gamma
			for k in range(0,Sample):
				grader = (paper-k+Assignments)%Assignments
				num = num + exp_reliabilty[grader]*(exp_bias[grader] + Reported_score[grader][k])
				denom = denom + exp_reliabilty[grader]

			exp_score[paper] = random.normalvariate(num/denom, 1.0/denom)
			
		if t>=Initial_Cut_off:
			Sample_Observations.append(exp_score)
			

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

	for paper in range(0,Assignments):
		temp = 0
		for l in range(0,Remaining):
			temp = temp + Sample_Observations[l][paper]
		exp_score[paper] = temp/Remaining
	
	return exp_score
def Calculated_Score_using_Mean():
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

	return Calculate_score
def Calculated_Score_Using_Median():
	Calculate_score = []
	for paper in range(0,Assignments):
		Marks = []
		for i in range(0,Sample_normal):
			grader = (paper-i+Normal_Assignments)%Normal_Assignments
			Marks.append(Reported_score[grader][i])
		temp = median(Marks)
		Calculate_score.append(temp)

	return Calculate_score
def plot_graph():
	fig = plt.figure()
	ax = fig.add_subplot(111)


	## the data
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

	rects3 = ax.bar(ind+2*width, Y3, width,
						color = 'blue',
						yerr = std3, 
						error_kw=dict(elinewidth=2,ecolor='black'))

	rects4 = ax.bar(ind + 3*width, Y4, width,
						color ='green',
						yerr = std4,
						error_kw =dict(elinewidth=2,ecolor='black'))

	# axes and labels
	ax.set_xlim(-3*width,len(ind)+3*width)
	ax.set_ylim(0,45)
	ax.set_ylabel('RMS Error')
	#ax.set_yscale('log')
	ax.set_title('Beta='+str(Beta)+' Gamma ='+str(Gamma)+' neta ='+str(neta))
	xTickMarks = [str(i) for i in alpha]
	ax.set_xticks(ind+2*width)
	xtickNames = ax.set_xticklabels(xTickMarks)
	plt.setp(xtickNames, rotation=45, fontsize=10)

	## add a legend
	ax.legend( (rects1[0], rects2[0], rects3[0],rects4[0]), ('Quest', 'Gibs','Mean','Median') )

	plt.show()


Probe_Assignments = 50
Assignments = 100
Normal_Assignments = Assignments - Probe_Assignments

Sample_probe =2
Sample_normal =2
Sample = Sample_probe + Sample_normal
Total_Gibs_Iterations = 800
Initial_Cut_off = 80


prior_mean = 1.0/2
Gamma = 36
prior_var = 1.0/Gamma
neta = 64
Alpha = 3
Beta = 30

alpha = [1,2,3,4,5]
#(Beta,Gamma,neta)
parameters = [(10,16,36),(10,16,64),(10,36,36),(10,36,64),(30,16,34),(30,16,64),(30,36,36),(30,36,64)]

Total_Iterations = 100


for (Beta,Gamma,neta) in parameters:
	Y1 = [] # mean of the errors for QUEST Algorithm
	std1 = [] #standard deviation for QUEST
	Y2 = [] # mean of the errors for Gibbs Algorithm 
	std2 = [] # standard deviation for GIBBS
	Y3 = [] # mean of the error using AVERAGE
	std3 = [] # standard deviation using Average
	Y4 = []
	std4 = []
	for Alpha in alpha:

		true_score = np.random.normal(prior_mean, 1.0/Gamma, Assignments)
		bias = np.random.normal( 0, 1.0/neta , Assignments)
		reliability = np.random.gamma (Alpha , Beta, Assignments)
		
		Error_in_Quest = []
		Error_in_Gibbs = []
		Error_in_Mean = []
		Error_in_Median = []
		
		for iteration in range(0,Total_Iterations):
			#-------------------------------------------------------------------
			# METHOD-1 (OUR ALGORITHM - SIMILAR TO QUEST)
			#REPORTED SCORE
			#submisssions - g,g+1,...g+Normal_Assignment-1 , Reported_score - marks given by grader to those assignments
			Reported_score = Reported_Score_in_QUEST()

			temp = Expected_Quality_in_Quest()
			expected_bias = temp[0]
			expected_variance = temp[1]

			Calculate_score = Calculate_Score_in_Quest()

			error = Error()
			Error_in_Quest.append(error)

			#-------------*******************-------------------------------
			#METHOD-2 (Gibbs Sampling)
			#Reported_score

			Reported_score = Reported_Score_in_Gibbs()

			#initialising unobserved variables
			exp_bias = []
			exp_reliabilty = []
			exp_score = []
			for i in range(0,Assignments):
				exp_score.append(0.5)
				exp_bias.append(0.05)
				exp_reliabilty.append(100)

			Calculate_score = Calculated_Score_in_Gibbs()

			error = Error()
			Error_in_Gibbs.append(error)
			#--------------------------------------------------------------
			#METHOD-3 (Mean)

			Calculate_score = Calculated_Score_using_Mean()
			
			error = Error()
			Error_in_Mean.append(error)

			#-----------------------------------------------
			#METHOD_4 (Median)
			Calculate_score = Calculated_Score_Using_Median()
			error = Error()
			Error_in_Median.append(error)

		x= np.mean(Error_in_Quest)
		Y1.append(x)
		x = np.var(Error_in_Quest)
		x = x**(0.5)
		std1.append(x)
		x= np.mean(Error_in_Gibbs)
		Y2.append(x)
		x = np.var(Error_in_Gibbs)
		x = x**(0.5)
		std2.append(x)
		x= np.mean(Error_in_Mean)
		Y3.append(x)
		x = np.var(Error_in_Mean)
		x = x**(0.5)
		std3.append(x)
		x= np.mean(Error_in_Median)
		Y4.append(x)
		x = np.var(Error_in_Median)
		x = x**(0.5)
		std4.append(x)
	plot_graph()



#-----------------------------------------------------------------------


