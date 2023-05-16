#!/usr/bin/env python 

import obspy
import numpy as np
import cartopy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import linecache
import pandas as pd
import os
from numpy.polynomial import Polynomial

"""
This code creates a temp0.txt file which stores the output from TauP for one specific degree.
This is acheived through using the temp2.nd file to vary the values of Pdiff and Sdiff in the PREM model using 
a loop so only one temporary file is required. These values from temp2.nd are run through TauP to produce
the output stored in temp0.txt

Variables:
c_list = list to define range of percentage differences to apply to Pdiff and Sdiff velocities
value1 = Pdiff velocity at 2871 km from PREM
value2 = Sdiff velocity at 2871 km from PREM
value3 = Sdiff velocity at 2891 km from PREM
value4 = Pdiff velocity at 2871 km from PREM
z = variable to run data from temp2.nd through TauP

Chloe Steele, August 2022
All corrections made by Jamie Ward
"""


#List of % changes for the velocity values
c_list= np.arange(0.85, 1.16, 0.01)

#Open a temporary txt file to store the output of TauP to
open('temp0.txt', 'w')
for i in c_list:
    #open the prem model
    fin = open('/nfs/a301/ee19ces/summer_project/TauP-2.6.0/StdModels/prem.nd', 'r')

    #Calculating the Pdiff and Sdiff perturbations to substitute into the prem model
    value1 = i*7.26486
    #print('value1:', value1)  
    value2 = i*13.71168
    #print('value2:', value2)
    value3 = i*13.71660
    #print('value3', value3)
    value4 = i*7.26466
    #print('value4', value4)
    
    #Open a temporary file where the Pdiff + Sdiff perturbations can be written into
    with open('temp2.nd', 'w') as fout:
        for j,line in enumerate(fin):
            #Replacing the values at a depth of 2871 km
            if j == 49:
                newline = line.split()
                newline[1] = str(value2)
                newline[2] = str(value1)
                fout.write(' '.join(newline))
                fout.write('\n')
                #print(str(newline))

            #Replacing the values at the CMB
            elif j == 50:
                newline1 = line.split()
                newline1[1] = str(value3)
                newline1[2] = str(value4)
                fout.write(' '.join(newline1))
                fout.write('\n')
                #print(str(newline1))
            #Data from other lines is printed to file without being changed
            else:
                fout.write(line)
              
    with open('temp0.txt', 'a') as f:
        #Write title to file
        f.write("Sdiff         Pdiff" "\n")
        #Writing the new Pdiff/Sdiff preturbations to the file
        per = [str(i)]  
        f.write(' '.join(per))
        f.write('\n')
        #Run the new data through TauP
        z = os.popen("taup_time -h 100 -ph Sdiff,Pdiff -mod /nfs/a301/ee19ces/summer_project/TauP-2.6.0/StdModels/percent_change/premtemps.nd -deg 90 --rayp").read()
        #Write to file in string format
        f.write("%s"  % (z))
        print(str(i))

"""
This code creates a temp1.txt file which stores the output from TauP for one specific degree.
This is acheived through using the temp2.nd file to vary the values of Pdiff and Sdiff in the PREM model using 
a loop so only one temporary file is required. These values from temp2.nd are run through TauP to produce
the output stored in temp1.txt

Variables:
c_list = list to define range of percentage differences to apply to Pdiff and Sdiff velocities
value1 = Pdiff velocity at 2871 km from PREM
value2 = Sdiff velocity at 2871 km from PREM
value3 = Sdiff velocity at 2891 km from PREM
value4 = Pdiff velocity at 2871 km from PREM
z = variable to run data from temp2.nd through TauP

This file temp1.txt has been combined with temp0.txt in bash to produce finaloutcome.txt

Chloe Steele
"""

#List of % changes for the velocity values
c_list= np.arange(0.85, 1.16, 0.01)

#Open a temporary txt file to store the output of TauP to
open('temp1.txt', 'w')
for i in c_list:
    #open the prem model
    fin = open('/nfs/a301/ee19ces/summer_project/TauP-2.6.0/StdModels/prem.nd', 'r')

    #Calculating the Pdiff and Sdiff perturbations to substitute into the prem model
    value1 = i*7.26486
    #print('value1:', value1)  
    value2 = i*13.71168
    #print('value2:', value2)
    value3 = i*13.71660
    #print('value3', value3)
    value4 = i*7.26466
    #print('value4', value4)
    
    #Open a temporary file where the Pdiff + Sdiff perturbations can be written into
    with open('temp2.nd', 'w') as fout:
        for j,line in enumerate(fin):
            #Replacing the values at a depth of 2871 km
            if j == 49:
                newline = line.split()
                newline[1] = str(value2)
                newline[2] = str(value1)
                fout.write(' '.join(newline))
                fout.write('\n')
                #print(str(newline))

            #Replacing the values at the CMB
            elif j == 50:
                newline1 = line.split()
                newline1[1] = str(value3)
                newline1[2] = str(value4)
                fout.write(' '.join(newline1))
                fout.write('\n')
                #print(str(newline1))
            #Data from other lines is printed to file without being changed
            else:
                fout.write(line)
              
    with open('temp1.txt', 'a') as f:
        #Write title to file
        f.write("Sdiff         Pdiff" "\n")
        #Writing the new Pdiff/Sdiff preturbations to the file
        per = [str(i)]  
        f.write(' '.join(per))
        f.write('\n')
        #Run the new data through TauP
        z = os.popen("taup_time -h 100 -ph Sdiff,Pdiff -mod /nfs/a301/ee19ces/summer_project/TauP-2.6.0/StdModels/percent_change/premtemps.nd -deg 150 --rayp").read()
        #Write to file in string format
        f.write("%s"  % (z))
        print(str(i))

"""
This code produces a graph of velocity perturbations of Pdiff and Sdiff waves in the mantle and their 
impact on the ray parameter.

This code uses data from finaloutcome.txt, the combined document of temp0.txt and temp1.txt.
This was necessary due to the impact of the degree value, that was put into TauP, on the output for 
certain percentage perturbations.

This code also ignores any perturbations with a decrease of less than 4% due to a bug in TauP resulting
in there being no variations in the values writen to finaloutcome.txt

Variables:
data = Pdiff and Sdiff values from finaloutcome.txt
percent = percentage changes from finaloutcome.txt
Sdiffs = list to put the Sdiff data into
Pdiffs = list to put the Pdiff data into
labels = list to put the percent data into
Sdiff = allows data to be split into Sdiff and Pdiff accordingly
Pdiff = allows data to be split into Sdiff and Pdiff accordingly
Vp/Vs = P-diff to S-diff wave velocity ratio
S_slope = gradient of Sdiff line on graph
S_intercept = intercept of Sdiff line on graph
P_slope = gradient of Pdiff line on graph
P_intercept = intercept of Sdiff line on graph

Chloe Steele

"""
#Open document that has the data that needs to be plotter
with open('finaloutcome.txt', 'r') as f:
    #Load in the data
    np.loadtxt('finaloutcome.txt', dtype=str, delimiter=',')
    #Read the data
    lines = f.readlines()
    #Read the relevant lines
    data = lines[46::4]
    percent = lines[45::4]
    #Create empty lists
    Sdiffs = []
    Pdiffs = []
    labels = []
    #Loop to split the Pdiff and Sdiff data and append to Pdiffs/Sdiffs
    for i in range(0, len(data)):
        a = data[i].split()
        Sdiff = a[0]
        Pdiff = a[1]
       
        Sdiffs.append(Sdiff)
        Pdiffs.append(Pdiff)

#Loop for % change to create a list with all values to 2dp only
for p in percent:
    p1 = float(p)
    labels.append(f"{p1:.2f}")


#Vp/Vs ratio from Pdiffs and Sdiffs data
Vp_Vs = np.divide(np.array(Pdiffs).astype(float),np.array(Sdiffs).astype(float))

#Changing strings to arrays in order to plot these variables on a graph
Sdiffs = np.array(Sdiffs).astype(float)
labels = np.array(labels).astype(float)
Pdiffs = np.array(Pdiffs).astype(float)


#Checks that the correct data is being printed and as the correct type
print((Sdiffs))
print(labels)
print(type(Sdiffs))
print(type(labels))
print(type(Pdiffs))

#Create figure
fig = plt.figure(figsize=(8,6), tight_layout=True)

#Finding the gradient of the Sdiff line
S_slope, S_intercept = np.polyfit(labels, Sdiffs, 1)
#Print the equation of the line
print('Gradient of Sdiff line: y =', S_slope, 'x +', S_intercept)
#plt.text(slope, intercept, 'hi')

#Finding the gradient of the Pdiff line
P_slope, P_intercept = np.polyfit(labels, Pdiffs, 1)
#Print the equation of the line
print('Gradient of Pdiff line: y =', P_slope, 'x +', P_intercept)


#Plot Sdiff data
plt.plot(labels, Sdiffs)
plt.scatter(labels, Sdiffs, label = 'Sdiff')
#Plot Pdiff data
plt.plot(labels, Pdiffs)
plt.scatter(labels, Pdiffs, label = 'Pdiff')
#Plot Vp/Vs data
plt.plot(labels, Vp_Vs)
plt.scatter(labels, Vp_Vs, label = 'Vp/Vs')
#Create x label
plt.xlabel('Velocity % change')
#Set x limit
plt.xlim(0.95, 1.15)
#Set y limit
plt.ylim(0, 12)
#Create y lavel
plt.ylabel('Ray parameter')
#Add gradient of Sdiff line to the graph
plt.plot(S_slope, S_intercept, label='Sdiff: y={:.2f}x+{:.2f}'.format(S_slope, S_intercept))
#Add gradient of Pdiff line to the graph
plt.plot(P_slope, P_intercept, label='Pdiff: y={:.2f}x+{:.2f}'.format(P_slope, P_intercept))
#Create title
plt.title('Effect of velocity perturbations on the ray parameter')
#Create legend
plt.legend()
#Save figure
plt.savefig('Velocity_perturbations.pdf')
plt.savefig('Velocity_perturbations.png')
#Show figure
plt.show()
