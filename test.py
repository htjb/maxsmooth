import numpy as np
import pylab as pl
from maxsmooth.settings import setting
from maxsmooth.msf import msf_fit

setting=setting()

#EDGES Data
freq,weight,Tsky,Tres1,Tres2,Tmodel,T21=np.loadtxt('EDGES_Data/figure1_plotdata.csv',delimiter=',',unpack=True,usecols=[0,1,2,3,4,5,6],skiprows=1)
x=[]
y=[]
for i in range(len(Tsky)):
    if Tsky[i]!=0:
        y.append(Tsky[i])
        x.append(freq[i])
x=np.array(x)
y=np.array(y)

N=[3,4]#,5,6,7,8,9,10]

result=msf_fit(x,y,N,setting)
print('Objective Funtion Evaluations:\n', result.objective)
#print('Parameters:\n',result.parameters[2])
print('RMS:\n',result.rms)
#print('Fitted Temps:\n',result.y_fit)
#print('Sign Combinations:\n',result.signs_results)
#print('Derivatives:\n',result.derivatives)

pl.subplot(111)
pl.plot(N,result.rms)
pl.xlabel('N')
pl.ylabel('RMS [mk]')
pl.show()
