import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filename = 'LayerAll.txt'

#df = pd.read_csv(filename, sep=',')
#print(df)

data = np.loadtxt(filename,delimiter=',')

# 0: Hidden dimension
# 1: Layers
# 2: Iterations
# 3: Train error
# 4: Test error

#data = data[np.logical_and(
#		data[:,1]==1, 
#		data[:,2]==1000
#	)]

dataA = data[np.logical_and(
	data[:,1]==1,
	data[:,3]<10
)]
dataB = data[np.logical_and(
	data[:,1]==2,
	data[:,3]<10
)]
dataC = data[np.logical_and(
	data[:,1]==3,
	data[:,3]<10
)]

a, = plt.plot(dataA[:,0],dataA[:,3])
b, = plt.plot(dataB[:,0],dataB[:,3])
c, = plt.plot(dataC[:,0],dataC[:,3])
#b, = plt.plot(data[:,0],data[:,4])

#plt.legend([a, b], ['Train error', 'Test error'])
plt.legend([a,b,c],['Layer 1', 'Layer 2', 'Layer 3'])
plt.yticks(np.arange(0,10,1))
plt.title("Iterations = 10000\nAll layers")
plt.xlabel("Hidden Dimension Size")
plt.ylabel("Error")

plt.show()
