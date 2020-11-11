import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filename = 'FirstDataSettings.csv'

#df = pd.read_csv(filename, sep=',')
#print(df)

data = np.loadtxt(filename,delimiter=',')

# 0: Hidden dimension
# 1: Layers
# 2: Iterations
# 3: Train error
# 4: Test error

data = data[np.logical_and(
		data[:,1]==1, 
		data[:,2]==1000
	)]

a, = plt.plot(data[:,0],data[:,3])
b, = plt.plot(data[:,0],data[:,4])

plt.legend([a, b], ['Train error', 'Test error'])

plt.title("Iterations = 1000\nLayers = 1")
plt.xlabel("Hidden Dimension Size")
plt.ylabel("Error")

plt.show()
