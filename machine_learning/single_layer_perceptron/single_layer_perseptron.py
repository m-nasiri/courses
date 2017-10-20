#*****************************************************************************/
# @file    single_layer_perseptron.c 
# @author  Majid Nasiri 95340651
# @version V1.0.0
# @date    18 April 2017
# @brief   Implementation Gates
#*****************************************************************************/

import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import cm

for i in range(50): print(' ')

x=np.array([[0,0],
            [0,1],
            [1,0],
            [1,1]]) 

#AND
#d=np.array([0,0,0,1])
#OR
#d=np.array([0,1,1,1])
#NAND
#d=np.array([1,1,1,0])
#NOR
#d=np.array([1,0,0,0])
#XOR
d=np.array([0,1,1,0])
#XNOR
#d=np.array([1,0,0,1])


z=np.ones((4,1), dtype=np.int)
W=np.array([[1.9],[-0.5],[0.5]])
e=np.array([1,1,1,0], dtype=np.int)
X=np.concatenate((x,z), axis=1)
WdW=np.zeros((3,1))
mu=0.2
cnt=0
sample=0
x1=np.linspace(-1,2,11)
x1=np.matrix(x1)
x2a=np.zeros(x1.shape)
x2a=np.matrix(x2a)
x2=np.empty((0,x1.shape[1]))
e=np.empty((0,1))
error_array=np.empty((0,1))
while (cnt<100):
    v=np.dot(X[sample,:],W[:,cnt])

    if (v>0):
        sigma=1
    else:
        sigma=0
    
    err=d[sample]-sigma
    e=np.append(e, err)
    dW=mu*e[cnt]*X[sample,:]
    WdW[:,0]=W[:,cnt]+dW
    W=np.append(W, WdW, axis=1)
    
    x2a=(-W[2,cnt]/W[1,cnt])-(W[0,cnt]/W[1,cnt])*x1
    x2=np.append(x2, x2a, axis=0)

    sample=sample+1
    if (sample==4):
        sample=0
    
    if (cnt>3):    
        error=np.sum(np.abs(e[-4:]))
        error_array=np.append(error_array, error)
        if (error==0):
            break
      
    cnt=cnt+1

    
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

for i in range(1,cnt):
    ax1.plot(x1.T, x2[i].T, linewidth=2 , linestyle='--',label=i)

colormap = plt.cm.Blues
colors = [colormap(i) for i in np.linspace(0, 1,len(ax1.lines))]

for i,j in enumerate(ax1.lines):
    j.set_color(colors[i])
    
#ax1.legend(loc=2)
colormap=['r','g']
for i in range(4):
    plt.scatter(x[i,0],x[i,1], s=90, c=colormap[d[i]])

if (cnt!=100):
    ax1.plot(x1.T, x2[cnt-1].T, linewidth=3 , c='r')

plt.grid()
plt.axis([-1, 2, -5, 5])
plt.figure()
plt.plot(error_array)
plt.axis([0, cnt, -1, 5])
plt.grid()
print(cnt)
#print(e)    
#print(W) 



