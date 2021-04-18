

## Part 2
## K-Means

### data load and true data


import numpy as np
#%matplotlib widget
import matplotlib.pyplot as plt
import cv2 as cv
import random
from numpy.lib.function_base import append
from tqdm import tqdm



#ps is random indices
def myKMeans(data,K,r):
    #start with random indexes
    ps = random.sample(range(len(data[:,1])), K)
    points=[]
    #turn indexes into points
    for k in ps:
        points.append(data[k])
    #run algo
    for x in tqdm(range(r),desc="K-Mean",position=0):
        clus =  np.array([])
        #cluster to closest point
        for i in tqdm(data,desc="Calculating points",position=1):
            dis = []
            for j in points:
                dis.append(np.linalg.norm(j-i))
            clus = np.append(clus,dis.index(min(dis)))
        #remean
        points = []
        for y in range(K):
            points.append(data[np.where(clus == y)].mean(axis = 0))
    return clus,points


def findBest(data,truth,K,r):
    error = np.inf
    for i in range(r):
        temp_clus,temp_points = myKMeans(data,K,r)
        iter_error = 0
        for t in truth:
            for p in temp_points:
                iter_error += np.abs(np.linalg.norm(t-p))
                if iter_error < error:
                    error = iter_error
                    points = temp_points.copy()
                    clus = temp_clus.copy()
    return clus,points

K = 4
r= 10
data = np.loadtxt("510_cluster_dataset.txt")
print(data.shape)
plt.figure(figsize=(10,8))
plt.scatter(data[:,0],data[:,1])
truth= np.array([data[:500].mean(axis=0),data[500:1000].mean(axis=0),data[1000:1500].mean(axis=0)])
plt.scatter(truth[:,0],truth[:,1],color = 'black',marker='x')
plt.show()

#clus,points = myKMeans(data,K,r)
clus,points = findBest(data,truth,K,r)
print(type(clus[0]))

plt.figure(figsize=(10,8))
for y in range(K):
    d1 = data[np.where(clus == y)]
    plt.scatter(d1[:,0],d1[:,1],marker='o')

for i in points:
    plt.scatter(i[0],i[1],marker='x')
plt.show()


img = cv.imread('Kmean_img1.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img, (0,0), fx=0.5, fy=0.5)
xx,yy,zz = img.shape
plt.figure(figsize=(10,8))
plt.title("Original: beach 1")
plt.imshow(img)
plt.show()


print(img.shape)
print(img.reshape(-1,3).shape)
img_dat = img.reshape(-1,3)
'''
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(img_dat[:,0], img_dat[:,1], img_dat[:,2], marker='o')
plt.show()
'''
K = 10
r = 10
clus,points = myKMeans(img_dat,K,r)
print(clus)


'''
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for y in range(K):
    d1 = img_dat[np.where(clus == y)]
    ax.scatter(d1[:,0],d1[:,1],d1[:,2],marker='o')

for i in points:
    ax.scatter(i[0],i[1],i[2],marker='x')
plt.show()
'''

new_pic = np.array([0,0,0])
print(clus,points)
for i in tqdm(clus,desc="re-color: "):
    new_pic = np.vstack((new_pic,points[int(i)]))
print(new_pic)
new_pic = new_pic[1:]
new_pic = np.reshape(new_pic,img.shape).astype(np.uint8)
plt.imshow(new_pic)
plt.show()