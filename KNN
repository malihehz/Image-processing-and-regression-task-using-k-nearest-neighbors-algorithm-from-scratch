import mnist
import numpy as np
import time
import math
import operator
import matplotlib.pyplot as plt
class knn(object):
    # Constructor
    def __init__(self,k=9,weighted=True,classify=True):  
        self.k = k
        self.weighted = weighted
        self.classify = classify
       
    def fit(self,x,y):  
        self.x_train = x.astype(np.float32)
        self.y_train = y
       
    def pca(self,x_train,x_test,n):    
        m = np.mean(x_train)
        CM = np.cov(x_train,rowvar=False)
        eig_vals, eig_vecs = np.linalg.eig(CM)
        s = np.argsort(-eig_vals)
        eig_vecs=eig_vecs.T
        eig_vals = eig_vals[s]
        eig_vecs = eig_vecs[:n].real
        cs = (np.cumsum(eig_vals)/np.sum(eig_vals)).real
        PC_train = np.dot(x_train-m,eig_vecs.T)    
        PC_test = np.dot(x_test-m,eig_vecs.T)
        print("Informatin preserved: ",cs[n-1])
        print(eig_vecs.shape)
        plt.plot(np.cumsum(eig_vals)/np.sum(eig_vals))
        plt.ylabel('cummulative infor mation')
        plt.xlabel('number of components')
        plt.show()
        return  PC_train,  PC_test
   
    #def euclidDis(self,x_test,x_train):
     #       return np.sum ((x_test - x_train)**2)

    def predict(self,x_train,x_test,y_train):    
        pred =[]
        eps = 0.001
        for i in range(len(x_test)):
            neighbourDis = []
            for j in range(len(x_train)):
                respectiveDis = np.sum((x_test[i] - x_train[j])**2)#uclidian dis
                #self.euclidDis(x_test[i],x_train[j])
                if respectiveDis == 0:
                    respectiveDis = eps
                weight = 1/respectiveDis
                
######neighbourDis.append((int(y_train[j]), respectiveDis, weight))####classify(mnist)
                neighbourDis.append(((y_train[j]), respectiveDis, weight))###regression
            #sort based on distance
            neighbourDis.sort(key=operator.itemgetter(1))
            neighbourDis = np.array(neighbourDis,dtype=float)          
            #selecting K-nearest Neighbour
            kNeighbour = neighbourDis[0:self.k]
            predi = 0.
            # Classification: Wighted vs. Unweighted
            if self.classify:  
                n_classes = np.max(y_train) + 1    
                count = np.zeros(n_classes)
                if self.weighted:
                    idx=0
                    for y in kNeighbour[:,0]:
                        count[int(y)] += kNeighbour[idx,2]
                        idx +=1
                    predi = np.argmax(count)
                    pred.append(predi)    
                else:
                    for y in kNeighbour[:,0]:
                        count[int(y)] +=1
                    predi = np.argmax(count)
                    pred.append(predi)             
            # Regression: Wighted vs. Unweighted
            else:
                numinator=0.
                weightSum=0.
                predi=0.
                if self.weighted:
                    weightSum = sum(kNeighbour[:,2])
                    numinator = sum(kNeighbour[:,0] * kNeighbour[:,2])
                    predi = numinator/weightSum
                    pred.append(predi) 
                else:
                    predi = np.mean(kNeighbour[:,0])
                    pred.append(predi)
        pred = np.array(pred)
        if (self.classify):
            pred = pred.astype(int)
        return pred
    
if __name__ == "__main__":
    
    print('MNIST dataset')
    x_train, y_train, x_test, y_test = mnist.load()
    x_train = np.array(x_train[::1000], dtype=np.int)
    y_train = np.array(y_train[::1000], dtype=np.int)
    x_test = np.array(x_test[::1000], dtype=np.int)
    y_test = np.array(y_test[::1000], dtype=np.int)
    model = knn()

    n = 64
    #x_train_pc, x_test_pc, = model.pca(x_train, x_test, n)
   
    start = time.time()
    model.fit(x_train, y_train)
    elapsed_time = time.time()-start
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    start = time.time()    
    pred = model.predict(x_train,x_test,y_train)
    #pred = model.predict(x_train_pc,x_test_pc,y_train)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))  
    print('Accuracy:',np.sum(pred==y_test)/len(y_test))
'''  
    print('\nSolar particle dataset')
    x_train = np.load('x_ray_data_train.npy')
    y_train = np.load('x_ray_target_train.npy')
    x_test = np.load('x_ray_data_test.npy')
    y_test = np.load('x_ray_target_test.npy')
    x_train = np.array(x_train[::1800])
    y_train = np.array(y_train[::1800])
    x_test = np.array(x_test[::1800])
    y_test = np.array(y_test[::1800])
    
    model = knn(classify=False)
    
    n = 6
    #x_train_pc, x_test_pc, = model.pca(x_train, x_test, n) 
    
    start = time.time()
    model.fit(x_train, y_train)
    elapsed_time = time.time()-start
    print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  
    start = time.time()      
    pred = model.predict(x_train,x_test,y_train)
    #pred = model.predict(x_train_pc,x_test_pc,y_train)
    elapsed_time = time.time()-start
    print('Elapsed_time testing  {0:.6f} '.format(elapsed_time))  
    print('Mean square error:',np.mean(np.square(pred-y_test)))'''
