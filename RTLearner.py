"""
Name: AHMAD KHAN
GT ID: akhan361
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import time
from scipy import stats

class RTLearner(object):

    def __init__(self, leaf_size = 1, verbose = False):
        pass # move along, these aren't the drones you're looking for
        self.leaf_size = leaf_size

    def author(self):
        return 'akhan361' # replace tb34 with your Georgia Tech username

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        newdataX = np.column_stack((dataX, dataY))

        # build and save the model
        #self.model_coefs, residuals, rank, s = np.linalg.lstsq(newdataX, dataY)
        #print(newdataX)
        self.model = self.build_tree(newdataX, self.leaf_size)
        #print(self.model)
        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        node = 'not leaf'
        dTree = self.model
        y_all = np.empty(shape=(points.shape[0]))

        for i in range(0,points.shape[0]):
            currentRow = 0
            while dTree[currentRow,0] != 'leaf':

                #print(dTree[currentRow,1])
                #print((float(dTree[currentRow,0])))
                #print(points[i, dTree[currentRow,0]])

                if points[i, int(float(dTree[currentRow,0]))] <= float(dTree[currentRow,1]):
                    currentRow = currentRow + int(float(dTree[currentRow,2]))
                    #print currentRow

                elif points[i, int(float(dTree[currentRow,0]))] > float(dTree[currentRow,1]):
                    currentRow = currentRow + int(float(dTree[currentRow,3]))
                    #print currentRow

            y_predict = dTree[currentRow,1]
            y_all[i] = y_predict

        return np.nan_to_num(y_all)    

        
        #return (self.model_coefs[:-1] * points).sum(axis = 1) + self.model_coefs[-1]
    def build_tree(self, data, leaf_size):
        #print(data)
        #print(data.shape)
        if  data.shape[0]   ==  1:  
            return  np.array([['leaf',  stats.mode(data[:,data.shape[1]-1])[0][0], 'NA', 'NA']])

        if  data.shape[0]   <=  leaf_size:  
            return  np.array([['leaf',  stats.mode(data[:,data.shape[1]-1])[0][0], 'NA', 'NA']])
        #if  all data.y same:  
            #return  [leaf,  data.y, NA, NA]
        else:
            #determine best feature i to  split 
            #i = self.computeCorrelations(data)
            i = np.random.randint(0, data.shape[1]-1)
            #print i
            #randomRow1 = 0
            #randomRow2 = 0
            #while randomRow1 != randomRow2:
            randomRow1 = np.random.randint(0, data.shape[0])
            randomRow2 = np.random.randint(0, data.shape[0])
            #print randomRow
            #print data.shape
            #SplitVal = np.median(data[:,i])
            SplitVal = (data[randomRow1,i] + data[randomRow2,i])/2
            #print (i)
            #print(SplitVal)

            if data.shape[0] == data[data[:,i]<=SplitVal].shape[0]:
                return np.array([['leaf',  stats.mode(data[:,data.shape[1]-1])[0][0], 'NA', 'NA']])

            else:
                #SplitVal =  data[:,i].median()
                #j = j+1
                #print(j)
                lefttree =  self.build_tree(data[data[:,i]<=SplitVal], leaf_size)
                #print(j)
                righttree = self.build_tree(data[data[:,i]>SplitVal], leaf_size)
                #print(lefttree)
                #print(righttree)
                root    =   np.array([[i, SplitVal, 1, lefttree.shape[0] + 1]])
                #print(root)
            return  np.vstack((root, lefttree, righttree))
            #return  (np.append(root, lefttree, righttree))


if __name__=="__main__":
    '''
    seed=1481090004
    def compute_rmse(actual, predicted):
        mse = ((actual - predicted) ** 2).mean(axis=None)
        return np.sqrt(mse)

    def create_array(data):        
        ind = data.shape[1]-1
        data = data.iloc[:,1:data.shape[1]]
        dataX = data.iloc[:,0:data.shape[1]-1]
        dataY = data.iloc[:,data.shape[1]-1]
        dataX = dataX.as_matrix()
        dataY = dataY.as_matrix()
        return dataX, dataY

    def plot_data(df, title="RT Learner - RMSE vs Leaf Size", xlabel="Leaf Size", ylabel="RMSE"):
        """Plot stock prices with a custom title and meaningful axis labels."""
        ax = df.plot(title=title, fontsize=12)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.show()
      
    data = pd.read_csv('istanbul.csv',delimiter=',')
    train=data.sample(frac=0.6,random_state=200)
    test=data.drop(train.index)

    trainX, trainY = create_array(train)
    testX, testY = create_array(test)

    start_time = time.time()

    learner = RTLearner(leaf_size = 1, verbose = False) # constructor
    learner.addEvidence(trainX, trainY)

    Y_pred = learner.query(trainX)
    print "The IN Sample RMSE for DT Learner is:"
    print compute_rmse(trainY, Y_pred)
    print

    Y_pred = learner.query(testX)
    print "The OUT of Sample RMSE for DT Learner is:"
    print compute_rmse(testY, Y_pred)
    print 

    leaf_sizes = [1,3,5,7,10,15,20,25,30,35,40,45, 50,60,70, 80, 90,100]
    rmses = np.empty(shape = (len(leaf_sizes),3))
    #rmses = pd.DataFrame(index=index, columns=columns)


    for i  in range(0, len(leaf_sizes)):
        leaf = leaf_sizes[i]
        print "Leaf Size: " + str(i)
        rmses[i,0] = leaf

        learner = RTLearner(leaf_size = leaf, verbose = False) # constructor
        learner.addEvidence(trainX, trainY)

        print "The IN Sample RMSE for DT Learner is:"
        Y_pred = learner.query(trainX)
        rmse = compute_rmse(trainY, Y_pred)
        rmses[i,1] = rmse
        print rmse

        print "The OUT of Sample RMSE for DT Learner is:"
        Y_pred = learner.query(testX)
        rmse = compute_rmse(testY, Y_pred)
        rmses[i,2] = rmse
        print rmse

    df = pd.DataFrame(rmses, columns=['leaf_size', 'train_rmse', 'test_rmse'])
    df = df.set_index('leaf_size')
    print df

    plot_data(df)

    end_time = time.time()
    print end_time - start_time
    '''

