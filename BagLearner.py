#Name: Ahmad Khan
#GT ID: akhan361

import numpy as np
import pandas as pd
import random
import DTLearner as dt
import RTLearner as rt
from scipy import stats

#import LinRegLearner as lr
import matplotlib.pyplot as plt

class BagLearner(object):

    def __init__(self, learner,kwargs, bags, boost, verbose):
        pass # move along, these aren't the drones you're looking for
        self.bags = bags
        self.learner = learner
        self.verbose=verbose
        self.kwargs=kwargs
        self.boost=boost
        self.baggedLearners = []


    def author(self):
        return 'akhan361' # replace tb34 with your Georgia Tech username

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        newdataX = np.column_stack((dataX, dataY))

        for k in range(0, self.bags):
            self.baggedLearners.append(self.learner(**self.kwargs))

        for bagLearner in self.baggedLearners:
            randomSample = np.empty(shape=(newdataX.shape[0],newdataX.shape[1]))
            for i in range(0,newdataX.shape[0]):
                randomRow = np.random.randint(0, newdataX.shape[0])

                for j in range(0, newdataX.shape[1]):
                    randomSample[i,j] = newdataX[randomRow,j]   
            
            bagLearner.addEvidence(randomSample[:,0:newdataX.shape[1]-1], randomSample[:,newdataX.shape[1]-1])

        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """

        results = np.zeros(shape=(points.shape[0]))
        for l in self.baggedLearners:
            #print l
            y_preds = l.query(points)
            #print y_preds
            #results = results + y_preds
            results = np.column_stack((results,y_preds))

        #results = results/self.bags
        results = results[:,1:]
        results = stats.mode(results, axis = 1)[0][:]
        return results

 

if __name__=="__main__":
    
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

    def plot_data(df, title="RMSE vs Bag Size", xlabel="Bag Size", ylabel="RMSE"):
        """Plot stock prices with a custom title and meaningful axis labels."""
        ax = df.plot(title=title, fontsize=12)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.show()
      
    data = pd.read_csv('winequality-red.csv',delimiter=',')
    train=data.sample(frac=0.6,random_state=200)
    test=data.drop(train.index)

    trainX, trainY = create_array(train)
    testX, testY = create_array(test)

    learner = BagLearner(learner =rt.RTLearner, kwargs = {"leaf_size":5}, bags = 20, boost = False, verbose = False)
    learner.addEvidence(trainX, trainY)
    Y_pred = learner.query(testX)
    print compute_rmse(testY, Y_pred)

    learner = BagLearner(learner =dt.DTLearner, kwargs = {"leaf_size":5}, bags = 20, boost = False, verbose = False)
    learner.addEvidence(trainX, trainY)
    Y_pred = learner.query(testX)
    print compute_rmse(testY, Y_pred)

    learner = BagLearner(learner =lr.LinRegLearner, kwargs = {}, bags = 20, boost = False, verbose = False)
    learner.addEvidence(trainX, trainY)
    Y_pred = learner.query(testX)
    print compute_rmse(testY, Y_pred)


    #rmses = pd.DataFrame(index=index, columns=columns)

    def plot_error_with_bag_change(bag_sizes, data, trainX, trainY, learner_algo, title):
        for i  in range(0, len(bag_sizes)):
            bag = bag_sizes[i]
            print "Bag Size: " + str(bag)
            rmses[i,0] = bag

            if learner_algo != lr.LinRegLearner:
                learner = BagLearner(learner = learner_algo, kwargs = {"leaf_size":5}, bags = bag, boost = False, verbose = False)
            else:
                learner = BagLearner(learner = learner_algo, kwargs = {}, bags = bag, boost = False, verbose = False)

            learner.addEvidence(trainX, trainY)
            print "The IN Sample RMSE for Bag Learner is:"
            Y_pred = learner.query(trainX)
            rmse = compute_rmse(trainY, Y_pred)
            rmses[i,1] = rmse
            print rmse

            print "The OUT of Sample RMSE for Bag Learner is:"
            Y_pred = learner.query(testX)
            rmse = compute_rmse(testY, Y_pred)
            rmses[i,2] = rmse
            print rmse

        df = pd.DataFrame(rmses, columns=['bag_size', 'train_rmse', 'test_rmse'])
        df = df.set_index('bag_size')
        #print df
        plot_data(df, title)


    def plot_error_with_leaf_change(bag, leaf_sizes, data, trainX, trainY, title):
        for i  in range(0, len(leaf_sizes)):
            leaf = leaf_sizes[i]
            print "Leaf Size: " + str(i)
            rmses[i,0] = leaf

            learner = BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size":leaf}, bags = bag, boost = False, verbose = False) # constructor
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
        plot_data(df, title, xlabel="Leaf Size")


    bag_sizes = [1,3,5,7,10,15,20,25,30,35,40,45, 50,75,100]
    rmses = np.empty(shape = (len(bag_sizes),3))

    plot_error_with_bag_change(bag_sizes, rmses, trainX, trainY, lr.LinRegLearner, "LinReg Bagging - RMSE vs Bag Size")
    plot_error_with_bag_change(bag_sizes, rmses, trainX, trainY, dt.DTLearner, "DT Bagging - RMSE vs Bag Size")
    plot_error_with_bag_change(bag_sizes, rmses, trainX, trainY, rt.RTLearner, "RT Baggin - RMSE vs Bag Size")


    leaf_sizes = [1,3,5,7,10,15,20,25,30,35,40,45,50]
    rmses = np.empty(shape = (len(leaf_sizes),3))
    plot_error_with_leaf_change(15, leaf_sizes, rmses, trainX, trainY, "RT with Bagging RMSE vs Leaf Size")
    











