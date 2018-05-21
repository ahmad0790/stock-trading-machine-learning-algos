"""
Implementing QLearner
Name:Ahmad Khan
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.s = 0
        self.a = 0
        self.q = np.zeros((num_states,num_actions), dtype = float)
        self.gamma = gamma
        self.alpha = alpha
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.t = np.full((num_states,num_actions, num_states), 0.00001, dtype = float)
        self.R = np.zeros((num_states,num_actions), dtype = float)
        self.t_prob = np.zeros((self.num_states,self.num_actions,self.num_states), dtype = float)

        self.states = set()
        self.action_set = set()


    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """

        '''
        self.s = s
        action = rand.randint(0, self.num_actions-1)
        if self.verbose: print "s =", s,"a =",action
        return action
        '''

        self.s = s  

        if (np.random.random() <= self.rar):
            action = np.random.randint(0, self.num_actions)
            self.a = action
            return action

        else:
            action = np.argmax(self.q[s,:])
            self.a = action
            return action


    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The new state
        @returns: The selected action
        """
        s = self.s
        a = self.a
        q = self.q
        t = self.t
        R = self.R

        q[s,a] = (1-self.alpha)*q[s,a] + self.alpha * (r + self.gamma*q[s_prime, np.argmax(q[s_prime,:])])

        #update action
        if (np.random.random() <= self.rar):
            action = np.random.randint(0, self.num_actions)

            self.s = s_prime
            self.a = action
            self.rar = self.rar * self.radr

        else:
            action = np.argmax(self.q[s_prime,:])
            self.s = s_prime
            self.a = action
            self.rar = self.rar * self.radr


        #let's hallucinate an experience many times
        if self.dyna > 0:

            s_list = list(range(0,self.num_states))
            a_list = list(range(0,self.num_actions))
            
            t[s,a,s_prime] = t[s,a,s_prime] + 1
            R[s, a] = (1-self.alpha)*R[s,a] + self.alpha*r
            
            self.t_prob[s,a,:] = t[s,a,:]*1.00/t[s,a,:].sum()

 
            for i in range(0,self.dyna):
                ##initialize the experience tuple
                
                #s = np.random.randint(0,self.num_states)
                #a = np.random.randint(0,self.num_actions)   
                s = rand.choice(s_list)
                a = rand.choice(a_list)
                
                s_prime = np.argmax(self.t_prob[s,a,:])
                r = R[s,a]

                #update Q again
                q[s,a] = (1-self.alpha)*q[s,a] + self.alpha * (r + self.gamma*q[s_prime, np.argmax(q[s_prime,:])])


            self.q = q
            self.t = t
            self.R = R
        
        if self.verbose: 
            print "s =", s_prime,"a =",action,"r =",r
        
        return action
        

    def author(self):
        return 'akhan361'

if __name__=="__main__":
    learner = QLearner(dyna = 25)

    s = 99 # our initial state

    a = learner.querysetstate(s) # action for state s

    s_prime = 5 # the new state we end up in after taking action a in state s

    r = 10 # reward for taking action a in state s

    next_action = learner.query(s_prime, r)
    #next_action = learner.query(s_prime, r)

