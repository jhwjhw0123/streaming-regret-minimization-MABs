import os
import pickle
import numpy as np
import random
import copy
import tqdm
from scipy.stats import truncnorm


def truncated_normal(mean=0.5, sd=0.1, low=0.0, upp=0.8):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)



# define a class of a Bernoulli arm / Coin 
class Bernoulli_Arm:
    '''
    The class of an arm whose reward follows a Bernoulli distribution (aka a coin)
    ----- Parameters -----
    # p: the probability for the arm to give a reward
    ----- Methods ----
    # pull(): make an arm pull; return the reward
    # batch_pull(num_pull): make *num_pull* times of arm pulls  
    '''

    def __init__(self, p=None, random_mean = 'uniform'):
        '''
        :param p: the probability for the arm to give a reward (bias of a coin)
        '''
        if p is not None:
            self.p = p
        elif random_mean=='uniform':
            self.p = np.random.uniform()
        elif random_mean=='gaussian':
            truc_gaussian_sampler = truncated_normal()
            self.p = truc_gaussian_sampler.rvs()
        else:
            raise ValueError('Distribution of the mean not recognized!')

    def pull(self):
        '''
        make an arm pull by rejection sampling
        '''
        this_sample = np.random.uniform()
        
        if this_sample<=self.p:
            return 1.0
        else:
            return 0.0
        
    def batch_pull(self, num_pull):
        '''
        make *num_pull* times of arm pulls, return the average
        '''
        
        if num_pull<=1e5:
            # use vectorized operations to speed up
            vec_sample = np.random.uniform(size=[(int)(num_pull)])
            # implementation: p subtract vec_sample is either >=0 or <0, same as rejection sampling
            # taking the ceiling tells us how many pulls gives reward
            # take the average of the rewards
            # add 1e-20 so that the reward will not round to -1
        
            return np.average(np.ceil(self.p - vec_sample + 1e-20))
        else:
            # sampling a gaussian noise to approximate
            return self.p + np.random.normal(0, 2/np.sqrt(num_pull))


# We need to code a coin-feeding buffer to simulate the `bank'
# such that inside the algorithms it truly simulates a `local memory'
class Arm_Reading_Buffer:
    '''
    The class to read arms one after another -- simulating the 'bank' that feeds the stream with arms
    ----- Parameters -----
    # arm_set: the set of arms. supported type numpy array
    # n: number of arms in the set
    # best_arm: the index of the best arm
    # Delta: the gap between the best and the second-best arm
    ----- Methods ----
    # read_next_arm(): read the next arm and move the index +1
    '''
    
    def __init__(self, arm_set):
        '''
        :param arm_set: the set of arms. supported type numpy array
        '''
        if type(arm_set) is not np.ndarray:
            raise ValueError('Expecting numpy array of arms. Please convert')
        # flatten the array if it's not already
        self.arm_set = np.reshape(arm_set, [-1])
        # number of arms
        self.n = np.shape(self.arm_set)[0]
        # collect the reward parameters
        list_rewards = []
        for arm in self.arm_set:
            list_rewards.append(arm.p)
        array_rewards = np.array(list_rewards)
        # sort and find the indices for best and second-best arms
        sorted_indices = np.argsort(-array_rewards)
        self._best_index = sorted_indices[0]
        self._runner_up_index = sorted_indices[1]
        # collect the best arms -- information withholding from the algorithm
        self._best_arm = self.arm_set[self._best_index]
        self._runner_up_arm = self.arm_set[self._runner_up_index]
        self._best_reward = array_rewards[self._best_index]
        self._runner_up_reward = array_rewards[self._runner_up_index]
        # the gap is supposed to be known
        self.Delta = self._best_reward - self._runner_up_reward
        # the sub-exponential gaps as the budget
        self._Delta_list = (self._best_reward - array_rewards[sorted_indices])[1:]
        self.H2 = np.sum((1/np.square(self._Delta_list)) * np.log(np.log(1/self._Delta_list)))
        # the internal index
        self._current_arm_ind = 0
        self._terminate_flag = False
        # maintain a counter for the regret
        self.regret_counter = 0
        

    def read_next_arm(self):
        '''
        Read the next arm, and increase the index by 1
        '''
        if not self._terminate_flag:
            # read the current arm
            return_arm = self.arm_set[self._current_arm_ind]
            if (self._current_arm_ind == self.n - 1):
                self._terminate_flag = True
            # increase the index by 1
            self._current_arm_ind = self._current_arm_ind + 1
            
            return return_arm
        else:
            # stream at the end
            return None
    
    def reset(self):
        '''
        Reset the pointer and the termination flag so that one can reuse the stream
        '''
        self._current_arm_ind = 0
        self._terminate_flag = False
        self.regret_counter = 0
        
    def regret_eval(self, arm_reward, arm_pulls=1):
        '''
        Incur a cost of (best_reward - arm_reward)*arm_pulls
        '''
        
        self.regret_counter = self.regret_counter + (self._best_reward-arm_reward)*arm_pulls





