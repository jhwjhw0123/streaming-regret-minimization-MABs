import os
import pickle
import numpy as np
import random
import copy
import tqdm


# the regret-minimization algorithm with the uniform exploration algorithm
def uniform_exploration_algorithm(arm_stream, T, verbose=True):
    '''
    :param arm_stream: the streaming buffer that `feeds' the arms
    :param T: the horizon of the number of trials
    :param verbose: flag to control whether to print information
    '''
    # read the number of arms -- neccessary for the uniform exploration algorithm
    num_arms = arm_stream.n
    # sample times for each arm -- the optimal number in uniform exploration
    batch_size = ((T/num_arms)**(2/3))*(np.log(T)**(1/3))
    # memory list 
    stored_arm = []
    buffer_arm = []
    # memory of the best reward
    stored_reward = 0
    # arm index 
    arm_ind = 0
    current_ind = 0
    # keep track of the total number of arm pulls
    total_arm_pulls = 0
    # batch size for arm pulling
    while(1):
        if verbose:
            print('\rProcessing arm number '+str(current_ind+1)+'..', end="")
        if total_arm_pulls>=T:
            print('Trial budgets exhausted!')
            break
        # read the next arm
        if not stored_arm:
            # this only happens for the first arm
            stored_arm.append(arm_stream.read_next_arm())
            best_stored_reward = stored_arm[0].batch_pull(batch_size)
            total_arm_pulls = total_arm_pulls + batch_size
            arm_stream.regret_eval(stored_arm[0].p, batch_size)
        else:
            # read the arm to the 'buffer' for the extra arm
            current_arm = arm_stream.read_next_arm()
            current_ind = current_ind + 1
            if current_arm is None:
                # end of the stream
                break
            # If stream not ended, compare
            buffer_reward = current_arm.batch_pull(batch_size)
            total_arm_pulls = total_arm_pulls + batch_size
            arm_stream.regret_eval(current_arm.p, batch_size)
            if verbose:
                print('\rProcessing arm number '+str(current_ind+1)+'..', end="")
            # replace the stored arm if it's beaten
            if stored_reward>=buffer_reward:
                continue
            else:
                stored_arm.append(current_arm)
                arm_ind = current_ind + 0
                stored_arm.pop(0)
                stored_reward = buffer_reward
                
    # pulling the arm that we commit to if there is any remaining trials
    remain_trial_nums = T - total_arm_pulls
    if remain_trial_nums>=0:
        arm_stream.regret_eval(stored_arm[0].p, remain_trial_nums)
    
    return stored_arm[0], arm_ind, arm_stream.regret_counter



# the regret-minimization algorithm with the native (n*log(n)/Delta^2) arm-pull eps-best arm algorithm
def naive_eps_best_algorithm(arm_stream, T, verbose=True):
    '''
    :param arm_stream: the streaming buffer that `feeds' the arms
    :param T: the horizon of the number of trials
    '''
    # read the number of arms -- neccessary for the naive algorithm
    num_arms = arm_stream.n
    # set the eps parameter
    eps = ((num_arms/T)**(1/3))
    # sample times for each arm
    batch_size = np.log(num_arms)/(eps**2)
    # memory list 
    stored_arm = []
    buffer_arm = []
    # memory of the best reward
    stored_reward = 0
    # arm index 
    arm_ind = 0
    current_ind = 0
    # keep track of the total number of arm pulls
    total_arm_pulls = 0
    # batch size for arm pulling
    while(1):
        if verbose:
            print('\rProcessing arm number '+str(current_ind+1)+'..', end="")
        if total_arm_pulls>=T:
            print('Trial budgets exhausted!')
            break
        # read the next arm
        if not stored_arm:
            # this only happens for the first arm
            stored_arm.append(arm_stream.read_next_arm())
            best_stored_reward = stored_arm[0].batch_pull(batch_size)
            total_arm_pulls = total_arm_pulls + batch_size
            arm_stream.regret_eval(stored_arm[0].p, batch_size)
        else:
            # read the arm to the 'buffer' for the extra arm
            current_arm = arm_stream.read_next_arm()
            current_ind = current_ind + 1
            if current_arm is None:
                # end of the stream
                break
            # If stream not ended, compare
            buffer_reward = current_arm.batch_pull(batch_size)
            total_arm_pulls = total_arm_pulls + batch_size
            arm_stream.regret_eval(current_arm.p, batch_size)
            if verbose:
                print('\rProcessing arm number '+str(current_ind+1)+'..', end="")
            # replace the stored arm if it's beaten
            if stored_reward>=buffer_reward:
                continue
            else:
                stored_arm.append(current_arm)
                arm_ind = current_ind + 0
                stored_arm.pop(0)
                stored_reward = buffer_reward
                
    # pulling the arm that we commit to if there is any remaining trials
    remain_trial_nums = T - total_arm_pulls
    if remain_trial_nums>=0:
        arm_stream.regret_eval(stored_arm[0].p, remain_trial_nums)
    
    return stored_arm[0], arm_ind, arm_stream.regret_counter



def best_among_remaining(arm_stream, arm_list, num_pull):
    '''
    :param arm_stream: the streaming buffer that `feeds' the arms
    :param arm_list: list of arms, expect length to be equal to the number of levels, 
                        and each level contains a single arm
    :param num_pull: the number of arm pulls on each level
    '''
    best_arm_level = 0
    best_emp_reward = 0
    for level in range(len(arm_list)):
        if arm_list[level]:
            current_emp_reward = arm_list[level][0].batch_pull(num_pull)
            arm_stream.regret_eval(arm_list[level][0].p, num_pull)
            if current_emp_reward>best_emp_reward:
                best_arm_level = level
    
    return best_arm_level 



# the regret-minimization algorithm with the O(log(n)) space
def log_eps_best_algorithm(arm_stream, T, verbose=True):
    '''
    :param arm_stream: the streaming buffer that `feeds' the arms
    :param T: the horizon of the number of trials
    '''
    # read the number of arms -- necessary for the regret-minimization problem
    num_arms = arm_stream.n
    # set the eps parameter
    eps = ((num_arms/T)**(1/3))
    # determine the levels of the tower -- we will discard at a rate of 4 so log_e is definitely enough
    num_levels = (int)(np.log(num_arms))
    # memory list -- every level keeps an empty list in the begining
    levels_stored_arms = []
    for level in range(num_levels):
        levels_stored_arms.append([])
    buffer_arm = []
    # batch size of each level
    levels_batch_size = []
    for level in range(num_levels):
        levels_batch_size.append(np.ceil(1.2**(level)*(1/eps**2)))
    # sample complexity on each level
    levels_sample_caps = []
    for level in range(num_levels):
        levels_sample_caps.append(np.ceil(6*levels_batch_size[level]))
    # memory of the best reward
    levels_stored_reward = [0]*num_levels
    # arm index 
    arm_ind = 0
    current_ind = 0
    # keep track of the total number of arm pulls
    total_arm_pulls = 0
    level_arm_pulls = [0]*num_levels
    # count the number of level the algorithm ever reached
    max_level = 0
    # batch size for arm pulling
    while(1):
        if verbose:
            print('\rProcessing arm number '+str(current_ind+1)+'..', end="")
        for level in range(num_levels):
            if max_level<level:
                max_level = level
            # first check if the level budget is exhausted
            if level_arm_pulls[level]<=levels_sample_caps[level]:
                # if not --> ordinary operations, and break and for loop afterwards
                # read the next arm
                if not levels_stored_arms[level]:
                    # this happens for the first arm or the first level is 'cleared'
                    # will only happen to the first level
                    current_arm = arm_stream.read_next_arm()
                    # if no next arm then terminate, run the remaining trials and return
                    if current_arm is None:
                        remain_trial_nums = T - total_arm_pulls
                        if remain_trial_nums>=max_level*levels_batch_size[0]:
                            best_arm_level = best_among_remaining(arm_stream, 
                                                                  levels_stored_arms, 
                                                                  levels_batch_size[0])
                            remain_trial_nums = remain_trial_nums - max_level*levels_batch_size[0]
                        else:
                            best_arm_level = max_level
                        if remain_trial_nums>=0:
                            arm_stream.regret_eval(levels_stored_arms[best_arm_level][0].p, remain_trial_nums)
                            
                        return levels_stored_arms[best_arm_level][0], arm_ind, arm_stream.regret_counter
                    # else proceed as normal
                    levels_stored_arms[level].append(current_arm)
                    buffer_reward = current_arm.batch_pull(levels_batch_size[level])
                    if buffer_reward>levels_stored_reward[level]:
                        levels_stored_reward[level] = buffer_reward
                    total_arm_pulls = total_arm_pulls + levels_batch_size[level]
                    level_arm_pulls[level] = level_arm_pulls[level] + levels_batch_size[level]
                    arm_stream.regret_eval(levels_stored_arms[level][0].p, 
                                           levels_batch_size[level])
                else:
                    # read the arm to the 'buffer' for the extra arm
                    if level == 0:
                        current_arm = arm_stream.read_next_arm()
                        # if no next arm then terminate, run the remaining trials and return
                        if current_arm is None:
                            remain_trial_nums = T - total_arm_pulls
                            if remain_trial_nums>=max_level*levels_batch_size[0]:
                                best_arm_level = best_among_remaining(arm_stream, 
                                                                      levels_stored_arms, 
                                                                      levels_batch_size[0])
                                remain_trial_nums = remain_trial_nums - max_level*levels_batch_size[0]
                            else:
                                best_arm_level = max_level
                            if remain_trial_nums>=0:
                                arm_stream.regret_eval(levels_stored_arms[best_arm_level][0].p, remain_trial_nums)
                                
                            return levels_stored_arms[best_arm_level][0], arm_ind, arm_stream.regret_counter
                    else:
                        if len(levels_stored_arms[level])>1:
                            current_arm = levels_stored_arms[level][-1]
                            levels_stored_arms[level].pop(-1)
                        else:
                            # if there's only one arm in this level -- record the level empirical reward first
                            levels_stored_reward[level] = levels_stored_arms[level][0].batch_pull(levels_batch_size[level])
                            arm_stream.regret_eval(levels_stored_arms[level][0].p, levels_batch_size[level])
                            break
                    # compare the current arm with the benchmark of the current level
                    buffer_reward = current_arm.batch_pull(levels_batch_size[level])
                    total_arm_pulls = total_arm_pulls + levels_batch_size[level]
                    level_arm_pulls[level] = level_arm_pulls[level] + levels_batch_size[level]
                    arm_stream.regret_eval(levels_stored_arms[level][0].p, levels_batch_size[level])
                    if level == 0:
                        current_ind = current_ind + 1
                        if verbose:
                            print('\rProcessing arm number '+str(current_ind+1)+'..', end="")
                    # replace the stored arm if it's beaten
                    if levels_stored_reward[level]>=buffer_reward:
                        pass
                    else:
                        levels_stored_arms[level].append(current_arm)
                        if level == 0:
                            arm_ind = current_ind
                        levels_stored_arms[level].pop(0)
                        levels_stored_reward[level] = buffer_reward
                if total_arm_pulls>=T:
                    print('Trial budgets exhausted!')
                    
                    return levels_stored_arms[max_level][0], arm_ind, arm_stream.regret_counter
                # break here to not going to the next level -- since this level is not full
                break
            else:
                # send the arm to the higher level and continue the subroutine
                if len(levels_stored_arms[level])>1:
                    arm_one_reward = levels_stored_arms[level][0].batch_pull(levels_batch_size[level])
                    arm_two_reward = levels_stored_arms[level][1].batch_pull(levels_batch_size[level])
                    arm_stream.regret_eval(levels_stored_arms[level][0].p, levels_batch_size[level])
                    arm_stream.regret_eval(levels_stored_arms[level][1].p, levels_batch_size[level])
                    if arm_one_reward >arm_two_reward:
                        levels_stored_arms[level+1].append(levels_stored_arms[level][0])
                    else:
                        levels_stored_arms[level+1].append(levels_stored_arms[level][1])
                else:
                    levels_stored_arms[level+1].append(levels_stored_arms[level][0])
                levels_stored_arms[level].clear()
                level_arm_pulls[level] = 0
                
                
    # pulling the arm that we commit to if there is any remaining trials
    remain_trial_nums = T - total_arm_pulls
    if remain_trial_nums>=max_level*levels_batch_size[0]:
        best_arm_level = best_among_remaining(arm_stream, 
                                              levels_stored_arms, 
                                              levels_batch_size[0])
        remain_trial_nums = remain_trial_nums - max_level*levels_batch_size[0]
    else:
        best_arm_level = max_level
    if remain_trial_nums>=0:
        arm_stream.regret_eval(levels_stored_arms[best_arm_level][0].p, remain_trial_nums)
    return levels_stored_arms[best_arm_level][0], arm_ind, arm_stream.regret_counter




# the regret-minimization algorithm with the O(log(n)) space
def loglog_eps_best_algorithm(arm_stream, T, verbose=True):
    '''
    :param arm_stream: the streaming buffer that `feeds' the arms
    :param T: the horizon of the number of trials
    '''
    # read the number of arms -- necessary for the regret-minimization problem
    num_arms = arm_stream.n
    # set the eps parameter
    eps = ((num_arms/T)**(1/3))
    # determine the levels of the tower -- we will discard at a rate of 4 so log_e is definitely enough
    num_levels = (int)(np.log(np.log(num_arms)))+1
    # memory list -- every level keeps an empty list in the begining
    levels_stored_arms = []
    for level in range(num_levels):
        levels_stored_arms.append([])
    buffer_arm = []
    # batch size of each level
    levels_batch_size = []
    for level in range(num_levels):
        if level<num_levels-1:
            levels_batch_size.append(np.ceil(1.2**(level)*(1/eps**2)))
        else:
            # change the last level to n*log(n)-sample comparison
            levels_batch_size.append(np.log(num_arms)/(eps**2))
    # sample complexity on each level
    levels_sample_caps = []
    for level in range(num_levels):
        if level<num_levels-1:
            levels_sample_caps.append(np.ceil(10*levels_batch_size[level]))
        else:
            # this will not be capped
            levels_sample_caps.append(np.ceil(num_arms*levels_batch_size[level]))
    # memory of the best reward
    levels_stored_reward = [0]*num_levels
    # arm index 
    arm_ind = 0
    current_ind = 0
    # keep track of the total number of arm pulls
    total_arm_pulls = 0
    level_arm_pulls = [0]*num_levels
    # count the number of level the algorithm ever reached
    max_level = 0
    # batch size for arm pulling
    while(1):
        if verbose:
            print('\rProcessing arm number '+str(current_ind+1)+'..', end="")
        for level in range(num_levels):
            if max_level<level:
                max_level = level
            # first check if the level budget is exhausted
            if level_arm_pulls[level]<=levels_sample_caps[level]:
                # if not --> ordinary operations, and break and for loop afterwards
                # read the next arm
                if not levels_stored_arms[level]:
                    # this happens for the first arm or the first level is 'cleared'
                    # will only happen to the first level
                    current_arm = arm_stream.read_next_arm()
                    # if no next arm then terminate, run the remaining trials and return
                    if current_arm is None:
                        remain_trial_nums = T - total_arm_pulls
                        if remain_trial_nums>=max_level*levels_batch_size[0]:
                            best_arm_level = best_among_remaining(arm_stream, 
                                                                  levels_stored_arms, 
                                                                  levels_batch_size[0])
                            remain_trial_nums = remain_trial_nums - max_level*levels_batch_size[0]
                        else:
                            best_arm_level = max_level
                        if remain_trial_nums>=0:
                            arm_stream.regret_eval(levels_stored_arms[best_arm_level][0].p, remain_trial_nums)
                            
                        return levels_stored_arms[best_arm_level][0], arm_ind, arm_stream.regret_counter
                    # else proceed as normal
                    levels_stored_arms[level].append(current_arm)
                    buffer_reward = current_arm.batch_pull(levels_batch_size[level])
                    if buffer_reward>levels_stored_reward[level]:
                        levels_stored_reward[level] = buffer_reward
                    total_arm_pulls = total_arm_pulls + levels_batch_size[level]
                    level_arm_pulls[level] = level_arm_pulls[level] + levels_batch_size[level]
                    arm_stream.regret_eval(levels_stored_arms[level][0].p, 
                                           levels_batch_size[level])
                else:
                    # read the arm to the 'buffer' for the extra arm
                    if level == 0:
                        current_arm = arm_stream.read_next_arm()
                        # if no next arm then terminate, run the remaining trials and return
                        if current_arm is None:
                            remain_trial_nums = T - total_arm_pulls
                            if remain_trial_nums>=max_level*levels_batch_size[0]:
                                best_arm_level = best_among_remaining(arm_stream, 
                                                                      levels_stored_arms, 
                                                                      levels_batch_size[0])
                                remain_trial_nums = remain_trial_nums - max_level*levels_batch_size[0]
                            else:
                                best_arm_level = max_level
                            if remain_trial_nums>=0:
                                arm_stream.regret_eval(levels_stored_arms[best_arm_level][0].p, remain_trial_nums)
                                
                            return levels_stored_arms[best_arm_level][0], arm_ind, arm_stream.regret_counter
                    else:
                        if len(levels_stored_arms[level])>1:
                            current_arm = levels_stored_arms[level][-1]
                            levels_stored_arms[level].pop(-1)
                        else:
                            # if there's only one arm in this level -- record the level empirical reward first
                            levels_stored_reward[level] = levels_stored_arms[level][0].batch_pull(levels_batch_size[level])
                            arm_stream.regret_eval(levels_stored_arms[level][0].p, levels_batch_size[level])
                            break
                    # compare the current arm with the benchmark of the current level
                    buffer_reward = current_arm.batch_pull(levels_batch_size[level])
                    total_arm_pulls = total_arm_pulls + levels_batch_size[level]
                    level_arm_pulls[level] = level_arm_pulls[level] + levels_batch_size[level]
                    arm_stream.regret_eval(levels_stored_arms[level][0].p, levels_batch_size[level])
                    if level == 0:
                        current_ind = current_ind + 1
                        if verbose:
                            print('\rProcessing arm number '+str(current_ind+1)+'..', end="")
                    # replace the stored arm if it's beaten
                    if levels_stored_reward[level]>=buffer_reward:
                        pass
                    else:
                        levels_stored_arms[level].append(current_arm)
                        if level == 0:
                            arm_ind = current_ind
                        levels_stored_arms[level].pop(0)
                        levels_stored_reward[level] = buffer_reward
                if total_arm_pulls>=T:
                    print('Trial budgets exhausted!')
                    
                    return levels_stored_arms[max_level][0], arm_ind, arm_stream.regret_counter
                # break here to not going to the next level -- since this level is not full
                break
            else:
                # send the arm to the higher level and continue the subroutine
                if len(levels_stored_arms[level])>1:
                    arm_one_reward = levels_stored_arms[level][0].batch_pull(levels_batch_size[level])
                    arm_two_reward = levels_stored_arms[level][1].batch_pull(levels_batch_size[level])
                    arm_stream.regret_eval(levels_stored_arms[level][0].p, levels_batch_size[level])
                    arm_stream.regret_eval(levels_stored_arms[level][1].p, levels_batch_size[level])
                    if arm_one_reward >arm_two_reward:
                        levels_stored_arms[level+1].append(levels_stored_arms[level][0])
                    else:
                        levels_stored_arms[level+1].append(levels_stored_arms[level][1])
                else:
                    levels_stored_arms[level+1].append(levels_stored_arms[level][0])
                levels_stored_arms[level].clear()
                level_arm_pulls[level] = 0
                
                
    # pulling the arm that we commit to if there is any remaining trials
    remain_trial_nums = T - total_arm_pulls
    if remain_trial_nums>=max_level*levels_batch_size[0]:
        best_arm_level = best_among_remaining(arm_stream, 
                                              levels_stored_arms, 
                                              levels_batch_size[0])
        remain_trial_nums = remain_trial_nums - max_level*levels_batch_size[0]
    else:
        best_arm_level = max_level
    if remain_trial_nums>=0:
        arm_stream.regret_eval(levels_stored_arms[best_arm_level][0].p, remain_trial_nums)
    return levels_stored_arms[best_arm_level][0], arm_ind, arm_stream.regret_counter



# the regret-minimization algorithm with the O(log(n)) space
def logstar_eps_best_algorithm(arm_stream, T, verbose=True):
    '''
    :param arm_stream: the streaming buffer that `feeds' the arms
    :param T: the horizon of the number of trials
    '''
    # read the number of arms -- necessary for the regret-minimization problem
    num_arms = arm_stream.n
    # set the eps parameter
    eps = ((num_arms/T)**(1/3))
    # determine the levels of the tower -- we will discard at a rate of 4 so log_e is definitely enough
    num_levels = 1
    log_ite = num_arms + 0
    while log_ite>=1:
        num_levels = num_levels + 1
        log_ite = np.log(log_ite)
    # memory list -- every level keeps an empty list in the begining
    levels_stored_arms = []
    for level in range(num_levels):
        levels_stored_arms.append([])
    buffer_arm = []
    # batch size of each level
    levels_batch_size_r = [2]
    levels_batch_size = [1/eps**2]
    for level in range(num_levels-1):
        levels_batch_size_r.append(2**levels_batch_size_r[level])
        levels_batch_size.append(np.minimum(np.ceil(levels_batch_size_r[level+1]*(1/eps**2)), 1e20))
    # sample complexity on each level
    levels_sample_caps = [4*(1/eps**2)]
    for level in range(num_levels-1):
        try:
            # might easily become too large
            levels_sample_caps.append(np.ceil((2**levels_batch_size_r[level+1])*levels_batch_size[level+1]/(2**levels_batch_size_r[level])))
        except:
            levels_sample_caps.append(np.ceil(1e10*levels_batch_size[level+1]))
    # memory of the best reward
    levels_stored_reward = [0]*num_levels
    # arm index 
    arm_ind = 0
    current_ind = 0
    # keep track of the total number of arm pulls
    total_arm_pulls = 0
    level_arm_pulls = [0]*num_levels
    # count the number of level the algorithm ever reached
    max_level = 0
    # batch size for arm pulling
    while(1):
        if verbose:
            print('\rProcessing arm number '+str(current_ind+1)+'..', end="")
        for level in range(num_levels):
            if max_level<level:
                max_level = level
            # first check if the level budget is exhausted
            if level_arm_pulls[level]<=levels_sample_caps[level]:
                # if not --> ordinary operations, and break and for loop afterwards
                # read the next arm
                if not levels_stored_arms[level]:
                    # this happens for the first arm or the first level is 'cleared'
                    # will only happen to the first level
                    current_arm = arm_stream.read_next_arm()
                    # if no next arm then terminate, run the remaining trials and return
                    if current_arm is None:
                        remain_trial_nums = T - total_arm_pulls
                        if remain_trial_nums>=max_level*levels_batch_size[0]:
                            best_arm_level = best_among_remaining(arm_stream, 
                                                                  levels_stored_arms, 
                                                                  levels_batch_size[0])
                            remain_trial_nums = remain_trial_nums - max_level*levels_batch_size[0]
                        else:
                            best_arm_level = max_level
                        if remain_trial_nums>=0:
                            arm_stream.regret_eval(levels_stored_arms[best_arm_level][0].p, remain_trial_nums)
                            
                        return levels_stored_arms[best_arm_level][0], arm_ind, arm_stream.regret_counter
                    # else proceed as normal
                    levels_stored_arms[level].append(current_arm)
                    buffer_reward = current_arm.batch_pull(levels_batch_size[level])
                    if buffer_reward>levels_stored_reward[level]:
                        levels_stored_reward[level] = buffer_reward
                    total_arm_pulls = total_arm_pulls + levels_batch_size[level]
                    level_arm_pulls[level] = level_arm_pulls[level] + levels_batch_size[level]
                    arm_stream.regret_eval(levels_stored_arms[level][0].p, 
                                           levels_batch_size[level])
                else:
                    # read the arm to the 'buffer' for the extra arm
                    if level == 0:
                        current_arm = arm_stream.read_next_arm()
                        # if no next arm then terminate, run the remaining trials and return
                        if current_arm is None:
                            remain_trial_nums = T - total_arm_pulls
                            if remain_trial_nums>=max_level*levels_batch_size[0]:
                                best_arm_level = best_among_remaining(arm_stream, 
                                                                      levels_stored_arms, 
                                                                      levels_batch_size[0])
                                remain_trial_nums = remain_trial_nums - max_level*levels_batch_size[0]
                            else:
                                best_arm_level = max_level
                            if remain_trial_nums>=0:
                                arm_stream.regret_eval(levels_stored_arms[best_arm_level][0].p, remain_trial_nums)
                                
                            return levels_stored_arms[best_arm_level][0], arm_ind, arm_stream.regret_counter
                    else:
                        if len(levels_stored_arms[level])>1:
                            current_arm = levels_stored_arms[level][-1]
                            levels_stored_arms[level].pop(-1)
                        else:
                            # if there's only one arm in this level -- record the level empirical reward first
                            levels_stored_reward[level] = levels_stored_arms[level][0].batch_pull(levels_batch_size[level])
                            arm_stream.regret_eval(levels_stored_arms[level][0].p, levels_batch_size[level])
                            break
                    # compare the current arm with the benchmark of the current level
                    buffer_reward = current_arm.batch_pull(levels_batch_size[level])
                    total_arm_pulls = total_arm_pulls + levels_batch_size[level]
                    level_arm_pulls[level] = level_arm_pulls[level] + levels_batch_size[level]
                    arm_stream.regret_eval(levels_stored_arms[level][0].p, levels_batch_size[level])
                    if level == 0:
                        current_ind = current_ind + 1
                        if verbose:
                            print('\rProcessing arm number '+str(current_ind+1)+'..', end="")
                    # replace the stored arm if it's beaten
                    if levels_stored_reward[level]>=buffer_reward:
                        pass
                    else:
                        levels_stored_arms[level].append(current_arm)
                        if level == 0:
                            arm_ind = current_ind
                        levels_stored_arms[level].pop(0)
                        levels_stored_reward[level] = buffer_reward
                if total_arm_pulls>=T:
                    print('Trial budgets exhausted!')
                    
                    return levels_stored_arms[max_level][0], arm_ind, arm_stream.regret_counter
                # break here to not going to the next level -- since this level is not full
                break
            else:
                # send the arm to the higher level and continue the subroutine
                if len(levels_stored_arms[level])>1:
                    arm_one_reward = levels_stored_arms[level][0].batch_pull(levels_batch_size[level])
                    arm_two_reward = levels_stored_arms[level][1].batch_pull(levels_batch_size[level])
                    arm_stream.regret_eval(levels_stored_arms[level][0].p, levels_batch_size[level])
                    arm_stream.regret_eval(levels_stored_arms[level][1].p, levels_batch_size[level])
                    if arm_one_reward >arm_two_reward:
                        levels_stored_arms[level+1].append(levels_stored_arms[level][0])
                    else:
                        levels_stored_arms[level+1].append(levels_stored_arms[level][1])
                else:
                    levels_stored_arms[level+1].append(levels_stored_arms[level][0])
                levels_stored_arms[level].clear()
                level_arm_pulls[level] = 0
                
                
    # pulling the arm that we commit to if there is any remaining trials
    remain_trial_nums = T - total_arm_pulls
    if remain_trial_nums>=max_level*levels_batch_size[0]:
        best_arm_level = best_among_remaining(arm_stream, 
                                              levels_stored_arms, 
                                              levels_batch_size[0])
        remain_trial_nums = remain_trial_nums - max_level*levels_batch_size[0]
    else:
        best_arm_level = max_level
    if remain_trial_nums>=0:
        arm_stream.regret_eval(levels_stored_arms[best_arm_level][0].p, remain_trial_nums)
    return levels_stored_arms[best_arm_level][0], arm_ind, arm_stream.regret_counter



# the GAME-OF-ARMs algorithm to find the best coin by O(n/eps^2) arm pulls
def Game_of_Arms_regret_min(arm_stream, T, verbose=True):
    '''
    :param arm_stream: the streaming buffer that `feeds' the arms
    :param T: the number of total trials
    '''
    # read the number of arms -- neccessary for the naive algorithm
    num_arms = arm_stream.n
    # set the eps parameter
    eps = ((num_arms/T)**(1/3))
    # count the number of samples
    sample_complexity_full = 0
    sample_complexity_full_cap = 10*num_arms/eps**2
    sample_complexity_frac = 0
    # in the eps-best arm algorithm, we need to use `guess' of each levels
    king_reward_est = eps/2
    # initalize a budget
    initial_king_budget = 1.2/(eps**2)
    king_budget_increment = 1/(eps**2)
    # memory list 
    stored_arm = []
    buffer_arm = []
    # arm index 
    arm_ind = 0
    current_ind = 0
    this_king_budget = 0
    max_ell = 0
    # define the fractional stream procedure
    sotred_frac_arm = None
    frac_best_reward = 0
    num_frac_arm_sample = np.log(num_arms)/(eps**2)
    # batch size for arm pulling
    while(1):
        if verbose:
            print('\rProcessing arm number '+str(current_ind+1)+'..', end="")
        # read the next arm
        if (not stored_arm) and (sotred_frac_arm is None):
            # this only happens for the first arm
            first_arm = arm_stream.read_next_arm()
            # full-stream procudure
            stored_arm.append(first_arm)
            this_king_budget = initial_king_budget
            # fractional-stream procedure
            sotred_frac_arm = copy.deepcopy(first_arm)
            # update the best reward and collect regret
            frac_best_reward = sotred_frac_arm.batch_pull(num_frac_arm_sample)
            arm_stream.regret_eval(sotred_frac_arm.p, num_frac_arm_sample)
        else:
            # read the arm to the 'buffer' for the extra arm
            current_arm = arm_stream.read_next_arm()
            current_ind = current_ind + 1
            # ************* The following is the 'full stream subroutine' *************
            if sample_complexity_full<=sample_complexity_full_cap:
                this_king_budget = this_king_budget + king_budget_increment
                if current_arm is None:
                    # end of the stream
                    break
                # challenge subroutine -- full stream
                for ell in range((int)(10*np.log(np.log(num_arms)))):
                    if ell>max_ell:
                        max_ell = ell
                    current_batch_size = 1/(eps**2) * (2**ell)
                    if this_king_budget<current_batch_size:
                        # king discard case
                        stored_arm.append(current_arm)
                        # reset arm
                        arm_ind = current_ind + 0
                        stored_arm.pop(0)
                        # reset budget
                        this_king_budget = initial_king_budget
                        # update the estimation and collect regret
                        new_king_reward = current_arm.batch_pull(initial_king_budget)
                        arm_stream.regret_eval(current_arm.p, initial_king_budget)
                        king_reward_est = np.minimum(np.maximum(new_king_reward, 
                                                                king_reward_est + arm_stream.Delta/2), 1.0)
                        sample_complexity_full = sample_complexity_full + initial_king_budget
                        break
                    # record the reward
                    # ell_sotred_reward = stored_arm[0].batch_pull(current_batch_size)
                    ell_buffer_reward = current_arm.batch_pull(current_batch_size)
                    # update the regret
                    arm_stream.regret_eval(current_arm.p, current_batch_size)
                    # update the king budget
                    sample_complexity_full = sample_complexity_full + current_batch_size
                    this_king_budget = this_king_budget - current_batch_size
                    if (king_reward_est-eps/5)>=ell_buffer_reward:
                        break
                    if ell == (int)(10*np.log(np.log(num_arms)))-1:
                        raise ValueError('Algorithm fail for exccessive sample complexity!')  
            else:
                # skip the full stream procedure
                pass
            # ************* The following is the 'partial stream subroutine' *************
            if np.random.uniform()<1.5/np.log(num_arms):
                # pull the arm and collect regret
                frac_current_reward = current_arm.batch_pull(num_frac_arm_sample)
                arm_stream.regret_eval(current_arm.p, num_frac_arm_sample)
                sample_complexity_frac = sample_complexity_frac + num_frac_arm_sample
                if frac_current_reward>frac_best_reward:
                    # update
                    frac_best_reward = frac_current_reward
                    sotred_frac_arm = copy.deepcopy(current_arm)
            # return at any point if the sample complexity too high
            if (sample_complexity_full + sample_complexity_frac)>=T:
                if sample_complexity_full<=sample_complexity_full_cap:
                    return stored_arm[0], arm_ind, arm_stream.regret_counter
                else:
                    return sotred_frac_arm, arm_ind, arm_stream.regret_counter
        
        if verbose:
            print('\rProcessing arm number '+str(current_ind+1)+'..', end="")
                
    remain_trial_nums = T - sample_complexity_frac - sample_complexity_full
    if sample_complexity_full<=sample_complexity_full_cap:
        # commit to the full stream arm
        arm_stream.regret_eval(stored_arm[0].p, remain_trial_nums)
        
        return stored_arm[0], arm_ind, arm_stream.regret_counter
    else:
        # commit to the fractional stream arm
        arm_stream.regret_eval(sotred_frac_arm.p, remain_trial_nums)
    
        return sotred_frac_arm, arm_ind, arm_stream.regret_counter