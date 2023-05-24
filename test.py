import os
import pickle
import numpy as np
import random
import copy
import tqdm

from streamingMAB import Bernoulli_Arm, Arm_Reading_Buffer
from algorithms import uniform_exploration_algorithm, naive_eps_best_algorithm, best_among_remaining
from algorithms import log_eps_best_algorithm, loglog_eps_best_algorithm, logstar_eps_best_algorithm
from algorithms import Game_of_Arms_regret_min


# parameter
num_arms = 50000
trial_mult_factor = 1000
trial_exp_factor = 3
num_trials = trial_mult_factor*(int)(num_arms**trial_exp_factor)
arm_setting = 'clear_cut_setting'  
# arm_setting = 'mix_in_setting'



uniform_exploration_regrets = []
naive_esp_alg_regrets = []
log_eps_alg_regrets = []
loglog_eps_alg_regrets = []
logstar_eps_alg_regrets = []
goa_esp_alg_regrest = []
for i_seed in tqdm.tqdm(range(50)):
    # fix the random seed
    np.random.seed(i_seed)
    # randomly construct a set of arms
    if arm_setting == 'clear_cut_setting':
        arm_list = [Bernoulli_Arm(random_mean='gaussian') for _ in range(num_arms-1)]
        arm_list.append(Bernoulli_Arm(p=0.82))
    elif arm_setting == 'mix_in_setting':
        arm_list = [Bernoulli_Arm(random_mean='uniform') for _ in range(num_arms)]
    else:
        raise ValueError('The arm_setting parameter is not correctly setup.')
    random.shuffle(arm_list)
    # construct the buffer
    streaming_arm_buffer = Arm_Reading_Buffer(arm_set = np.array(arm_list))
    # trials for uniform exploration
    streaming_arm_buffer.reset()
    uni_best_arm, uni_arm_ind, uni_regret = uniform_exploration_algorithm(streaming_arm_buffer, 
                                                                          num_trials, 
                                                                          verbose=False)
    uniform_exploration_regrets.append(uni_regret)
    # trials for the naive eps-best arm algorithm
    streaming_arm_buffer.reset()
    eps_naive_best_arm, eps_naive_arm_ind, eps_naive_regret = naive_eps_best_algorithm(streaming_arm_buffer, 
                                                                                       num_trials, 
                                                                                       verbose=False)
    naive_esp_alg_regrets.append(eps_naive_regret)
    # trials for the log(n)-space eps-best arm algorithm
    streaming_arm_buffer.reset()
    eps_log_best_arm, eps_log_arm_ind, eps_log_regret = log_eps_best_algorithm(streaming_arm_buffer, 
                                                                               num_trials, 
                                                                               verbose=False)
    log_eps_alg_regrets.append(eps_log_regret)
    # trials for the loglog(n)-space eps-best arm algorithm
    streaming_arm_buffer.reset()
    eps_loglog_best_arm, eps_loglog_arm_ind, eps_loglog_regret = loglog_eps_best_algorithm(streaming_arm_buffer, 
                                                                                           num_trials, 
                                                                                           verbose=False) 
    loglog_eps_alg_regrets.append(eps_loglog_regret)
    # trials for the logstar(n)-space eps-best arm algorithm
    streaming_arm_buffer.reset()
    eps_logstar_best_arm, eps_logstar_arm_ind, eps_logstar_regret = logstar_eps_best_algorithm(streaming_arm_buffer, 
                                                                                               num_trials, 
                                                                                               verbose=False)
    logstar_eps_alg_regrets.append(eps_logstar_regret)
    # trials for the game-of-arm eps-best arm algorithm
    streaming_arm_buffer.reset()
    eps_goa_best_arm, eps_goa_arm_ind, eps_goa_regret = Game_of_Arms_regret_min(streaming_arm_buffer, 
                                                                                num_trials, 
                                                                                verbose=False)
    goa_esp_alg_regrest.append(eps_goa_regret)



print('The 50-trial average regret by uniform exploration is', np.mean(uniform_exploration_regrets))
print('The 50-trial average regret by the naive algorithm for eps-best arm is', np.mean(naive_esp_alg_regrets))
print('The 50-trial average regret by the log-space algorithm for eps-best arm is', np.mean(log_eps_alg_regrets))
print('The 50-trial average  regret by the loglog-space algorithm for eps-best arm is', np.mean(loglog_eps_alg_regrets))
print('The 50-trial average regret by the logstar-space algorithm for eps-best arm is', np.mean(logstar_eps_alg_regrets))
print('The 50-trial average regret by the game-of-arm algorithm for eps-best arm is', np.mean(goa_esp_alg_regrest))

rst_id = 'num_multiplier='+str(trial_mult_factor)+'_exp='+str(trial_exp_factor)
save_dir = './experiments/'+arm_setting+'/num_arms='+str(num_arms)+'/'
# create the directory if it does not exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# save the information as numpy arrays
np.save(save_dir+rst_id+'_uniform.npy', np.array(uniform_exploration_regrets))
np.save(save_dir+rst_id+'_naive_eps.npy', np.array(naive_esp_alg_regrets))
np.save(save_dir+rst_id+'_log_eps.npy', np.array(log_eps_alg_regrets))
np.save(save_dir+rst_id+'_loglog_eps.npy', np.array(loglog_eps_alg_regrets))
np.save(save_dir+rst_id+'_logstar_eps.npy', np.array(logstar_eps_alg_regrets))
np.save(save_dir+rst_id+'_goa_eps.npy', np.array(goa_esp_alg_regrest))