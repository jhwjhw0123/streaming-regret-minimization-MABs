# streaming-regret-minimization-MABs
Codes for the experiments in paper 'Tight Regret Bounds for Single-pass Streaming Multi-armed Bandits.'

Run test.py or test.ipynb to get test the performances of the implemented algorithms.

Change the variable `num_arms` to adjust the number of arms ($K$); change `trial_mult_factor` ($\alpha$) and `trial_exp_factor` ($\beta$) to change the relationship between $K$ and $T$, i.e. $T=\alpha \cdot K^{\beta}$.

Change the `arm_setting` variable between `clear_cut_setting` and `mix_in_setting` to control the setting of the stream, i.e. one arm with much higher mean reward vs. arms with similar rewards.
