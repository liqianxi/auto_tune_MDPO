import numpy as np 
import tensorflow as tf

data1 = np.loadtxt('mdpo_off/chk.bed')
print(data1)
log_ent_coef = np.log(0.13755673)
print(-np.mean(log_ent_coef * (data1 - 6)))

