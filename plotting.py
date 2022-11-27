import seaborn as sns
import os
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import numpy as np






'''

col_size = 1

val3 = [[0.2],[3*10-4]] 
row_size = len(val3)
val2 = ["ee","aa"] 
val1 = ["{:02X}".format(row_size * i) for i in range(row_size)] 

print(val1,val2,val3)
fig, ax = plt.subplots() 
ax.set_axis_off() 
table = ax.table( 
    cellText = val3,  
    rowLabels = val2,  
    colLabels = val1, 
    rowColours =["palegreen"] * row_size,  
    colColours =["palegreen"] * col_size, 
    cellLoc ='center',  
    loc ='upper left')         
   
ax.set_title('matplotlib.axes.Axes.table() function Example', 
             fontweight ="bold") 
   
plt.show() 


assert 1==2'''




env_name = "Walker2d-v2."

root_path = "experiments/"+env_name

sns.set_style("whitegrid", {'axes.grid' : True,
                            'axes.edgecolor':'black'

                            })

entropy_file_name = "ent_coef.txt"
time_stamp_file = "time_stamp.txt"
ent_list = []
time_list = []
with open(entropy_file_name) as ent:
    
    ent_full_list = ent.read().splitlines()
        
with open(time_stamp_file) as time:
    time_full_list = time.read().splitlines()

for each in range(len(ent_full_list)):
    #print(ent_full_list[each])

    if ent_full_list[each]:
        ent_list.append(float(ent_full_list[each]))
        time_list.append(int(time_full_list[each]))
fig,ax = plt.subplots()
sns.lineplot(x=time_list,y=ent_list,color='r')
plt.xlabel('Timestamp', fontsize=18)
plt.ylabel('Entropy Coefficient', fontsize=16)
txt="Figure (2), the entropy converges over time and the average value is 0.2314."
plt.axhline(y = 0.2314, color = 'b', linestyle = '-',label ="0.2314")
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=10)
fig.set_size_inches(5.5, 6.5, forward=True)
plt.legend(bbox_to_anchor = (1.0, 1), loc = 'upper center')
fig.savefig("ent_coeff.png") 







all_data = {}
model_name_list = []

for model_name in os.listdir(root_path):
    all_data[model_name] = []
    
    if model_name != '.DS_Store':
        model_name_list.append(model_name)
        model_path = os.path.join(root_path, model_name,"bootstrap-2/selected")
        
        for run in os.listdir(model_path):

            if run != '.DS_Store' and 'm' not in run:
                result_file_path = os.path.join(model_path,run,"progress.csv")
                df = pd.read_csv(result_file_path)[['total timesteps','ep_rewmean']][1:]
                
                if df.shape[0] !=999:
                    print("in correct shape",df.shape,result_file_path)
                    assert 1==2
                all_data[model_name].append(df)

fig,ax = plt.subplots()


color = {"MDPO_Auto":'b',"MDPO":'r',"SAC":'yellow'}
line_list = []
for model in model_name_list:
    print(model)
    run_summary_list = []
    temp_reward = []
    time = None
    for run in all_data[model]:
        reward = [i for i in run['ep_rewmean']]
        time = run['total timesteps']
        temp_reward.append(reward)

    
    all_reward = []
    conf_up_list = []
    conf_low_list = []
    mean_list = []
    #print(len(temp_reward))
    #print(len(temp_reward[0]))
    try:
        for each_row_idx in range(len(temp_reward[0])):
            temp = [temp_reward[i][each_row_idx] for i in range(len(temp_reward))]
            #print(temp)
            mean_list.append(np.mean(temp))
            bound = st.t.interval(alpha=0.95, df=len(temp)-1, loc=np.mean(temp), scale=st.sem(temp)) 
            conf_low_list.append(bound[0])
            conf_up_list.append(bound[1])
    except Exception:
        print(temp)
        exit()

    #print(run['mean 100 episode reward'][:20])
    line = sns.lineplot(x=time,y=mean_list,color=color[model],label=model)
    line_list.append(line)
    #sns.lineplot(x=time,y=conf_low_list)
    #sns.lineplot(x=time,y=conf_up_list)
    #fig = plot.get_figure()
    ax.fill_between(time, conf_low_list, conf_up_list, color=color[model], alpha=.15)
for each in range(len(line_list)):
    line_list[each].set_label(model_name_list[each])
ax.legend()
txt = "Figure (1)"
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=11)
fig.set_size_inches(5.5, 6.5, forward=True)
fig.savefig("compare.png") 
    #assert 1==2
