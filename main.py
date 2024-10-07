from model.dft import my_dft
from model.datareader import dataReader
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm

folder_path = r'Data'
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')] 
L_csv = len(csv_files)

n_partitioner = 4320
idx_2hz = 605
threshold_derivative = 0.00015
offset_inicio = 350
output_df = {"filename":[], "time_rel(sec)":[]}
M = 1000

for file_ in tqdm (range(L_csv), desc="Processing ..."):
    time_segment = []
    velocity_segment = []
    data, name = dataReader(file_, folder_path)

    name = name[5:]
    name = name[:len(name) - 4]

    time_rel = data.iloc[:,1]
    velocity = data.iloc[:,2]
    k = 0
    time = []
    vel = []

    for i in range(len(time_rel)):
        time.append(time_rel[i])
        vel.append(velocity[i])
        k += 1
        if k % int(len(velocity)/4320) == 0:
            k = 0
            time_segment.append(time)
            velocity_segment.append(vel)
            time = []
            vel = []

    potencia_evento = []
    t_average = []

    for i in range(len(time_segment)):
        t_idx = np.array(time_segment[i])
        v_idx = np.array(velocity_segment[i])
            
        dt = t_idx[2] - t_idx[1]
        fs = 1/dt
        N = len(v_idx)
        X,w = my_dft(v_idx,t_idx,M)
        f = w*fs/(2*np.pi)
        psd = (1/(fs*N))*np.abs(X)**2
        psd[2:len(psd)-1] = 2*psd[2:len(psd)-1]
        P_ave = (1/(np.max(f[idx_2hz:]) - np.min(f[idx_2hz:])))*np.abs(np.trapz(f[idx_2hz:],psd[idx_2hz:]))
        potencia_evento.append(P_ave)
        t_average.append((np.max(t_idx) + np.min(t_idx))/2)

    normalized_P = potencia_evento/(np.sum(np.array(potencia_evento)))
    cdf_p = np.cumsum(normalized_P)
    dcdf_p = np.diff(cdf_p)/(t_average[2]-t_average[1])

    for i in range(len(dcdf_p)):
        if (dcdf_p[i] > threshold_derivative):
            output_df["filename"].append(name)
            output_df["time_rel(sec)"].append(t_average[i] - offset_inicio)
            break
    #pd.DataFrame(output_df).to_csv("output_catalog.csv",index=False)
        #plt.figure(file_)
        #plt.subplot(4,1,1)
        #plt.plot(time_rel,velocity)
        #plt.subplot(4,1,2)
        #plt.plot(t_average,potencia_evento)
        #plt.subplot(4,1,3)
        #plt.plot(t_average,cdf_p)
        #plt.subplot(4,1,4)
        #plt.plot(t_average[:len(t_average)-1],dcdf_p)
        #plt.show()

pd.DataFrame(output_df).to_csv("output_catalog.csv",index=False)
   



