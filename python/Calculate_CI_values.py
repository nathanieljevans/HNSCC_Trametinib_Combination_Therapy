import argparse 
import os

import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import statsmodels.api as sm

import pickle as pkl
import seaborn as sbn

import time

import torch

import sys 
sys.path.append('./python/')
from HillModel import * 
from LogisticModel import *
from DrugCombinationGP import *

#%matplotlib notebook


def get_args(): 
    parser = argparse.ArgumentParser(description='Calculate the CI values for the OHSU Trametinib combination HNSCC project')
    
    parser.add_argument('--ICxx', type=float, nargs=1,
                        help='IC value to use for the CI calculations')
    parser.add_argument('--out', type=str, nargs=1,
                        help='sum the integers (default: find the max)')

    return parser.parse_args()

def _get_ICxx(x, y, IC_value):  

    # Hill Regression
    model = HillModel(verbose=False)
    model.fit(10**x, y, epochs=15000, learningRate=1e-4, plot=False)

    ICxx, eq = model.get_IC(IC_value*100)
    
    # Hill Model returns ICxx in uM, Logistic returns it in logspace
    ICxx = np.log10(ICxx)
    
    return ICxx, eq

def get_CI(dataA, dataB, dataAB, IC_value, margin=0.0, num_meshgrid_samples=100, allowed_ic_delta = 0.0015): 
    
    # fit 1D (single agents) dose-response data 
    x_A = np.log10(dataA.conc.values.astype(np.float))
    y_A = dataA.cell_viab.values 
    ICxxA, eqA = _get_ICxx(x_A, y_A, IC_value=IC_value) # returned in log10(uM)

    x_B = np.log10(dataB.conc.values.astype(np.float))
    y_B = dataB.cell_viab.values
    ICxxB, eqB = _get_ICxx(x_B, y_B, IC_value=IC_value) # returned in log10 (uM)
    
    # Fit 2D (combo) dose-response data
    x = dataAB.conc.values
    Z = dataAB.cell_viab.values

    A=[]; B=[]
    for xx in x: 
        A.append( float(xx.split(';')[0]) )
        B.append( float(xx.split(';')[1]) )

    A = np.log10( np.array(A) )
    B = np.log10( np.array(B) )
    
    endog = dataAB.cell_viab.values
    exog = sm.add_constant( np.concatenate((A.reshape(-1,1),B.reshape(-1,1)), axis=1) )

    concs = [x.split(';') for x in dataAB.conc.values] 
    concs = np.array(concs)

    x_ab = torch.FloatTensor(np.log10(concs.astype(np.float)))
    y_ab = torch.FloatTensor(dataAB.cell_viab.values.astype(np.float))

    ComboAB_GP = DrugCombinationGP()
    ComboAB_GP.fit(x_ab, y_ab, num_steps=2500, learning_rate=0.005, plot_loss=False, verbose=False)

    log_concs = np.log10( concs.astype(np.float) )
    concA_range = (log_concs[:,0].min()-margin, log_concs[:,0].max()+margin, num_meshgrid_samples)
    concB_range = (log_concs[:,1].min()-margin, log_concs[:,1].max()+margin, num_meshgrid_samples)
    Av, Bv = np.meshgrid(np.linspace(*concA_range), np.linspace(*concB_range))

    Ax = Av.reshape((-1,))
    Bx = Bv.reshape((-1,))
    Xnew = torch.FloatTensor(np.concatenate((Ax.reshape(-1,1), Bx.reshape(-1,1)), 1))
    comb_samples = np.array( ComboAB_GP.sample(Xnew, n=300) )
    GP_comb_mean = comb_samples.mean(axis=0).reshape(Av.shape)

    idx_ic_comb = np.where((np.abs(GP_comb_mean - IC_value).min(axis=0)))
    comb = pd.DataFrame({'x':Av[idx_ic_comb].reshape(-1),'y':Bv[idx_ic_comb].reshape(-1),'z':GP_comb_mean[idx_ic_comb].reshape(-1)})
    comb = comb[lambda x: (x.z < IC_value + allowed_ic_delta) & (x.z > IC_value - allowed_ic_delta)]
    
    if len(comb) == 0: 
        return *([None]*7), ICxxA, ICxxB, eqA, eqB
    # isobologram 
    
    x = [10**ICxxA,0]
    y = [0, 10**ICxxB]
    
    # results 
    
    C_a = 10**np.array( comb.x.values )
    #print('range conc A:', (C_a.min(), C_a.max()))
    
    C_b = 10**np.array( comb.y.values)
    #print('range conc B:', (C_b.min(), C_b.max()))

    CI = C_a/(10**ICxxA) + C_b/(10**ICxxB)

    idx = np.where(CI == min(CI))
    Ca_min = C_a[idx]
    Cb_min = C_b[idx]
    
    min_CI = min(CI) 

    #print(f'minimum CI value [{min(CI):.2f}] found at: {drugA}={Ca_min}, {drugB}={Cb_min}]')
    idx2 = np.where(CI < 1)
    if len(idx2[0]) > 0: 
        Ca_range = C_a[idx2]
        Ca_range_lower, Ca_range_upper = min(Ca_range), max(Ca_range)

        Cb_range = C_b[idx2]
        Cb_range_lower, Cb_range_upper = min(Cb_range), max(Cb_range)
    else: 
        Ca_range_lower, Ca_range_upper = None, None
        Cb_range_lower, Cb_range_upper = None, None

    #print(f'This combination is synergistic between: {drugA}:[{min(Ca_range):.5f}-{max(Ca_range):.5f}], {drugB}:[{min(Cb_range):.5f}-{max(Cb_range):.5f}]')
    
    return min_CI, Ca_min, Cb_min, Ca_range_lower, Ca_range_upper, Cb_range_lower, Cb_range_upper, ICxxA, ICxxB, eqA, eqB
    


if __name__ == '__main__': 
    args = get_args() 
    
    print('-----'*10)
    print('processing Combination Index Values') 
    print('ICxx:', args.ICxx) 
    print('output dir:', args.out)
    print('-----'*10)
    
    
    # --------------------------------- LOAD DATA -------------------------------------------------
    trem13 = pd.read_csv('./../data/Trametinib_Data.csv')
    trem13 = trem13.assign(inhibitor='TRAMETINIB', conc=trem13['Conc(uM)'], lab_id=trem13['Cell_Line'], cell_viab=trem13['Cell_Viability'])
    trem13 = trem13[['lab_id', 'inhibitor', 'conc', 'cell_viab']].assign(dataset='trem-single13')

    combo_data = pd.read_csv('./../data/all_trem_combo_data.csv').dropna().drop('Unnamed: 0', axis=1).assign(dataset = 'trem-combo')

    data_all = pd.concat([combo_data, trem13], axis=0)
    
    res = {x:[] for x in 'lab_id,drugA,drugB,ICxx,ICxxA,ICxxB,min_CI,Ca_min,Cb_min,Ca_range_lower,Ca_range_upper,Cb_range_lower,Cb_range_upper,equality_A,equality_B'.split(',')}
    failures = []
    for lab_id in data_all.lab_id.unique():
        print('processing lab_id:', lab_id)
        if lab_id == 10250: 
            print('skipping lab_id 10250 (excluded due to batch effects)')
            continue
            
        try: 
            single_agents = data_all[~data_all.inhibitor.str.contains(';')].inhibitor.unique()
            comb_agents = data_all[data_all.inhibitor.str.contains(';')].inhibitor.unique()

            data_id = data_all[lambda x: x.lab_id == lab_id]

            for comb in comb_agents: 
                print('\tcombination:', comb)

                drugA, drugB = comb.split(';') 
                dataA = data_id[lambda x: x.inhibitor == drugA] 
                dataB = data_id[lambda x: x.inhibitor == drugB]
                dataAB = data_id[lambda x: x.inhibitor == comb]
                #print('dataA shape:', dataA.shape)
                #print('dataB shape:', dataB.shape) 
                #print('dataAB shape:', dataAB.shape)

                min_CI, Ca_min, Cb_min, Ca_range_lower, Ca_range_upper, Cb_range_lower, Cb_range_upper, ICxxA, ICxxB, eqA, eqB = get_CI(dataA, dataB, dataAB, IC_value=args.ICxx[0])
                
                #print(min_CI, Ca_min, Cb_min, Ca_range_lower, Ca_range_upper, Cb_range_lower, Cb_range_upper, ICxxA, ICxxB, eqA, eqB)

                res['lab_id'].append(lab_id)
                res['drugA'].append(drugA)
                res['drugB'].append(drugB)
                res['min_CI'].append(min_CI)
                res['Ca_min'].append(Ca_min)
                res['Cb_min'].append(Cb_min)
                res['Ca_range_lower'].append(Ca_range_lower)
                res['Ca_range_upper'].append(Ca_range_upper)
                res['Cb_range_lower'].append(Cb_range_lower)
                res['Cb_range_upper'].append(Cb_range_upper)
                res['ICxx'].append(args.ICxx[0])
                res['ICxxA'].append(ICxxA)
                res['ICxxB'].append(ICxxB)
                res['equality_A'].append(eqA)
                res['equality_B'].append(eqB)
        except: 
            failures.append((lab_id, comb) )
            #raise
            
    
    res = pd.DataFrame(res) 
    res.to_csv(args.out[0]) 
    
    print('# failures:', len(failures)) 
    print('failures:') 
    _ = [print('\t', x) for x in failures]
        
        
        
        
        
        
        
        
        
        
        
    
    