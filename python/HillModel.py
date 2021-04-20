'''
The Hill model is defined as: 

$$ F(c, E_{max}, E_0, EC_{50}, H) = E_0 + \frac{E_{max} - E_0}{1 + (\frac{EC_{50}}{C})^H} $$

Where concentration, $c$ is in uM, and is *not* in logspace. 

author: Nate Evans
email: evansna@ohsu.edu
'''

import torch
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt

class HillModel(torch.nn.Module): 
    def __init__(self, verbose=True): 
        super(HillModel, self).__init__()
        self.Emax = torch.nn.Parameter(torch.Tensor([1.])) 
        self.Emax.data.normal_(0.0, 0.25)
        if verbose: print('Emax init', self.Emax.detach().numpy())
        
        self.E0 = torch.nn.Parameter(torch.FloatTensor([1.]))
        self.E0.data.normal_(1., 0.25) 
        if verbose: print('E0 init', self.E0.detach().numpy())
        
        self.EC50 = torch.nn.Parameter(torch.FloatTensor([0.01]))
        #self.EC50.data.uniform_(0., 10.)
        
        self.H = torch.nn.Parameter(torch.FloatTensor([1.]))
        self.H.data.normal_(1., 0.25)
        if verbose: print('H init', self.H.detach().numpy())
        
        self.fitX = 0
        self.fitY = 0
        
    def forward(self, x): 
        
        yhat = self.E0 + (self.Emax - self.E0) / (1 + (self.EC50/x)**self.H)
        
        return yhat
    
    def fit(self, X,Y , epochs=100, learningRate=1e-3, plot=True): 
        '''
        X, conc, should not be in logspace 
        Y, cell viability 
        numpy arrays are okay 
        '''
        self.fitX = X
        self.fitY = Y
        
        X = torch.FloatTensor(X); Y = torch.FloatTensor(Y)
        
        criterion = torch.nn.MSELoss() 
        optimizer = torch.optim.Adam(self.parameters(), lr=learningRate)

        loss_ = []
        for i in range(epochs):
            optimizer.zero_grad()
            Yhat = self.forward(X)
            loss = criterion(Yhat, Y)
            loss.backward()
            optimizer.step()
            loss_.append(loss)
            #print('epoch {}, loss {}'.format(i, loss.item()))  
            
        if plot: 
            plt.figure()
            plt.plot(np.log10(loss_))
            plt.ylabel('log10 loss')
            plt.xlabel('epoch')
            plt.show() 
            
    def predict(self, X): 
        
        self.eval() 
        X = torch.FloatTensor(X)
        return self.forward(X).detach().numpy()
    
    def plot_fit(self, X=np.logspace(-7,1), ICxx=50): 
        
        Y = self.predict(X)
        
        plt.figure()
        plt.plot(np.log10(self.fitX), self.fitY, 'r.', label='obs')
        plt.plot(np.log10(X), Y, 'b--', label='fit')
        
        try: 
            IC = self.get_IC(ICxx)
            plt.axvline(np.log10(IC), c='g', label=f'log IC{ICxx} [{IC:.3f}]')
        except: 
            print('here')
            raise
        
        plt.xlim(-8,2)
        plt.xlabel('log10 Conc [uM]')
        plt.ylim(-0.1,1.2)
        plt.ylabel('cell viability (prob)')
        plt.title('Regression Fit')
        plt.show()
        
    def summary(self): 
        
        print( '| PARAMETER | VALUE |')
        print( '|-----------|-------|')
        print(f'|  Emax     |{self.Emax.detach().numpy()[0]:.5f}|')
        print(f'|  E0       |{self.E0.detach().numpy()[0]:.5f}|')
        print(f'|  EC50     |{self.EC50.detach().numpy()[0]:.5f}|')
        print(f'|  H        |{self.H.detach().numpy()[0]:.5f}|')
        
    def get_IC(self, XX, diff=0.05): 
        '''
        ICXX; XX should be percentage [0,100]
        '''
        XX = XX / 100
        
        x = np.logspace(-7,2,1000)
        y = self.predict(x)
        
        a = np.abs(y - XX)
        
        if min(a) > diff: 
            return 10
            #print('min a', min(a)) 
            #raise ValueError('ICxx can not be calculated') 
        else: 
            ICxx = x[a == min(a)][0]
            return ICxx         
        