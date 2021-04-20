'''
author: Nate Evans
email: evansna@ohsu.edu
'''

import statsmodels.api as sm
import numpy as np
from matplotlib import pyplot as plt 


def calculate_ICxx(x, model, ICxx=0.5): 
    '''
    returns IC in logspace
    '''
    x2 = np.linspace(min(x), 10., int(1e5))
    yhat = model.predict(sm.add_constant(x2))
    for i,yy in enumerate(yhat): 
        if yy <= ICxx: return x2[i]
        
    print('No IC50 found, is this an inhibitor assay (monotonic decreasing?)')
    return 10

    
def fit_logistic(inhib, x, y, ICxx=0.5, plot=True):
    '''
    x <array like> concentration, should be log10 normalized 
    y <array like> cell viability, must be between 0,1

    '''
    logit = sm.GLM(y, sm.add_constant(x), family=sm.families.Binomial()) 
    glm_res = logit.fit(disp=False)

    x2 = np.arange(-5, 5, 0.001)
    yhat = glm_res.predict(sm.add_constant(x2))
    
    IC = calculate_ICxx(x, glm_res, ICxx=ICxx)

    f, ax = plt.subplots(1,1, figsize=(7,7))
    plt.title('inhibitor: %s' %(inhib))
    plt.plot(x2, yhat, 'r-', label='logistic')
    plt.plot(x, y, 'bo', label='replicates')
    plt.axvline(IC, color='g', label=f'IC{ICxx*100} [{10**IC:.4f} (uM)]')
    plt.ylim((0,1.25))
    plt.legend()
    plt.show()

    intercept,beta1 = glm_res.params
    probit_AIC, probit_BIC = glm_res.aic, glm_res.bic
    probit_Deviance = glm_res.deviance
    probit_pval = glm_res.pvalues

    print(glm_res.summary())
    
    return IC