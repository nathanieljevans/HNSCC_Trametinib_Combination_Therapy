## Orginially taken from: https://pyro.ai/examples/gp.html

import os
import matplotlib.pyplot as plt
import torch

import pyro
from pyro.infer.mcmc.hmc import HMC
import pyro.contrib.gp as gp
import pyro.distributions as dist
from pyro.infer.mcmc import MCMC


from mpl_toolkits.mplot3d import Axes3D
import numpy as np

#smoke_test = ('CI' in os.environ)  # ignore; used to check code integrity in the Pyro repo
#assert pyro.__version__.startswith('1.3.0')
pyro.enable_validation(True)       # can help with debugging
pyro.set_rng_seed(0)


class DrugCombinationGP: 
    
    def __init__(self):
        pass

    def fit(self, X, y, num_steps = 2500, learning_rate=0.005, plot_loss=True, verbose=True): 
        '''
        X should be a 2D torch tensor ~ drug A,B concentrations 
        y should be a 1d torch tensor ~ measured cell viability ~[0,1]
        '''
        
        self.X = X
        self.y = y

        kernel = gp.kernels.RBF(input_dim=2, variance=torch.tensor([5.]),
                        lengthscale=torch.tensor([10.]))
        gpr = gp.models.GPRegression(X, y, kernel, noise=torch.tensor(1.))

        # note that our priors have support on the positive reals
        gpr.kernel.lengthscale = pyro.nn.PyroSample(dist.LogNormal(0.5, 1.5))
        gpr.kernel.variance = pyro.nn.PyroSample(dist.LogNormal(1, 3.0))

        optimizer = torch.optim.Adam(gpr.parameters(), lr=learning_rate)
        loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        losses = []
        for i in range(num_steps):
            if verbose: print(f'progress: {i/num_steps*100:.1f}%', end='\r')
            optimizer.zero_grad()
            loss = loss_fn(gpr.model, gpr.guide)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if plot_loss: self.plot_losses(losses)

        self.gpr = gpr

    def sample(self, X, n, include_liklihood=True): 
        '''
        X (N, 2) Numpy array
        n <int> number of samples to return
        include_liklihood <bool> whether to add liklihood variance to prediction, if False, returns theta. True makes this the posterior predictive. False returns posterior on f/theta

        returns 
        y (N,) Numpy array
        '''
        with torch.no_grad(): 
            mean, var = self.gpr(torch.FloatTensor(X), full_cov=False, noiseless=~include_liklihood)
            cov = torch.diag(var)
            mvnorm = dist.MultivariateNormal(loc=mean, covariance_matrix=cov)
            samples = [mvnorm.sample().detach().numpy().ravel() for i in range(n)]
            
        return samples

    def plot_losses(self, losses): 

        plt.figure()
        plt.plot(losses)
        plt.title('training loss')
        plt.xlabel('steps')
        plt.ylabel('loss')
        plt.show()

    def plot_results(self, figsize=(10,10), plot_mean=True, alpha=0.3, X_range = (-2., 7., 0.1), savepath=None): 

        xv, yv = torch.meshgrid([torch.arange(*X_range), torch.arange(*X_range)])

        X1 = xv.reshape([-1,1])
        X2 = yv.reshape([-1,1])
        XX = torch.cat((X1,X2), 1)

        with torch.no_grad():
            mean, cov = self.gpr(XX, full_cov=True, noiseless=False)

        zv_mean = mean.reshape(xv.size())
        
        zv_mean = zv_mean.detach().numpy()
        xv = xv.detach().numpy()
        yv = yv.detach().numpy()

        # plot Results Expectation 
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        if plot_mean:
            ax.plot_surface(xv, yv, zv_mean, alpha=alpha)

        # plot two sigma uncertainty
        sd = cov.diag().sqrt().detach().numpy()  # standard deviation at each input point x

        sd_zv_upper = zv_mean + 2. * sd.reshape(xv.shape)
        ax.plot_surface(xv, yv, sd_zv_upper, alpha=alpha, color='c')

        sd_zv_lower = zv_mean - 2.*sd.reshape(xv.shape)
        ax.plot_surface(xv, yv, sd_zv_lower, alpha=alpha, color='c')

        ### Plot Obs 
        xx = self.X[:,0].detach().numpy()
        yy = self.X[:,1].detach().numpy()
        zz = self.y.detach().numpy()

        ax.scatter(xs=xx, ys=yy, zs=zz, c='r', label='obs')

        plt.legend()

        if savepath is not None: 
            plt.savefig(savepath + '/fitted_GP.png')
        else: 
            plt.show()

        return ax 


if __name__ == '__main__': 

    N = 500
    X = dist.Uniform(0.0, 5.0).sample(sample_shape=(N,2))
    y = 0.5 * torch.sin(X[:,0]) + 0.5 * torch.sin(X[:,1])  +  dist.Normal(0.0, 0.2).sample(sample_shape=(N,))

    model = DrugCombinationGP()
    model.fit(X,y, num_steps=500)
    model.plot_results()

    plt.figure()
    xxx = np.linspace(-3,8,200).reshape(-1,1)
    Xnew = np.concatenate((xxx, np.zeros((200,1))), axis=1)

    samples = model.sample(Xnew, n=100)
    for yyy in samples:
        plt.plot(xxx, yyy, 'r-', alpha=0.05)

    plt.show()
   