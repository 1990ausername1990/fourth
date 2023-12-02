"""
Student Number: 4949736
Name: Salim Dridi
NetID: sd727

Note: ChatGPT was used for all functions. A combination of StackOverFlow and GitHub was used to attempt to fix the assignment.
"""

#!/usr/bin/env python3
import os
import math
import matplotlib 
import pickle
import numpy as np
import scipy as sp
from scipy.linalg import lu_factor, lu_solve
import scipy.stats as stats
import scipy.special
import mnist
from tqdm import tqdm
matplotlib.use('agg')
from matplotlib import pyplot
from matplotlib import animation
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch.nn as nn
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal

## you may wish to import other things like torch.nn

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

### hyperparameter settings and other constants

gamma = 0.01
alpha = 0.1
beta = 0.9
B = 32
num_epochs = 10

### end hyperparameter settings

def load_MNIST_dataset_with_validation_split():
    PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
    try:
        dataset = pickle.load(open(PICKLE_FILE, 'rb'))
    except:
        # load the MNIST dataset
        mnist_data = mnist.MNIST(mnist_data_directory, return_type="numpy", gz=True)
        Xs_tr, Lbls_tr = mnist_data.load_training();
        Xs_tr = Xs_tr.transpose() / 255.0
        Ys_tr = numpy.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
        # shuffle the training data
        numpy.random.seed(8675309)
        perm = numpy.random.permutation(60000)
        Xs_tr = numpy.ascontiguousarray(Xs_tr[:,perm])
        Ys_tr = numpy.ascontiguousarray(Ys_tr[:,perm])
        # extract out a validation set
        Xs_va = Xs_tr[:,50000:60000]
        Ys_va = Ys_tr[:,50000:60000]
        Xs_tr = Xs_tr[:,0:50000]
        Ys_tr = Ys_tr[:,0:50000]
        # load test data
        Xs_te, Lbls_te = mnist_data.load_testing();
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = numpy.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = numpy.ascontiguousarray(Xs_te)
        Ys_te = numpy.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_va, Ys_va, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset


# compute the cumulative distribution function of a standard Gaussian random variable
def gaussian_cdf(u):
    return 0.5*(1.0 + torch.special.erf(u/math.sqrt(2.0)))

# compute the probability mass function of a standard Gaussian random variable
def gaussian_pmf(u):
    return torch.exp(-u**2/2.0)/math.sqrt(2.0*math.pi)


# compute the Gaussian RBF kernel matrix for a vector of data points (in PyTorch)
#
# Xs        points at which to compute the kernel (size: d x m)
# Zs        other points at which to compute the kernel (size: d x n)
# gamma     gamma parameter for the RBF kernel
# returns   an (m x n) matrix Sigma where Sigma[i,j] = K(Xs[:,i], Zs[:,j])
def rbf_kernel_matrix(Xs, Zs, gamma):
    # Xs: points at which to compute the kernel (size: d x m)
    # Zs: other points at which to compute the kernel (size: d x n)
    # gamma: gamma parameter for the RBF kernel
    
    # Calculate squared Euclidean distances between each pair of points
    dist_squared = torch.sum((Xs.unsqueeze(2) - Zs.unsqueeze(1))**2, dim=0)
    
    # Calculate the RBF kernel matrix using the Gaussian RBF kernel formula
    kernel_matrix = torch.exp(-gamma * dist_squared)
    
    return kernel_matrix

def gp_prediction(Xs, Ys, gamma, sigma2_noise):
    # Compute RBF kernel matrix for training data
    K = rbf_kernel_matrix(Xs, Xs, gamma)
    
    # Add noise to the diagonal for numerical stability
    K += sigma2_noise * torch.eye(len(Xs))
    
    # Compute the inverse of the kernel matrix
    K_inv = torch.inverse(K)
    
    # Define a nested function to compute the mean and variance for a test point
    def prediction_mean_and_variance(Xtest):
        # Compute the kernel vector between test point and training points
        k_vector = rbf_kernel_matrix(Xtest.unsqueeze(1), Xs, gamma)
        
        # Compute the mean and variance of the predicted distribution
        mean = torch.matmul(k_vector, torch.matmul(K_inv, Ys))
        variance = 1.0 / gamma + rbf_kernel_matrix(Xtest.unsqueeze(1), Xtest.unsqueeze(1), gamma) - torch.matmul(k_vector, torch.matmul(K_inv, k_vector.t()))
        
        return (mean.item(), variance.item())
    
    # Return the nested function
    return prediction_mean_and_variance


# compute the probability of improvement (PI) acquisition function
# Ybest     value at best "y"
# mean      mean of prediction
# stdev     standard deviation of prediction (the square root of the variance)
# returns   PI acquisition function
def pi_acquisition(Ybest, mean, stdev, kappa=1.96):
    # Ensure standard deviation is not zero to avoid division by zero
    if stdev == 0:
        return 0.0
    # Calculate z-score
    z = (mean - Ybest) / stdev

    # Calculate probability of improvement
    pi = stats.norm.cdf(z, loc=0, scale=1)

    return pi
# compute the expected improvement (EI) acquisition function
# stdev     standard deviation of prediction
# returns   EI acquisition function

# gradient descent to do the inner optimization step of Bayesian optimization
# objective     the objective function to minimize, as a function that takes a torch tensor and returns an expression
# x0            initial value to assign to variable (torch tensor)
# alpha         learning rate/step size
# num_iters     number of iterations of gradient descent
# returns     (obj_min, x_min), where
#       obj_min     the value of the objective after running iterations of gradient descent
#       x_min       the value of x after running iterations of gradient descent
def gradient_descent(objective, x0, alpha, num_iters):
    x = x0.detach().clone()  # create a fresh copy of x0
    x.requires_grad = True   # make it a target for differentiation
    optimizer = torch.optim.SGD([x], lr=alpha)
    
    for it in range(num_iters):
        optimizer.zero_grad()
        f = objective(x)
        f.backward()
        optimizer.step()

    x.requires_grad = False  # make x no longer require gradients
    return float(f.item()), x

# run Bayesian optimization to minimize an objective
# objective     objective function; takes a torch tensor, returns a python float scalar
# d             dimension to optimize over
# gamma         gamma to use for RBF hyper-hyperparameter
# sigma2_noise  additive Gaussian noise parameter for Gaussian Process
# acquisition   acquisition function to use (e.g. ei_acquisition)
# random_x      function that returns a random sample of the parameter we're optimizing over (a torch tensor, e.g. for use in warmup)
# gd_nruns      number of random initializations we should use for gradient descent for the inner optimization step
# gd_alpha      learning rate for gradient descent
# gd_niters     number of iterations for gradient descent
# n_warmup      number of initial warmup evaluations of the objective to use
# num_iters     number of outer iterations of Bayes optimization to run (including warmup)
# returns       tuple of (y_best, x_best, Ys, Xs), where
#   y_best          objective value of best point found
#   x_best          best point found
#   Ys              vector of objective values for all points searched (size: num_iters)
#   Xs              matrix of all points searched (size: d x num_iters)

def rbf_kernel(x1, x2, gamma):
    diff = x1 - x2
    return torch.exp(-gamma * torch.sum(diff ** 2))


def bayes_opt(objective, d, gamma, sigma2_noise, acquisition, random_x, gd_nruns, gd_alpha, gd_niters, n_warmup, num_iters):
    Xs = torch.zeros((d, num_iters))
    Ys = torch.zeros(num_iters)

    # Warm-up phase
    for i in range(n_warmup):
        x_rand = random_x()
        y_rand = objective(x_rand)
        Xs[:, i] = x_rand.flatten()
        Ys[i] = y_rand

    # Bayesian optimization loop
    for i in range(n_warmup, num_iters):
        # Build Gaussian Process model
        # Placeholder for the inverse of the covariance matrix

        def objective_model(x):            
             K = rbf_kernel(Xs[:, :i], x.unsqueeze(1), gamma)
             K_s = rbf_kernel(Xs[:, :i], x.unsqueeze(1), gamma)

             # Compute the predictive mean
             lu, pivots = torch.linalg.lu(K + sigma2_noise * torch.eye(len(Ys[:i])))
             mean = torch.matmul(K_s, torch.linalg.lu_solve(Ys[:i].unsqueeze(1), lu, pivots=pivots)[0])

             return mean.squeeze()



        # Define acquisition function
        def acquisition_function(x):
            mean, var = objective_model(x)
            std_dev = torch.sqrt(var)
            return acquisition(Ys.max(), mean, std_dev)

        # Perform inner optimization using gradient descent
        best_x = None
        best_y = float('inf')

        for _ in range(gd_nruns):
            x_init = random_x()
            obj_min, x_min = gradient_descent(acquisition_function, x_init, gd_alpha, gd_niters)
            
            if obj_min < best_y:
                best_y = obj_min
                best_x = x_min.detach().clone()

        # Evaluate true objective at the selected point
        true_y = objective(best_x)
        
        # Update observations
        Xs[:, i] = best_x.flatten()
        Ys[i] = true_y

    # Return the best result
    best_idx = torch.argmin(Ys)
    y_best = Ys[best_idx]
    x_best = Xs[:, best_idx]

    return float(y_best), x_best, Ys, Xs

# a one-dimensional test objective function on which to run Bayesian optimization
def test_objective(x):
    assert isinstance(x, torch.Tensor)
    assert x.shape == (1,)
    x = x.item() # convert to a python float
    return (math.cos(8.0*x) - 0.3 + (x-0.5)**2)



# compute the gradient of the multinomial logistic regression objective, with regularization (SIMILAR TO PROGRAMMING ASSIGNMENT 2)
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# ii        the list/vector of indexes of the training example to compute the gradient with respect to
# gamma     L2 regularization constant
# W         parameters        (c * d)
# returns   the average gradient of the regularized loss of the examples in vector ii with respect to the model parameters
def multinomial_logreg_batch_grad(Xs, Ys, ii, gamma, W):
    # Compute the logits
    logits = np.dot(W, Xs[:, ii])
    
    # Compute the softmax probabilities
    softmax_probs = scipy.special.softmax(logits, axis=0)
    
    # Compute the gradient of the regularized loss
    gradient = np.dot(softmax_probs - Ys[:, ii], Xs[:, ii].T) / len(ii) + gamma * W
    
    return gradient

 



# compute the error of the classifier (SAME AS PROGRAMMING ASSIGNMENT 3)
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# returns   the model error as a percentage of incorrect labels
def multinomial_logreg_error(Xs, Ys, W):
    predictions = np.argmax(np.dot(W, Xs), axis=0)
    true_labels = np.argmax(Ys, axis=0)
    error = np.mean(predictions != true_labels)
    return error


# compute the cross-entropy loss of the classifier (SAME AS PROGRAMMING ASSIGNMENT 3)
# returns   the model cross-entropy loss
def multinomial_logreg_loss(Xs, Ys, gamma, W):
    logits = np.dot(W, Xs)
    
    # Compute cross-entropy loss
    loss = -np.sum(np.log(softmax_probs) * Ys) / Xs.shape[1]
    
    # Add L2 regularization term
    reg_term = (gamma / 2) * np.linalg.norm(W, "fro")**2
    
    return loss + reg_term

# SGD + Momentum: add momentum to the previous algorithm
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# returns         the final model, after training
def sgd_mss_with_momentum(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs):
    (d, n) = Xs.shape
    V = np.zeros(W0.shape)
    W = W0
    niter = 0
    
    print("Running minibatch sequential-scan SGD with momentum")
    
    for it in tqdm(range(num_epochs)):
        for ibatch in range(int(n/B)):
            niter += 1
            ii = range(ibatch * B, (ibatch + 1) * B)
            
            # Calculate the gradient using the minibatch
            gradient = multinomial_logreg_batch_grad(Xs, Ys, ii, gamma, W)
            
            # Update momentum
            V = beta * V - alpha * gradient
            
            # Update parameters
            W = W + V
            
    return W


# produce a function that runs SGD+Momentum on the MNIST dataset, initializing the weights to zero
# mnist_dataset         the MNIST dataset, as returned by load_MNIST_dataset_with_validation_split
# num_epochs            number of epochs to run for
# B                     the batch size
# returns               a function that takes parameters
#   params                  a numpy vector of shape (3,) with entries that determine the hyperparameters, where
#       gamma = 10^(-8 * params[0])
#       alpha = 0.5*params[1]
#       beta = params[2]
#                       and returns (the validation error of the final trained model after all the epochs) minus 0.9.
#                       if training diverged (i.e. any of the weights are non-finite) then return 0.1, which corresponds to an error of 1.
def mnist_sgd_mss_with_momentum(mnist_dataset, num_epochs, B):
    X_train, Y_train, X_val, Y_val = mnist_dataset

    def train(params):
        # Unpack hyperparameters
        gamma = 10**(-8 * params[0])
        alpha = 0.5 * params[1]
        beta = params[2]

        # Initialize weights to zero
        W0 = np.zeros((Y_train.shape[0], X_train.shape[0]))

        # Run SGD with Momentum
        W = sgd_mss_with_momentum(X_train, Y_train, gamma, W0, alpha, beta, B, num_epochs)

        # Check for divergence
        if not np.all(np.isfinite(W)):
            return 0.1  # Return a high error for diverged training

        # Evaluate validation error
        val_error = multinomial_logreg_error(X_val, Y_val, W)

        return val_error - 0.9  # Minus 0.9 to make the task a minimization problem

    return train



# produce an animation of the predictions made by the Gaussian process in the course of 1-d Bayesian optimization
# objective     objective function
# acq           acquisition function
# Ys            vector of objective values for all points searched (size: num_iters)
# Xs            matrix of all points searched (size: d x num_iters)
# xs_eval       torch vector of xs at which to evaluate the mean and variance of the prediction at each step of the algorithm
# filename      path at which to store .mp4 output file
def animate_predictions(objective, acq, gamma, sigma2_noise, Ys, Xs, xs_eval, filename):
    mean_eval = []
    variance_eval = []
    acq_eval = []
    acq_Xnext = []
    for it in range(len(Ys)):
        print("rendering frame %i" % it)
        Xsi = Xs[:, 0:(it+1)]
        Ysi = Ys[0:(it+1)]
        ybest = Ysi.min()
        gp_pred = gp_prediction(Xsi, Ysi, gamma, sigma2_noise)
        pred_means = []
        pred_variances = []
        pred_acqs = []
        for x_eval in xs_eval:
            XE = x_eval.reshape(1)
            (pred_mean, pred_variance) = gp_pred(XE)
            pred_means.append(float(pred_mean))
            pred_variances.append(float(pred_variance))
            pred_acqs.append(float(acq(ybest, pred_mean, math.sqrt(pred_variance))))
        mean_eval.append(torch.Tensor(pred_means))
        variance_eval.append(torch.Tensor(pred_variances))
        acq_eval.append(torch.Tensor(pred_acqs))
        if it + 1 != len(Ys):
            XE = Xs[0,it+1].reshape(1)
            acq_Xnext.append(float(acq(ybest, pred_mean, math.sqrt(pred_variance))))

    fig = pyplot.figure()
    fig.tight_layout()
    ax = fig.gca()
    ax2 = ax.twinx()

    def animate(i):
        ax.clear()
        ax2.clear()
        ax.set_xlabel("parameter")
        ax.set_ylabel("objective")
        ax2.set_ylabel("acquisiton fxn")
        ax.set_title("Bayes Opt After %d Steps" % (i+1))
        l1 = ax.fill_between(xs_eval, mean_eval[i] + 2.0*torch.sqrt(variance_eval[i]), mean_eval[i] - 2.0*torch.sqrt(variance_eval[i]), color="#eaf1f7")
        l2, = ax.plot(xs_eval, [objective(x.reshape(1)) for x in xs_eval])
        l3, = ax.plot(xs_eval, mean_eval[i], color="r")
        l4 = ax.scatter(Xs[0,0:(i+1)], Ys[0:(i+1)])
        l5, = ax2.plot(xs_eval, acq_eval[i], color="g", ls=":")
        ax.legend([l2, l3, l5], ["objective", "mean", "acquisition"], loc="upper right")
        if i + 1 == len(Ys):
            return l1, l2, l3, l4, l5
        else:
            l6 = ax2.scatter([Xs[0,i+1]], [acq_Xnext[i]], color="g")
            return l1, l2, l3, l4, l5, l6


    ani = animation.FuncAnimation(fig, animate, frames=range(len(Ys)), interval=600, repeat_delay=1000)

    ani.save(filename)

    return -(6 * x - 2)**2 * torch.sin(12 * x - 4)

def test_random_x():
    return 1.5 * torch.rand(1) - 0.25


def predict_objective(Xs_plot, Xs, Ys, gamma, sigma2_noise):
    K = rbf_kernel(Xs_plot.unsqueeze(1), Xs.unsqueeze(1), gamma)
    K_s = rbf_kernel(Xs_plot.unsqueeze(1), Xs.unsqueeze(1), gamma)

    mean = torch.matmul(K_s, torch.lu_solve(Ys.unsqueeze(1), K + sigma2_noise * torch.eye(len(Ys)))[0])

    # Compute the predictive variance
    stdev = torch.sqrt(torch.diag(rbf_kernel(Xs_plot.unsqueeze(1), Xs_plot.unsqueeze(1), gamma) -
                                  torch.matmul(K_s, torch.lu_solve(K_s.T, K + sigma2_noise * torch.eye(len(Ys)))[0])))

    return mean.squeeze(), stdev

if __name__ == "__main__":
    y_best, x_best, Ys, Xs = bayes_opt(test_objective, 1, 10.0, 0.001, pi_acquisition, test_random_x, 20, 0.01, 20, 3, 20)
    trained_weights = mnist_sgd_mss_with_momentum(mnist_dataset, num_epochs, B)
    validation_error = multinomial_logreg_error(X_val, Y_val, trained_weights)

    print("Best y:", y_best)
    print("Best x:", x_best)
    print("All observed Ys:", Ys)
    print("All observed Xs:", Xs)

    Xs_plot = torch.linspace(-0.5, 1.5, steps=256)
    animate_predictions(test_objective, ei_acquisition, 10.0, 0.001, Ys, Xs, Xs_plot, "bayes_opt_ei.gif")
    
    print("Validation Error:", validation_error)
