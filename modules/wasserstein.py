import tensorflow as tf
import numpy as np
from itertools import permutations

from tensorflow.python.ops.math_ops import reduce_logsumexp

def sample_integers(n, shape):
    sample = tf.random_uniform(shape, minval=0, maxval=tf.cast(n, 'float32'))
    sample = tf.cast(sample, 'int32')
    return sample

def resample_rows_per_column(x):
    """Permute all rows for each column independently."""
    n_batch = tf.shape(x)[0]
    n_dim = tf.shape(x)[1]
    row_indices = sample_integers(n_batch, (n_batch * n_dim,))
    col_indices = tf.tile(tf.range(n_dim), [n_batch])
    indices = tf.transpose(tf.stack([row_indices, col_indices]))
    x_perm = tf.gather_nd(x, indices)
    x_perm = tf.reshape(x_perm, (n_batch, n_dim))
    return x_perm

def z_score(x):
    """
    Z_scores each dimension of the data (across axis 0)
    """
    #mean_vals = tf.reduce_mean(x,axis=0,keep_dims=True)
    #std_vals = tf.sqrt(tf.reduce_var(x,axis=0,keep_dims=True))
    mean_vals,var_vals = tf.nn.moments(x,axes=[0],keep_dims=True)
    std_vals = tf.sqrt(var_vals)
    x_normalized = (x - mean_vals)/std_vals
    return x_normalized

def cost_matrix(x, y, p=2):
    "Returns the cost matrix C_{ij}=|x_i - y_j|^p"
    x_col = tf.expand_dims(x,1)
    y_lin = tf.expand_dims(y,0)
    c = tf.reduce_sum((tf.abs(x_col-y_lin))**p,axis=2)
    return c

def sinkhorn_loss(x, y, x_weights=None, y_weights=None, epsilon=0.01, num_iters=200, p=2):
    """
    Description:
        Given two emprical measures with locations x and y
        outputs an approximation of the OT cost with regularization parameter epsilon
        num_iter is the max. number of steps in sinkhorn loop
    
    Args:
        x,y:  The input sets representing the empirical measures.  Each are a tensor of shape (n,D)
        x_weights, y_weights: weights for ensembles x and y
        epsilon:  The entropy weighting factor in the sinkhorn distance, epsilon -> 0 gets closer to the true wasserstein distance
        num_iters:  The number of iterations in the sinkhorn algorithm, more iterations yields a more accurate estimate
        p: p value used to define the cost in Wasserstein distance
    
    Returns:
        The optimal cost or the (Wasserstein distance) ** p
    """
    # The Sinkhorn algorithm takes as input three variables :
    C = cost_matrix(x, y, p=p)  # Wasserstein cost function
    
    # both marginals are fixed with equal weights
    if x_weights is None:
        n = x.shape[0]
        x_weights = tf.constant(1.0/n,shape=[n])

    if y_weights is None:
        n = y.shape[0]
        y_weights = tf.constant(1.0/n,shape=[n])
    # Elementary operations
    def M(u,v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + tf.expand_dims(u,1) + tf.expand_dims(v,0) )/epsilon
    def lse(A):
        return tf.reduce_logsumexp(A,axis=1,keepdims=True)
    
    log_x_w = tf.math.log(x_weights)
    log_y_w = tf.math.log(y_weights)
    # Actual Sinkhorn loop
    u, v = 0. * x_weights, 0. * y_weights
    for _ in range(num_iters):
        u = epsilon * (log_x_w - tf.squeeze(lse(M(u, v)) )  ) + u
        v = epsilon * (log_y_w - tf.squeeze( lse(tf.transpose(M(u, v))) ) ) + v
    
    #u_final,v_final = u,v
    pi = tf.exp(M(u, v))
    cost = tf.reduce_sum(pi*C)
    return cost

def sinkhorn_div(x, y, alpha=None, beta=None, epsilon=0.01, num_iters=50, p=2):
    OT_alpha_beta = sinkhorn_loss(x, y, alpha, beta, epsilon, num_iters, p)
    OT_alpha_alpha = sinkhorn_loss(x, x, alpha, alpha, epsilon, num_iters, p)
    OT_beta_beta = sinkhorn_loss(y, y, beta, beta, epsilon, num_iters, p)
    return OT_alpha_beta - 0.5 * OT_alpha_alpha - 0.5 * OT_beta_beta