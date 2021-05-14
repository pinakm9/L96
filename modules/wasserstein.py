import tensorflow as tf
import numpy as np

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

def compute_cost_matrix(x, y, p=2):
    m, n = len(x), len(y)
    c = np.ones((n, m))
    for i in range(n):
        for j in range(m):
            c[i][j] = (np.abs(x[i] - y[j])**p).sum()
    return c

def lse(x):
    return np.log(np.exp(x).sum())

def sinkhorn_div_tf(x, y, alpha=None, beta=None, epsilon=0.01, num_iters=50, p=2):
    c = cost_matrix(x, y, p=p)
    n, m = x.shape[0], y.shape[0]
    if alpha is None:
        alpha = tf.fill((n), 1./n)

    if beta is None:
        beta = tf.fill((n), 1./n)

    log_alpha = tf.expand_dims(tf.math.log(alpha), 1)
    log_beta = tf.math.log(beta)

    f, g = 0. * alpha, 0. * beta
    for _ in range(num_iters):
        f = - epsilon * tf.reduce_logsumexp(log_beta + (g - c) / epsilon, axis=1)
        g = - epsilon * tf.reduce_logsumexp(log_alpha + (tf.expand_dims(f, 1) - c) / epsilon, axis=0)

    OT_alpha_beta = tf.reduce_sum(f * alpha) + tf.reduce_sum(g * beta)
    
    c = cost_matrix(x, x, p=p)
    f = 0. * alpha
    log_alpha = tf.squeeze(log_alpha)
    for _ in range(int(num_iters/5)):
        f = 0.5 * (f - epsilon * tf.reduce_logsumexp(log_alpha + (f - c) / epsilon, axis=1) )

    c = cost_matrix(y, y, p=p)
    g = 0. * alpha
    for _ in range(int(num_iters/5)):
        g = 0.5 * (g - epsilon * tf.reduce_logsumexp(log_beta + (g - c) / epsilon, axis=1) )
    
    return OT_alpha_beta - tf.reduce_sum(f * alpha) - tf.reduce_sum(g * beta)


def sinkhorn_div(x, y, alpha=None, beta=None, epsilon=0.01, num_iters=50, p=2):
    c = compute_cost_matrix(x, y, p=p)
    n, m = len(x), len(y)
    if alpha is None:
        alpha = np.ones(n) / n

    if beta is None:
        beta = np.ones(m) / m 

    log_alpha = np.log(alpha)
    log_beta = np.log(beta)

    f, g = 0. * alpha, 0. * beta
    for _ in range(num_iters):
        for i in range(n):
            f[i] = - epsilon * lse(log_beta + (g - c[i, :]) / epsilon)
        for j in range(m):
            g[j] = - epsilon * lse(log_alpha + (f - c[:, j]) / epsilon)

    OT_alpha_beta = np.dot(f, alpha) + np.dot(g, beta)

    c = compute_cost_matrix(x, x, p=p)
    f = 0. * alpha
    for _ in range(int(num_iters/5)):
        for i in range(n):
            f[i] = 0.5 * (f[i] - epsilon * lse(log_alpha + (f - c[i, :]) / epsilon) )

    c = compute_cost_matrix(y, y, p=p)
    g = 0. * alpha
    for _ in range(int(num_iters/5)):
        for i in range(m):
            g[i] = 0.5 * (g[i] - epsilon * lse(log_beta + (g - c[i, :]) / epsilon) )

    return OT_alpha_beta - np.dot(f, alpha) - np.dot(g, beta)



