import sys
import tensorflow as tf              #version '1.12.0'
import tensorflow_probability as tfp #version '0.5.0'
import os
import numpy as np
import itertools
from collections import defaultdict
from functools import reduce
import time
import pickle as pkl

tfd = tfp.distributions
tfb = tfp.bijectors



"""
This code runs STRUCTURE-style inference for the DIACL data set for some value of K (clusters).
We fix lambda, the component-level prior over feature distributions, at 1.
We infer alpha, the language-level prior over components using an improper prior (~ U(0,inf)).
"""



langs_to_exclude = ['Avestan','Classical_Greek','Gothic','Hittite','Latin','Luwian','Middle_Breton', 'Middle_Dutch', 'Middle_English', 'Middle_Greek', 'Middle_High_German', 'Middle_Irish', 'Middle_Low_German', 'Middle_Persian', 'Middle_Welsh','Old_Church_Slavonic', 'Old_Dutch', 'Old_English', 'Old_French', 'Old_Frisian', 'Old_Georgian', 'Old_High_German', 'Old_Irish', 'Old_Italian', 'Old_Norse', 'Old_Persian', 'Old_Portuguese', 'Old_Provençal', 'Old_Prussian', 'Old_Russian', 'Old_Saxon', 'Old_Spanish', 'Old_Swedish','Pali','Prakrit','Sanskrit','Sogdian','Tocharian_A', 'Tocharian_B']


def generate_data():
    #load data
    f = open('diacl_qualitative_coding.tsv','r')
    features = f.read()
    f.close()
    #convert to list
    features = [l.split('\t') for l in features.split('\n')]
    #get rid of data from ancient/medieval languages
    features = [l for l in features if l[0] not in langs_to_exclude]
    #sorted list of all unique languages
    lang_list = sorted(set([l[0] for l in features]))
    #sorted list of all unique feature-variant pairs
    feat_var_list = sorted(set([(l[1],l[2]) for l in features]))
    #convert data to numeric values
    lang_inds = np.array([lang_list.index(l[0]) for l in features],dtype=np.int32)
    feat_var_inds = np.array([feat_var_list.index((l[1],l[2])) for l in features],dtype=np.int32)
    #number of all features pairs
    X = len(sorted(set([l[1] for l in features])))
    #number of languages
    L = len(lang_list)
    #number of all feature,variant pairs
    S = len(feat_var_list)
    #number of all feature,variant pairs - 1 (for unconstrained parameterization)
    S_u = S-X
    #length of each distribution in collection
    R = [len([l for l in feat_var_list if l[0]==f]) for f in sorted(set([l[1] for l in features]))]
    #length of each distribution in collection - 1 (for unconstrained parameterization)
    R_u = [r-1 for r in R]
    #indices of first and last+1 element in each partition
    part = [[0,R[0]]]+[[reduce(lambda x,y:x+y,R[:i]),reduce(lambda x,y:x+y,R[:i+1])] for i in range(1,len(R))]
    part_u = [[0,R_u[0]]]+[[reduce(lambda x,y:x+y,R_u[:i]),reduce(lambda x,y:x+y,R_u[:i+1])] for i in range(1,len(R_u))]
    #number of datapoints
    N = len(features)
    return(lang_inds,feat_var_inds,X,L,S,S_u,R,R_u,part,part_u,N)



def initialize_unconstrained_params(K,L,S_u):
    lambda_u = tfd.Uniform(-1.,1.).sample()
    alpha_u = tfd.Uniform(-1.,1.).sample()
    theta_u = tfd.Uniform(tf.ones([L,K-1])*-1.,tf.ones([L,K-1])*1.).sample()
    phi_u = tfd.Uniform(tf.ones([K,S_u])*1.,tf.ones([K,S_u])*1.).sample()
    #phi_u = [tfd.Uniform(tf.ones([K,R_u[x]])*-100.,tf.ones([K,R_u[x]])*100.).sample() for x in range(X)]
    return(lambda_u,alpha_u,theta_u,phi_u)


def transform_params(lambda_u,alpha_u,theta_u,phi_u,X,part_u):
    lambda_c = tfb.Softplus().forward(lambda_u)
    alpha_c = tfb.Softplus().forward(alpha_u)
    theta_c = tfb.SoftmaxCentered().forward(theta_u)
    phi_c = tf.concat([tfb.SoftmaxCentered().forward(phi_u[:,part_u[x][0]:part_u[x][1]]) for x in range(X)],axis=1)
    #phi_c = [tfb.SoftmaxCentered().forward(phi_u[x]) for x in range(X)]
    return(lambda_c,alpha_c,theta_c,phi_c)


def joint_log_prob(lang_array,feat_var_array,K,L,R,X,part,part_u,lambda_u,alpha_u,theta_u,phi_u):
    lambda_c,alpha_c,theta_c,phi_c = transform_params(lambda_u,alpha_u,theta_u,phi_u,X,part_u)
    #define the priors
    lambda_ = tfd.Gamma(1.,1.)
    theta = tfd.Dirichlet(tf.ones([L,K])*alpha_c)
    phi = [tfd.Dirichlet(tf.ones([R[x]])*lambda_c) for x in range(X)]
    #compute log priors
    lprior = lambda_.log_prob(lambda_c)
    lprior += tf.reduce_sum(theta.log_prob(theta_c))
    lprior += tf.reduce_sum([phi[x].log_prob(phi_c[:,part[x][0]:part[x][1]]) for x in range(X)])
    #compute log likelihood
    llik = tf.reduce_sum(tf.reduce_logsumexp(tf.gather(tf.log(theta_c),lang_array) + tf.gather(tf.log(tf.transpose(phi_c)),feat_var_array),axis=1))
    #return log posterior
    return(lprior+llik)





def run_inference(K,chains):
    #create step size variable
    step_size = tf.get_variable(name='step_size',initializer=1e-5,use_resource=True,trainable=False)
    #define HMC transition kernel
    n_results = 800
    discard = 2000
    lang_inds,feat_var_inds,X,L,S,S_u,R,R_u,part,part_u,N = generate_data()
    posts = []
    for c in range(chains):
        #tf.reset_default_graph()
        lambda_u,alpha_u,theta_u,phi_u = initialize_unconstrained_params(K,L,S_u)
        def target_log_prob_fn(lambda_u,alpha_u,theta_u,phi_u):
            return(joint_log_prob(lang_inds,feat_var_inds,K,L,R,X,part,part_u,lambda_u=lambda_u,alpha_u=alpha_u,theta_u=theta_u,phi_u=phi_u))
        kernel = tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=target_log_prob_fn,
                    step_size=step_size,
                    step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(),
                    num_leapfrog_steps=5)
        states, kernel_results = tfp.mcmc.sample_chain(
            num_results=n_results,num_burnin_steps=discard,num_steps_between_results=10,
            current_state=[lambda_u,alpha_u,theta_u,phi_u],
            kernel=kernel)
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            init_op.run()
            states_, kernel_results_ = sess.run([states, kernel_results])
            print(kernel_results_.is_accepted.mean())
            print(sess.run(step_size))
        posts.append((states_,kernel_results_))
        print('chain {} finished'.format(c))
    f = open('posterior_infer_{}.pkl'.format(K),'wb')
    pkl.dump(posts,f)
    f.close()




def main():
    assert(len(sys.argv)==2)
    try:
        int(sys.argv[1])
    except ValueError:
        print('usage: python run_K_infer.py [n clusters]')
    K = int(sys.argv[1])
    run_inference(K,4)
    
    


if __name__=='__main__':
    main()