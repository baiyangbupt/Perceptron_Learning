#!/bin/env python

import numpy as np
import matplotlib 


def learn_perceptron(neg_examples_nobias, pos_examples_nobias, w_init, w_gen_feas):
    num_err_history = list()
    
    w_dist_history = []
    
    
    rows = neg_examples_nobias.shape[0]
    cols = neg_examples_nobias.shape[1]

    neg_examples = np.ones((rows,cols+1))
    for i in range(neg_examples_nobias.shape[0]):
        for j in range(neg_examples_nobias.shape[1]):
            neg_examples[i,j] = neg_examples_nobias[i,j]

    pos_examples = np.ones((pos_examples_nobias.shape[0],pos_examples_nobias.shape[1]+1))
    for i in range(pos_examples_nobias.shape[0]):
        for j in range(pos_examples_nobias.shape[1]):
            pos_examples[i,j] = pos_examples_nobias[i,j]


    w = w_init
    if w_gen_feas.shape[0] is 0:
        w_gen_feas = []

    itera = 0
    [mistakes0, mistakes1] = eval_perceptron(neg_examples,pos_examples,w)
    num_errs = len(mistakes0) + len(mistakes1)
    num_err_history.append(num_errs)

    print "Number of errors in iteration %d: %d\n" % (itera, num_errs)

    print "weights:\t", str(w)

    key = raw_input('Press enter to continue, q to quit')
    if key is 'q':
        return

    while num_errs > 0:
        itera = itera + 1

        w = update_weights(neg_examples,pos_examples,w)
        [mistake0, mistake1] = eval_perceptron(neg_examples, pos_examples,w)
        num_errs = len(mistake0) + len(mistake1)
        
        print "Number of errors in iteration %d: %d\n" % (itera,num_errs)

        print "weights:\t", str(w)

        key = raw_input("Press enter to continue, q to quit")
        if key == 'q':
            break

def update_weights(neg_examples, pos_examples, w_current):
    
    w = w_current
    num_neg_examples = neg_examples.shape[0]
    num_pos_examples = pos_examples.shape[0]

    for i in range(num_neg_examples):
        this_case = neg_examples[i,:]
        x = this_case.reshape(this_case.shape[0], 1)

        activation = np.dot(this_case,w)

        if activation >= 0:
            w = w-x

    for i in range(num_pos_examples):
        this_case = pos_examples[i,:]
        x = this_case.reshape(this_case.shape[0], 1)
        activation = np.dot(this_case,w)
        if activation <0:
            w = w+x
    
    return w

def eval_perceptron(neg_examples, pos_examples, w):
    num_neg_examples = neg_examples.shape[0]
    num_pos_examples = pos_examples.shape[0]

    mistake0 = []
    mistake1 = []

    for i in range(num_neg_examples):
        x = neg_examples[i,:].transpose()
        activation = np.dot(x.transpose(), w)

        if activation >= 0:
            mistake0.append(i)

    for i in range(num_pos_examples):
        x = pos_examples[i,:].transpose()
        activation = np.dot(x.transpose(),w)

        if activation < 0:
            mistake1.append(i)

    return [mistake0, mistake1]

