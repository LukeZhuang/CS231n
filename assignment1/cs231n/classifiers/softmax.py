import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train=X.shape[0]
  num_dim=X.shape[1]
  num_classes=W.shape[1]
  score=X.dot(W)
  for i in xrange(num_train):
    correct_class=y[i]
    sum_exp_score=0.0
    max_score=-np.Inf
    for j in xrange(num_classes):
      max_score=max(max_score,score[i,j])
    for j in xrange(num_classes):
      score[i,j]=np.exp(score[i,j]-max_score)
      sum_exp_score+=score[i,j]
    for j in xrange(num_classes):
      if j==correct_class:
        dW[:,j]+=(-sum_exp_score/score[i,correct_class])*((sum_exp_score-score[i,j])*score[i,j]/(sum_exp_score**2)*X[i,:])
      else:
        dW[:,j]+=(-sum_exp_score/score[i,correct_class])*(-score[i,correct_class]*score[i,j]/(sum_exp_score**2)*X[i,:])
    loss+=-np.log(score[i,correct_class]/sum_exp_score)
  loss/=num_train
  W_sum=0.0
  for i in xrange(num_dim):
    for j in xrange(num_classes):
      W_sum+=W[i,j]**2
  loss+=0.5*reg*W_sum
  dW/=num_train
  dW+=reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train=X.shape[0]
  score=X.dot(W)
  score-=np.max(score,1,keepdims=True)
  score=np.exp(score)
  loss_i=score[xrange(num_train),y]/np.sum(score,1)
  loss=np.sum(-np.log(loss_i))/num_train+0.5*reg*np.sum(np.square(W))
  partion_deriv=-(-1.0/loss_i).reshape(num_train,1)*score*score[xrange(num_train),y].reshape(num_train,1)/(np.sum(score,1,keepdims=True)**2)
  partion_deriv[xrange(num_train),y]*=-(np.sum(score,1)-score[xrange(num_train),y])/score[xrange(num_train),y]
  dW=X.T.dot(partion_deriv)
  dW/=num_train
  dW+=reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

if __name__ == '__main__':
  score=np.array([[1,2,3],[2,4,3],[3,4,5],[4,5,6]])
  y=np.array([1,1,1,2])
  # score-=np.max(score,1,keepdims=True)
  # score=np.exp(score)
  # print score[xrange(score.shape[0]),y]/np.sum(score,1)
  # print score
  # print score[xrange(score.shape[0]),y].reshape(score.shape[0],1)
  # print score*score[xrange(score.shape[0]),y].reshape(score.shape[0],1)
  # print score*y.reshape(score.shape[0],1)
  print 1.0/y
