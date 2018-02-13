import torch
from torch.autograd import Variable
#----------------------------------------------------------------------------
dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
#----------------------------------------------------------------------------
Batch_size = 64     # Batch size
Q = 1000            # Input size
S = 100             # Number of neurons
a = 10              # Network output size
#----------------------------------------------------------------------------

p = Variable(torch.randn(Batch_size, Q).type(dtype), requires_grad=False)
t = Variable(torch.randn(Batch_size, a).type(dtype), requires_grad=False)

w1 = Variable(torch.randn(Q, S).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(S, a).type(dtype), requires_grad=True)

learning_rate = 1e-6
#----------------------------------------------------------------------------
for t in range(500):

    a_net = p.mm(w1).clamp(min=0).mm(w2)
    loss = (a_net - t).pow(2).sum()
    print(t, loss.data[0])

    loss.backward()

    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    w1.grad.data.zero_()
    w2.grad.data.zero_()