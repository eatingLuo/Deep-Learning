import torch
#----------------------------------------------------------------------------
dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
#----------------------------------------------------------------------------
Batch_size = 64     # Batch size
Q = 1000            # Input size
S = 100             # Number of neurons
a = 10              # Network output size
#----------------------------------------------------------------------------
p = torch.randn(Batch_size, Q).type(dtype)
t = torch.randn(Batch_size, a).type(dtype)
#----------------------------------------------------------------------------
w1 = torch.randn(Q, S).type(dtype)
w2 = torch.randn(S, a).type(dtype)

learning_rate = 1e-6
#----------------------------------------------------------------------------
for t in range(500):

    h = p.mm(w1)
    h_relu = h.clamp(min=0)
    a_net = h_relu.mm(w2)

    loss = (a_net - t).pow(2).sum()
    print(t, loss)

    grad_y_pred = 2.0 * (a_net - t)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = p.t().mm(grad_h)

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2