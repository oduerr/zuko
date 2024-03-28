import torch
import zuko
import numpy as np
from zuko.transforms import *

import matplotlib.pyplot as plt

############## Testing to_theta ################
theta_im = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.6]], dtype=torch.float32)
result =  BernTransform.to_theta(theta_im)
#Calculated with R to_theta(c(1,2,3,4,5,6.6))
#tf.Tensor(
#[[ 1.        3.126928  6.175515]
# [ 4.        9.006716 15.608075]], shape=(2, 3), dtype=float32)
expected_result = torch.tensor([[1.,        3.126928,  6.175515], [4.,        9.006716, 15.608075]], dtype=torch.float32)
if torch.allclose(result, expected_result, atol=1e-6):
    print("Test passed.")
else:
    print("Test failed.")    

############### Testing call ##################
theta_un = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.6]], dtype=torch.float32)
theta_un = theta_im.unsqueeze(dim=1) #Shape [2,3] --> [2, 1, 3]
bern = BernTransform(theta_un)

####### This is not working #####
y = torch.tensor([0.1,0.2], dtype=torch.float32).view(-1)
result = bern(y)
print(result)
##################################

#We assume that the input is a tensor of shape [batch_size, 1] and not [batch_size]
y = torch.tensor([0.1,0.2], dtype=torch.float32).view(-1,1)
result = bern(y)
print(result)
if torch.allclose(result,  torch.tensor([[1.434602], [6.066472]]), atol=1e-6):
    print("Va bene")

############### Testing Probability ##################
result = bern.eval_h_dash(y)
if torch.allclose(result,  torch.tensor([[4.438188], [10.651289]]), atol=1e-6):
     print("Va bene")



############### Old Faithfull ##################
M = 30

ynp = np.asarray(
    (0.6694, 0.3583, 0.6667, 0.6667, 0.6667, 0.3333, 0.7306, 0.7139, 0.3389, 0.8056, 0.3056, 0.9083, 0.2694, 0.8111,
     0.7306, 0.2944, 0.7778, 0.3333, 0.7889, 0.7028, 0.3167, 0.8278, 0.3333, 0.6667, 0.3333, 0.6667, 0.4722, 0.75,
     0.6778, 0.6194, 0.5861, 0.7444, 0.3694, 0.8139, 0.4333, 0.6917, 0.3667, 0.7944, 0.3056, 0.7667, 0.3778, 0.6889,
     0.3333, 0.6667, 0.3333, 0.6667, 0.3139, 0.7111, 0.3472, 0.7444, 0.4167, 0.6667, 0.2944, 0.7222, 0.3639, 0.7472,
     0.6472, 0.5556, 0.6222, 0.6667, 0.325, 0.8778, 0.3333, 0.6667, 0.3333, 0.6667, 0.3333, 0.6667, 0.5889, 0.3611, 0.75,
     0.3361, 0.6917, 0.7, 0.7222, 0.3222, 0.775, 0.6361, 0.6722, 0.6944, 0.7778, 0.3028, 0.6667, 0.5, 0.6667, 0.3333, 0.7417,
     0.3417, 0.7083, 0.3194, 0.7778, 0.2889, 0.7306, 0.2944, 0.7667, 0.3111, 0.7417, 0.2722, 0.8389, 0.3028, 0.85, 0.2722, 0.7139,
     0.3333, 0.6667, 0.3333, 0.7556, 0.3333, 0.6667, 0.4889, 0.7889, 0.65, 0.325, 0.6861, 0.3, 0.7778, 0.3056, 0.7833, 0.3528, 0.7972,
     0.3028, 0.6833, 0.775, 0.6667, 0.3333, 0.6667, 0.6667, 0.7028, 0.6889, 0.6556, 0.625, 0.7361, 0.4111, 0.6944, 0.6333, 0.7194,
     0.6444, 0.7806, 0.2833, 0.8278, 0.7111, 0.7639, 0.6667, 0.6667, 0.6667, 0.6667, 0.3306, 0.7667, 0.1389, 0.8194, 0.2889, 0.7639,
     0.2833, 0.7917, 0.3056, 0.75, 0.3111, 0.7417, 0.7417, 0.6667, 0.8, 0.6667, 0.6667, 0.3333, 0.6667, 0.3222, 0.7639, 0.3333, 0.6167,
     0.4778, 0.8056, 0.575, 0.7306, 0.3, 0.7333, 0.4139, 0.7528, 0.35, 0.725, 0.7278, 0.2972, 0.8194, 0.3028, 0.6667, 0.6667, 0.6667, 0.6444,
     0.3083, 0.7833, 0.3361, 0.7444, 0.3111, 0.6944, 0.3167, 0.7083, 0.5417, 0.7028, 0.3139, 0.8306, 0.3083, 0.6667, 0.3278, 0.7944, 0.6667, 0.3333,
     0.6667, 0.6667, 0.3972, 0.7361, 0.7028, 0.7278, 0.3333, 0.7417, 0.2917, 0.75, 0.2694, 0.7833, 0.4278, 0.6167, 0.7056, 0.3222, 0.725, 0.6667, 0.6667,
     0.6667, 0.7028, 0.6667, 0.6889, 0.3139, 0.7444, 0.325, 0.7028, 0.2861, 0.7417, 0.7083, 0.6611, 0.7306, 0.3278, 0.7417, 0.7111, 0.3194, 0.7361, 0.5,
     0.6667, 0.3333, 0.6667, 0.5472, 0.3056, 0.7694, 0.3056, 0.7694, 0.7667, 0.7083, 0.3222, 0.8306, 0.3278, 0.7167, 0.7, 0.7556, 0.7333, 0.7694, 0.3333,
     0.6667, 0.6667, 0.6528, 0.3333, 0.75, 0.3, 0.6667, 0.4583, 0.7889, 0.6611, 0.325, 0.8278, 0.3083, 0.8, 0.6667, 0.6667, 0.6667, 0.6667, 0.6667, 0.6667,
     0.6667, 0.3333, 0.6667, 0.3222, 0.7222, 0.2778, 0.7944, 0.325, 0.7806, 0.3222, 0.7361, 0.3556, 0.6806, 0.3444, 0.6667, 0.6667, 0.3333),
    np.float32
)
y = torch.tensor(ynp, dtype=torch.float32).view(-1, 1)
x = torch.ones_like(y)


# Neural spline flow (NSF) with 1 sample features (y) and 5 context features (x's)
#transforms is chaining
flow = zuko.flows.NSF(1, 1, transforms=1, hidden_features=[128] * 3)

#flow = zuko.flows.SOSPF(1, 1, transforms=1, hidden_features=[128] * 3)
#flow = zuko.flows.MAF(1, 1, transforms=1, hidden_features=[128] * 3)

# Bernstein Flow with 1 sample features (y) and 5 context features (x's)
flow = zuko.flows.BERN(1, 1, transforms=1, hidden_features=[128] * 3, degree=M)

# Train to maximize the log-likelihood
optimizer = torch.optim.AdamW(flow.parameters(), lr=1e-3)

loss_history = []
for e in range(500):
    loss = -flow(x).log_prob(y)  # -log p(y | x)
    loss = loss.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_history.append(loss.detach().numpy())
    if (e) % 10 == 0:
        print( f"epoch {e} loss{loss.detach().numpy()}")

flow(x[:1])

#Plot the loss
plt.plot(loss_history)
plt.title("Loss")
plt.show()

# create histogram of y and the samples x
xx = torch.linspace(0.1, 0.9, 1000)
yy = flow(x[:1]).log_prob(xx[...,None]).exp().detach().numpy()
y_sample = flow(x[0]).sample((6400,)).detach().numpy()

plt.hist(y.detach().numpy(), bins=30, density=True, alpha=0.5, color='green')
plt.hist(y_sample, bins=30, density=True, alpha=0.5)
plt.plot(xx.detach().numpy(), yy.squeeze())
plt.title("Old Faithfull unconditional data")
plt.legend([f'p(y) M={M}', 'Data (Training)', 'Samples from p(y)'])
plt.show()