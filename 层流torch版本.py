import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from collections import OrderedDict

if not os.path.isdir('./models'):
        os.mkdir('./models')

# Domain parameters
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
U_lid = 1.0
nu = 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def same_seed(seed): 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

same_seed(42)

# Neural Network
class PINN(nn.Module):  
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        depth,
        act=torch.nn.Tanh,
    ):
        super(PINN, self).__init__()
        
        layers = [('input', torch.nn.Linear(input_size, hidden_size))] # Input

        layers.append(('input_activation', act())) # Activation

        for i in range(depth): 
            layers.append(('hidden_%d' % i, torch.nn.Linear(hidden_size, hidden_size))) # Hidden Layer
            layers.append(('activation_%d' % i, act())) # Activation for Hidden Layer

        layers.append(('output', torch.nn.Linear(hidden_size, output_size)))

        layerDict = OrderedDict(layers)
        self.layers = torch.nn.Sequential(layerDict)

    # 添加权重初始化
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        out = self.layers(x)
        u, v, p = out[:, 0], out[:, 1], out[:, 2]  # 解包输出
        return u, v, p

# Data generation
N_f = 10000
N_b = 2000

# Collocation points
x_f = torch.rand((N_f, 1), dtype=torch.float32) * (x_max - x_min) + x_min
y_f = torch.rand((N_f, 1), dtype=torch.float32) * (y_max - y_min) + y_min
X_f = torch.cat([x_f, y_f], dim=1).to(device)

# Boundary points generation
def generate_boundary():
    n = N_b // 4
    # Left Wall (x = 0)
    left = torch.cat([x_min * torch.ones((n, 1), device=device), 
                      torch.rand((n, 1), device=device) * (y_max - y_min) + y_min], dim=1)
    
    # Right Wall (x = 1)
    right = torch.cat([x_max * torch.ones((n, 1), device=device), 
                       torch.rand((n, 1), device=device) * (y_max - y_min) + y_min], dim=1)
    
    # Bottom Wall (y = 0)
    bottom = torch.cat([torch.rand((n, 1), device=device) * (x_max - x_min) + x_min, 
                        y_min * torch.ones((n, 1), device=device)], dim=1)
    
    # Top Lid (y = 1)
    top = torch.cat([torch.rand((n, 1), device=device) * (x_max - x_min) + x_min, 
                     y_max * torch.ones((n, 1), device=device)], dim=1)
    
    return torch.cat([left, right, bottom, top], dim=0)

X_b = generate_boundary()

# Boundary conditions
# 修正边界条件张量形状
u_top = U_lid * torch.ones((N_b // 4,), device=device)  # 形状 (N_b//4,)
v_top = torch.zeros((N_b // 4,), device=device)
u_side = torch.zeros((3 * N_b // 4,), device=device)
v_side = torch.zeros((3 * N_b // 4,), device=device)

u_b = torch.cat([u_side, u_top], dim=0)  # 形状 (N_b,)
v_b = torch.cat([v_side, v_top], dim=0)

# Loss function
def compute_loss(model, X_f, X_b, u_b, v_b):
    X_f.requires_grad_(True)
    
    # First forward pass for continuity
    u, v, p = model(X_f)
    # print(u.shape) # torch.Size([10000])
    u = torch.squeeze(u)
    v = torch.squeeze(v)
    p = torch.squeeze(p)
    
    # First derivatives
    u_x = torch.autograd.grad(u, X_f, torch.ones_like(u), True, True)[0][:, 0]
    u_y = torch.autograd.grad(u, X_f, torch.ones_like(u), True, True)[0][:, 1]
    v_x = torch.autograd.grad(v, X_f, torch.ones_like(v), True, True)[0][:, 0]
    v_y = torch.autograd.grad(v, X_f, torch.ones_like(v), True, True)[0][:, 1]
    
    # Second derivatives
    u_xx = torch.autograd.grad(u_x, X_f, torch.ones_like(u_x), True, True)[0][:, 0]
    u_yy = torch.autograd.grad(u_y, X_f, torch.ones_like(u_y), True, True)[0][:, 1]
    v_xx = torch.autograd.grad(v_x, X_f, torch.ones_like(v_x), True, True)[0][:, 0]
    v_yy = torch.autograd.grad(v_y, X_f, torch.ones_like(v_y), True, True)[0][:, 1]
    
    # Pressure gradients
    p_x = torch.autograd.grad(p, X_f, torch.ones_like(p), True, True)[0][:, 0]
    p_y = torch.autograd.grad(p, X_f, torch.ones_like(p), True, True)[0][:, 1]
    
    # Momentum equations
    continuity = u_x + v_y
    momentum_u = u*u_x + v*u_y + p_x - nu*(u_xx + u_yy) 
    momentum_v = u*v_x + v*v_y + p_y - nu*(v_xx + v_yy)
    
    # Loss components
    pde_loss = torch.mean(continuity**2) + torch.mean(momentum_u**2) + torch.mean(momentum_v**2)

    u_pred, v_pred, _ = model(X_b)
    bc_loss = torch.mean((u_pred - u_b)**2) + torch.mean((v_pred - v_b)**2)
    
    return pde_loss + bc_loss

# Training setup
model = PINN(
            input_size=2,
            hidden_size=20,
            output_size=3,
            depth=8,
            act=torch.nn.Tanh
        ).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

best_loss = math.inf

# Training loop
for epoch in range(30000):
    optimizer.zero_grad()
    loss = compute_loss(model, X_f, X_b, u_b, v_b)
    loss.backward()
    optimizer.step()

    if loss < best_loss:
        best_loss = loss
        torch.save(model.state_dict(), './models/model.ckpt')
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.5f}")
        print('Saving model with loss {:.5f}...'.format(best_loss))

# Visualization
nx, ny = 50, 50
x = torch.linspace(x_min, x_max, nx, device=device)
y = torch.linspace(y_min, y_max, ny, device=device)
X, Y = torch.meshgrid(x, y, indexing='xy')
XY = torch.stack([X.ravel(), Y.ravel()], -1)


model.load_state_dict(torch.load('models/model.ckpt', map_location=device, weights_only=True))

with torch.no_grad():
    u, v, _ = model(XY)
    u = u.reshape(ny, nx).cpu()
    v = v.reshape(ny, nx).cpu()

plt.figure(figsize=(8,6))
plt.streamplot(X.cpu().numpy(), Y.cpu().numpy(), u.numpy(), v.numpy(), density=2)
plt.title('Predicted Velocity Field')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()

'''
Epoch 29500, Loss: 0.00260
Saving model with loss 0.00223...
Epoch 29600, Loss: 0.01676
Saving model with loss 0.00223...
Epoch 29700, Loss: 0.01214
Saving model with loss 0.00223...
Epoch 29800, Loss: 0.01059
Saving model with loss 0.00223...
Epoch 29900, Loss: 0.01020
Saving model with loss 0.00223...
'''