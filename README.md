# PINN-Lid-Driven-Cavity-Problem
PINN盖驱动腔问题（层流）

原文地址：https://medium.com/@oladayo_7133/solving-the-lid-driven-cavity-problem-using-physics-informed-neural-networks-pinns-2dd14005751a

# Problem Description

We aim to simulate the steady-state, incompressible flow of a Newtonian fluid within a two-dimensional square cavity:

- **Domain**: (x, y) in \[0, 1\] x \[0, 1\]
## **Boundary Conditions**:

- **Top Lid** ( y = 1 ): Moves rightward with velocity U_lid = 1.0
- **Other Walls**: Stationary (U = V = 0 )
- **Fluid Properties**: **Kinematic Viscosity**: nu = 0.01 (Reynolds number _Re = 100_)

## 控制方程

$$\frac{\partial U}{\partial x}+\frac{\partial V}{\partial y}=0$$
$$U\frac{\partial U}{\partial x}+V\frac{\partial U}{\partial y}=-\frac{\partial P}{\partial x}+\nu\left(\frac{\partial^2U}{\partial x^2}+\frac{\partial^2U}{\partial y^2}\right)$$
$$U\frac{\partial V}{\partial x}+V\frac{\partial V}{\partial y}=-\frac{\partial P}{\partial y}+\nu\left(\frac{\partial^2V}{\partial x^2}+\frac{\partial^2V}{\partial y^2}\right)$$

## 边界条件

- Top Lid: _U_ = _U_lid_, _V = 0_  
- Other Walls: _U = V = 0_

TenserFlow版本：
![](https://github.com/srrdhy/PINN-Lid-Driven-Cavity-Problem/blob/main/images/3b77b897cba238ff73aafa64993799e.png)
Pytorch版本：
![](https://github.com/srrdhy/PINN-Lid-Driven-Cavity-Problem/blob/main/images/cdf839c75eb2e1ea07ad58e63b664e4.png)
原作者结果：（使用TenserFlow）
![](https://github.com/srrdhy/PINN-Lid-Driven-Cavity-Problem/blob/main/images/Pasted%20image%2020250216153643.png)
## 结果解释：

- 初级涡流：由于盖子的运动，在腔体中形成较大的循环涡流。
- 二次涡旋：雷诺数较高时，角落附近可能会出现较小的涡旋。
- 速度分布：腔内速度分量U和V平滑变化，满足不可压缩条件。

## 训练中遇到的错误

边界条件张量形状不匹配，导致损失计算错误。
错误结果图：损失无法下降
![](https://github.com/srrdhy/PINN-Lid-Driven-Cavity-Problem/blob/main/images/b474cf0760b1bac92b35766be3d951d.png)

在 PyTorch 版本中：
- 模型输出 `u_pred` 和 `v_pred` 的形状为 `(N_b,)`（一维张量）。
- 边界条件 `u_b` 和 `v_b` 的形状为 `(N_b, 1)`（二维张量）。

当计算 `(u_pred - u_b)` 时，由于形状不匹配，PyTorch 会触发广播机制，导致实际计算的是所有元素对的差值，产生错误的巨大损失值，阻碍模型训练。

修复步骤
1. 调整边界条件张量形状：将 `u_b` 和 `v_b` 从 `(N_b, 1)` 转换为 `(N_b,)`，使其与模型输出形状一致。
2. 修改边界条件生成代码：在生成 `u_top`, `u_side` 等张量时，去掉多余的维度。
错误代码：
```python
# Boundary conditions
u_top = U_lid * torch.ones((N_b // 4, 1), device=device)
v_top = torch.zeros((N_b // 4, 1), device=device)

u_side = torch.zeros((3 * N_b // 4, 1), device=device)
v_side = torch.zeros((3 * N_b // 4, 1), device=device)

u_b = torch.cat([u_side, u_top], dim=0)
v_b = torch.cat([v_side, v_top], dim=0)
```
修改为：
```python
# 修正边界条件张量形状
u_top = U_lid * torch.ones((N_b // 4,), device=device)  # 形状 (N_b//4,)
v_top = torch.zeros((N_b // 4,), device=device)
u_side = torch.zeros((3 * N_b // 4,), device=device)
v_side = torch.zeros((3 * N_b // 4,), device=device)

u_b = torch.cat([u_side, u_top], dim=0)  # 形状 (N_b,)
v_b = torch.cat([v_side, v_top], dim=0)
```
补充速度大小分布图：可见两者区别不大，与COMSOL结果接近

TensorFlow:
![](https://github.com/srrdhy/PINN-Lid-Driven-Cavity-Problem/blob/main/images/%E5%B1%82%E6%B5%81%E7%9B%96%E8%85%94%E9%80%9F%E5%BA%A6%E5%A4%A7%E5%B0%8F%E5%88%86%E5%B8%83%E5%9B%BE_tf.png)
Pytorch:
![](https://github.com/srrdhy/PINN-Lid-Driven-Cavity-Problem/blob/main/images/%E5%B1%82%E6%B5%81%E7%9B%96%E8%85%94%E9%80%9F%E5%BA%A6%E5%A4%A7%E5%B0%8F%E5%88%86%E5%B8%83%E5%9B%BE_torch.png)
COMSOL：
![](https://github.com/srrdhy/PINN-Lid-Driven-Cavity-Problem/blob/main/images/comsol%E7%BB%93%E6%9E%9C.jpg)
