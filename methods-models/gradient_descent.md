## Gradient Descent summary
### 1 导数、偏导数、方向导数与梯度
+ derivative
$$f{x_{0}}'=\lim_{\Delta x\rightarrow 0}\frac{\Delta y}{\Delta x}=\lim_{\Delta x\rightarrow 0}\frac{f(x_{0}+\Delta x)-f(x_{0})}{\Delta x}$$
函数y=f(x)在某一点x0处沿着x轴正方向的变化率。如果f'(x)＞0，说明f(x)在x点处沿着x轴正方向是趋于增加的；反之，则趋于减少。
+ partial derivative
$$\frac{\partial f(x_{0},x_{1},\cdots ,x_{n}) }{\partial x_{j}}=\lim_{\Delta x\rightarrow 0}\frac{\Delta y}{\Delta x}=\lim_{\Delta x\rightarrow 0}\frac{f(x_{0},\cdots ,x_{j}+\Delta x,\cdots ,x_{n})-f(x_{0},\cdots ,x_{j},\cdots ,x_{n})}{\Delta x}$$
多元函数中，函数f(x1,x2,……,xn)在某一点x处沿坐标轴xj正方向的变化率。
+ directional derivative
$$\frac{\partial f(x_{0},x_{1},\cdots ,x_{n}) }{\partial l}=\lim_{\rho \rightarrow 0}\frac{\Delta y}{\Delta x}=\lim_{\rho \rightarrow 0}\frac{f(x_{0}+\Delta x_{0},\cdots ,x_{j}+\Delta x_{j},\cdots ,x_{n}+\Delta x_{n})-f(x_{0},\cdots ,x_{j},\cdots ,x_{n})}{\rho }$$
$$\rho =\sqrt{(\Delta x_{0})^{2}+\cdots +(\Delta x_{j})^{2}+\cdots +(\Delta x_{n})^{2}}$$
多元函数中，函数f(x1,x2,……,xn)在某一点x处沿 某一趋近方向上的 变化率。


在微积分里面，对多元函数的参数求∂偏导数，把求得的各个参数的偏导数以向量的形式写出来，就是梯度。比如函数f(x,y), 分别对x,y求偏导数，
求得的梯度向量就是(∂f/∂x, ∂f/∂y)T,简称grad f(x,y)或者▽f(x,y)。对于在点(x0,y0)的具体梯度向量就是(∂f/∂x0, ∂f/∂y0)T.或者▽f(x0,y0).

**梯度向量的几何意义**：函数变化增加最快的地方。具体来说，对于函数f(x,y),在点(x0,y0),沿着梯度向量的方向就是(∂f/∂x0, ∂f/∂y0)T的方向，
是f(x,y)增加最快的地方。沿着梯度向量的方向，更容易找到函数的最大值；沿着梯度向量相反的方向，即沿着-(∂f/∂x0, ∂f/∂y0)T的方向，梯度减少
最快，更容易找到函数的最小值。
### 2 梯度下降与梯度上升
### 
