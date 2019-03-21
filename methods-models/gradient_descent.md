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
+ gradient
$$grad f(x_{0},\cdots ,x_{n})=(\frac{\partial f}{\partial x_{0}},\cdots ,\frac{\partial f}{\partial x_{j}},\cdots ,\frac{\partial f}{\partial x_{n}})$$
函数在变量空间的某一点处，沿着哪一个方向上，有最大的变化率？梯度是一个向量，是函数f(x1,x2,……,xn)在某一点x处 沿最大方向导数方向上 的变化率，其方向与取得最大方向导数的方向一致，模长为方向导数的最大值。

在微积分里面，对多元函数的参数求∂偏导数，把求得的各个参数的偏导数以向量的形式写出来，就是梯度。比如函数f(x,y), 分别对x,y求偏导数，
求得的梯度向量就是(∂f/∂x, ∂f/∂y)T,简称grad f(x,y)或者▽f(x,y)。对于在点(x0,y0)的具体梯度向量就是(∂f/∂x0, ∂f/∂y0)T.或者▽f(x0,y0).

**梯度向量的几何意义**：函数变化增加最快的地方。具体来说，对于函数f(x,y),在点(x0,y0),沿着梯度向量的方向就是(∂f/∂x0, ∂f/∂y0)T的方向，
是f(x,y)增加最快的地方。沿着梯度向量的方向，更容易找到函数的最大值；沿着梯度向量相反的方向，即沿着-(∂f/∂x0, ∂f/∂y0)T的方向，梯度减少
最快，更容易找到函数的最小值。
### 2 梯度下降与梯度上升
机器学习中，在最小化损失函数时，通过梯度下降算法一步步迭代求解，得到最小化的损失函数 和 模型参数值。反之，如果需要求解
损失函数的最大值，则使用梯度上升法。
### 3 梯度下降算法详解
#### 3.1 regression problem
给定训练样本：
$$\mathbf{T}=\{(x_{1},y_{1}),\cdots ,(x_{N},y_{N})\}$$
其中，样本个数为N，每个样本有m个属性。任务是：预测未知样本的输出。
+ 代数表示法loss function:
$$J(\theta )=\frac{1}{2}\sum_{i=1}^{N}(h_{\theta }(x^{(i)})-y^{i})^{2}$$
我们的目标是：得到 使得损失函数$J(\theta )$最小的 参数Θ。solution：gradient descent。
> inital parameter Θ

> repeat 
$$\theta \_{j}=\theta \_{j}-\alpha \frac{\partial J(\theta )}{\partial \theta_{j}}$$
其中，
$$\frac{\partial J(\theta )}{\partial \theta_{j}}=\sum_{i=1}^{N}(h_{\theta }(x^{(i)})-y^{i})\*x_{j}^{i}$$
+ 向量表示法 loss function：
$$J(\theta )=\frac{1}{2}\sum_{i=1}^{N}(h_{\theta }(x^{(i)})-y^{(i)})^{2}=\frac{1}{2}\mathbf{(\mathbf{X}\theta -y)}^\mathrm{T}(\mathbf{X}\theta -y)$$
输入矩阵X表示为：![Matrix_X](https://github.com/Vita112/notes_for_NLP/blob/master/methods-models/img/Matrix_X.gif)
求关于Θ的偏导：
$$\bigtriangledown \_{\theta }(J(\theta ))=\bigtriangledown \_{\theta }\frac{1}{2}\mathbf{(\mathbf{X}\theta -y)}^\mathrm{T}(\mathbf{X}\theta -y)=\frac{1}{2}\bigtriangledown \_{\theta }(\theta ^\mathrm{T}\mathbf{X}^\mathrm{T}\mathbf{X}\theta-\theta ^\mathrm{T}\mathbf{X}^\mathrm{T}y-y ^\mathrm{T}\mathbf{X}\theta+ y^\mathrm{T}y)=\mathbf{X}^\mathrm{T}\mathbf{X}\theta-\mathbf{X}^\mathrm{T}y$$
令偏导结果为0，有
$$\mathbf{X}^\mathrm{T}\mathbf{X}\theta-\mathbf{X}^\mathrm{T}y=0$$
于是得到，
$$\hat{\theta }=(\mathbf{X}^\mathrm{T}\mathbf{X})^{-1}\mathbf{X}^\mathrm{T}y$$
### 4 梯度下降算法调优
+ 学习率α的选择
+ 参数Θ初始值得选择
+ 归一化
### 5 梯度下降算法变体
存在3种变体，他们的区别在于：每一次梯度更新中，会使用多少训练数据。
#### 5.1 批量梯度下降算法（batch gradient descent）
在整个数据集上求出J(Θ)，并对每个参数Θ求J(Θ)的偏导。由于在每次更新前，会对相似的样本求梯度值，因此在较大数据集上会出现redundant。
可能得到一个局部极小值。

速度慢，对于较大的、内存无法容纳的数据集，该方法不可用。
#### 5.2 随机梯度下降算法（stochastic gradient descent）
每次更新时，随机选取训练集中的一个样本（x，y），求出J(Θ)，并对参数Θ求J(Θ)的偏导。

更新值的方差很大，J(Θ)显示出剧烈的波动。一方面，算法在收敛过程中的波动，会帮助J(Θ)跳入另一个 可能的更小的 极小值；另一方面，
由于算法可能持续波动而不停止，导致有些时候无法判断算法是否已经收敛。
#### 5.3 小批量梯度下降算法（mini-batch gradient descent）
在每次更新中，对 n 个样本构成的一批数据，计算损失函数 J(θ)，并对相应的参数求导。这种方法：(a) 降低了更新参数的方差（variance），使得收敛过程更为稳定；(b) 能够利用最新的深度学习程序库中高度优化的矩阵运算器，能够高效地求出每小批数据的梯度。
### 6 梯度下降的优化算法
#### 6.1 动量法（momentum）
![SGD_with_momentum](https://github.com/Vita112/notes_for_NLP/blob/master/methods-models/img/SGD_with_momentum.jpg)

如上图所示，momentum帮助SGD在相关方向加速前进，并减少震荡。公式中在原有项前增加一个折损稀疏γ，如下：
$$v_{t}=\gamma v_{t-1}+\eta \bigtriangledown \_{\theta }J(\theta ))$$
$$\theta =\theta -v_{t}$$
此处，动量项γ通常被设置为 0.9.通俗来讲，动量法就像从高坡上退下一个小球，小球在向下滚动的过程中积累了动量，
在途中越来越快。在参数更新时，动量项在梯度指向方向相同的 方向上逐渐增大，在梯度指向改变的方向逐渐减小。
#### 6.2 Nesterov 加速梯度法
![Nesterov_update]()

给予梯度下降方向引导的方法。在momentum方法中，我们使用动量项$\gamma v_{t-1}$来移动参数项Θ；
**Nesterov 加速梯度法通过基于未来参数的近似值，计算相应的J(θ)，并求偏导，使算法高效地前进并收敛**，公式如下：
$$v_{t}=\gamma v_{t-1}+\eta \bigtriangledown \_{\theta }J(\theta -\gamma v_{t-1}))$$
$$\theta =\theta -v_{t}$$
结合上图，蓝色向量代表momentum方法的过程，现在当前梯度值前进一小步，然后在更新后的梯度向量上前进一大步。

在NAG方法中，首先在棕色向量上迈出一大步，再根据当前情况(红色向量)修正，最后得到最终的前进方向(绿色向量)。
#### 6.3 Adagrad法
对不同参数调整学习率，具体而言，对低频出现的参数进行大的更新，对高频出现的参数进行小的更新。

适合于处理稀疏数据；大大提升了SGD的鲁棒性，在训练单词向量映射(word embedding)时，其中不频繁出现的词需要比频繁出现的词
更大的更新值。

1.对每个参数的更新过程：记在迭代次数t下，对参数$\theta \_{i}$求J(θ)梯度的结果记为gt,i:
$$g_{t,i}=\bigtriangledown \_{\theta }J(\theta \_{t,i})$$


 
