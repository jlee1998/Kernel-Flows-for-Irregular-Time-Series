# Kernel Flows for Irregular Time Series

Kernel flows is an algorithm developed by Owhadi and Yoo (2018) for optimising the parameters of the kernel for kernel ridge regression. The following description is adapted from the paper by Hamzi and Owhadi (2021), who successfully applied the method to the logistic map, the Henon map and the Lorenz attractor:

Suppose we have an <img src="https://render.githubusercontent.com/render/math?math=\mathrm{R}^d"> dimensional time series <img src="https://render.githubusercontent.com/render/math?math=x_1, ..., x_k,..."> taken from a deterministic dynamical system, and that we observe the process at regular intervals up to some time <img src="https://render.githubusercontent.com/render/math?math=t_n"> . How can we forecast the time series at time <img src="https://render.githubusercontent.com/render/math?math=t_{n + 1}">; and more generally, up to some future time <img src="https://render.githubusercontent.com/render/math?math=t_{\tau+n}">, where the time lag <img src="https://render.githubusercontent.com/render/math?math=\tau"> may be unknown? Assume that there exists a surrogate dynamical system with solution 

<img src="https://render.githubusercontent.com/render/math?math=z_{k+1}=f^\dagger (z_k, ... , z_{k-\tau^\dagger+1})">

, where no assumptions have been made about <img src="https://render.githubusercontent.com/render/math?math=f^\dagger ">
 and <img src="https://render.githubusercontent.com/render/math?math=\tau^\dagger ">. Let the input of <img src="https://render.githubusercontent.com/render/math?math=f^\dagger "> be <img src="https://render.githubusercontent.com/render/math?math=\mathrm{R}^d "> . Next, fit a regular kernel regression model on $X=(X_1,...,X_N)$ and $Y=(Y_1,...,Y_N)$ where $N=n-\tau$:

\begin{equation}
    f(x)= K(x,X)(K(X,X))^{-1}Y
\end{equation}

The fact that $f(x)$ is a suitable interpolant is demonstrated by the boundedness of the error between $f$ and $f^\dagger$ by :

\begin{equation}
    |f^\dagger (x)-f(x)|\leq \sigma(x)||f^\dagger (x)||_{\mathcal{H}}
\end{equation}

where f(x) is the conditional mean and

\begin{equation}
    \sigma^2 (x)=K(x,x)-K(x,X)(K(X,X))^{-1}K(x,X)^T
\end{equation}

is the conditional standard deviation of the Gaussian process $\xi \widetilde \mathcal{N} (0,K)$ 

Let $K_\theta (x,x')$ be the interpolating kernel parametrized by $\theta$. The algorithm runs as follows:

1. Select $\frac{M}{2}$ points at random from the labelled points: $x_1,...,x_{\frac{M}{2}}$, $y_1,...,y_{\frac{M}{2}}$, and interpolate them with the kernel $K_\theta$

2. Define $u_{\frac{M}{2}} (x)= K_\theta (x,X_\frac{M}{2})(K(X_{\frac{M}{2}},X_{\frac{M}{2}}))^{-1}Y_{\frac{M}{2}}$, where $X_{\frac{M}{2}}$ is the vector containing a subset of the data points and $Y$ is the vector of labels. It is clear that $\rho(\theta)$ lies between 0 and 1 inclusive.

3. Define

\begin{equation}
\rho(\theta) =\frac{ ||u_M -u_{\frac{M}{2}}||^2_{K_\theta}}{||u_M||^2_{K_\theta}}=1-\frac{Y^T_{\frac{M}{2}} K_\theta (X_{\frac{M}{2}},x_{\frac{M}{2}})^{-1}Y_{\frac{M}{2}}}{Y^T K_\theta (X,X)Y}
\end{equation}
, the relative square error between the interpolants $u_M$ and $u_{\frac{M}{2}}$

4. Apply stochastic gradient descent with respect to $\theta$ in the direction of $\rho$: $\theta \leftarrow \theta -\delta \nabla_\theta \rho$

5. Repeat until the error reaches a minimum.



