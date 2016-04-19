# Assignment 02
$$
\declareMathOperator{\cor}{cor}
\declareMathOperator{\cov}{cov}
\declareMathOperator{\quantile}{quantile}
$$

# Libraries used 


```r
library("pols503")
library("mvtnorm")
library("ggplot2")
library("dplyr")
library("broom")
```
If you do not have the **pols503** package installed, you can install it with,

```r
library("devtools")
install_github("UW-POLS503/r-pols503")
```

# OLS Estimator

The purpose of this homework is to provide a guided, hands-on tour through the properties of the least squares estimator, especially under common violations of the Gauss Markov assumptions. We will work through a series of programs which use simulated data --- i.e., data created with known properties --- to investigate how these violations affect the accuracy and precision of least squares estimates of slope parameters. Using repeated study of simulated datasets to explore the properties of statistical models is called Monte Carlo experimentation. Although you will not have to write much R code, you will need to read through the provided programs carefully to understand what is happening.

Monte Carlo experiments always produce the same results as analytic proofs for the specific case considered. Each method has advantages and disadvantages: proofs are more general and elegant, but are not always possible. Monte Carlo experiments are much easier to construct and can always be carried out, but findings from these experiments only apply to the specific scenario under study. Where proofs are available, they are generally preferable to Monte Carlo experiments, but proofs of the properties of more complicated models are sometimes impossible or impractically difficult. This is almost always the case for the properties of models applied to small samples of data. Here, we use Monte Carlo not out of necessity but for pedagogical purposes, as a tool to gain a more intuitive and hands-on understanding of least squares and its properties.

All of the simulations in this assignment will follow the same structure:

1. Define a population
2. Repeat $m$ times:

    1. Draw a sample from the population
    2. Run OLS on that sample
    3. Save statistics, e.g. coefficients, standard errors, $p$-values, from the sample regression.

3. Evaluate the distributions of the sample statistics, or summaries thereof, to determine how well OLS recovers the parameters of the population.

In this section, we will work through the code necessary to run a simulation.
However, in the problems, functions written for this problem set will do most of the simulation computation. 
This section is to help you to understand what those functions are doing, and to provide a mapping from the math to the code.


## Sampling Distribution of OLS

Start with example in which the population satisfies all the Gauss-Markov assumptions and we run a correctly specified regression on the samples drawn from that population.

In this example, the population model is
$$
\begin{aligned}[t]
Y_i &= \beta_0 + \sum_{j = 1}^k \beta_j x_{i,j} + \epsilon_i \\
\epsilon_i & \sim N(0, \sigma^2)
\end{aligned}
$$
For a sample $y$ from that population, the OLS regression which will be run is
$$
\begin{aligned}[t]
y_i &= \hat\beta_0 + \sum_{j = 1}^k \hat\beta_j x_{i,j} + \hat\epsilon_i \\
\hat\sigma^2 &= \frac{\sum \hat\epsilon_i }{n - k - 1}
\end{aligned}
$$
In this case, the regression run on the samples has the correct specification, but that will not necessarily be true for other exercises.



```r
sim_linear_normal <- function(.data, beta, sigma) {
  # X gives the number of observations in the data
  n <- nrow(X)
  # Draw data
  # This creates the X matrix
  X <- model.matrix(~., .data)
  # Create E(y | X)
  yhat <- X %*% beta
  # errors drawn from a normal distribution
  epsilon <- rnorm(n, mean = 0, sd = sigma)
  # actual y's
  y <- yhat + epsilon
  .data$y <- y
  # Estimate model
  mod <- lm(y ~ ., data = .data)
  # Return results
  tidy(mod)
}
```


```r
run_iterations <- function(.iter, FUN, ...) {
  results <- vector(mode = "list", length = .iter)
  p <- progress_estimated(.iter, min_time = 2)
  for (i in seq_len(.iter)) {
    .data <- FUN(...)
    .data[[".iter"]] <- i
    results[[i]] <- .data
    p$tick()$print()
  }
  bind_rows(results)
}
```



```r
summarize_params <- function(.data, beta) {
  ret <- .data %>%
    group_by(term) %>%
    summarize(estimate_mean = mean(estimate),
              estimate_sd = sd(estimate),
              std_error_mean = mean(std.error),
              estimate_se = sd(estimate) / sqrt(length(estimate)),
              std_error_se = sd(std.error) / sqrt(length(estimate)),
              iter = length(estimate))
  ret[["beta_true"]] <- beta
  ret
}
```

Now draw many samples of sample size 10,

```r
n <- 10
k <- 2
mu_X <- rep(0, k)
s_X <- rep(1, k)
R_X <- diag(k)
beta <- c(0, rep(1, k))
X <- as.data.frame(rmvnorm(n, mu_X, sdcor2cov(s_X, R_X)))
names(X) <- paste("x", seq_len(k), sep = "")
sigma <- 0.1
```
Run 1024 iterations

```r
run_iterations(1024, sim_linear_normal, .data = X, beta = beta, sigma = sigma)
```

```
## Source: local data frame [3,072 x 6]
## 
##           term    estimate  std.error  statistic      p.value .iter
##          (chr)       (dbl)      (dbl)      (dbl)        (dbl) (int)
## 1  (Intercept) -0.02049312 0.03126445 -0.6554769 5.331030e-01     1
## 2           x1  0.95325460 0.02069438 46.0634509 5.940591e-10     1
## 3           x2  0.97919229 0.05011569 19.5386382 2.295939e-07     1
## 4  (Intercept) -0.10556293 0.03762724 -2.8054919 2.631577e-02     2
## 5           x1  1.04598418 0.02490600 41.9972687 1.132077e-09     2
## 6           x2  0.84302655 0.06031500 13.9770632 2.271331e-06     2
## 7  (Intercept) -0.03501616 0.03931273 -0.8907079 4.026606e-01     3
## 8           x1  1.02500755 0.02602165 39.3905657 1.769895e-09     3
## 9           x2  0.98173058 0.06301676 15.5788796 1.085407e-06     3
## 10 (Intercept)  0.01158691 0.03468550  0.3340565 7.481218e-01     4
## ..         ...         ...        ...        ...          ...   ...
```

Suppose that we want to run this for several different sample sizes


## Correlated Variables

In the previous problem, the covariates were assumed to be independent.
Now, we will evaluate the properties of OLS estimates when covariates are correlated.
As before, the population is
$$
\begin{aligned}[t]
Y_i &= 0 + 1 \cdot x_{1,i} + 1 \cdot x_{2,i} + 1 \cdot x_{3,i} + \epsilon_i \\
\epsilon_i &\sim N(0, \sigma^2) \\
\sigma &= 1.7
\end{aligned}
$$
In this problem keep $\mu_X = (0, 0, 0)$ and $s_X = (1, 1, 1)$, but $R_X$ will differ between simulations to allow for different levels of correlation between $x_1$ and $x_2$.
The covariate $x_3$ is independent of the other covariates, $\cor(x_1, x_3) = \cor(x_2, x_3) = 0$.
Thus, the correlation matrix for $X$ in these simulations is the following, where $\rho_{1,2}$ will vary:
$$
R_X =
\begin{bmatrix}
1 & \rho_{1,2} & 0 \\
\rho_{1,2} & 1 & 0 \\
0 & 0 & 1 
\end{bmatrix}
$$

Simulate using `sim_lin_normal` with the following levels of correlation between $x_1$ and $x_2$ ($\rho_{1,2}$): 0, 0.5, 0.95, -0.5, -0.95
Based on the results of those simulations, how does $\cor(x_1, x_2)$ affect the following?

- The bias of each $\hat{\beta}_j$?
- The variance of each $\hat{\beta}_j$?
- The bias of the standard error of each $\hat{\beta}_j$?
- The bias of the robust standard error of each $\hat{\beta}_j$?

Remember to consider the effects of correlation on *all* the estimates: $\hat{\beta}_1$, $\hat{\beta}_2$, and $\hat{\beta}_3$.

What happens when $\rho = 1$ (or $\rho = -1$)? What assumption is violated?

## Omitted Variable Bias

The population is
$$
\begin{aligned}[t]
Y_i &= 0 + 1 \cdot x_{1,i} + 1 \cdot x_{2,i} + 1 \cdot x_{3,i} + \epsilon_i \\
\epsilon_i &\sim N(0, \sigma^2) \\
\sigma &= 1.7
\end{aligned}
$$

In all simulations, $(x_1, x_2)$ and $(x_2, x_3)$ are uncorrelated.
The correlation between $x_1$ and $x_3$ will vary between simulations.
In other words, the correlation matrix for the $x$ variables is
$$
R =
\begin{bmatrix}
1 & 0 & \rho_{1,3} \\
0 & 1 & 0 \\
\rho_{1,3} & 0 & 1 
\end{bmatrix}
$$

In all simulations, the sample regression will only include $x_1$ and $x_2$:
$$
y_i = \hat\beta_0 + \hat\beta_1 x_{1,i} + \hat\beta_2 x_{2,i} + \hat\epsilon_i
$$
Use $n = 1024$ for all simulations.


```r
sim_omitted_variables <- function(.data, beta, sigma, formula) {
  # X gives the number of observations in the data
  n <- nrow(X)
  # Draw data
  # This creates the X matrix
  X <- model.matrix(~., .data)
  # Create E(y | X)
  yhat <- X %*% beta
  # errors drawn from a normal distribution
  epsilon <- rnorm(n, mean = 0, sd = sigma)
  # actual y's
  y <- yhat + epsilon
  .data$y <- y
  # Estimate model
  mod <- lm(formula, data = .data)
  # Return results
  tidy(mod)
}
```


```r
n <- 1024
mu_X <- c(0, 0, 0)
s_X <- c(1, 1, 1)
rho <- 0
R_X <- matrix(c(1, 0, rho,
                0, 1, 0,
                rho, 0, 1), byrow = TRUE, nrow = 3)
X <- as.data.frame(rmvnorm(n, mu_X, sdcor2cov(s_X, R_X)))
beta <- c(0, 1, 1, 1)
sigma <- 1.7
#sim_omitted_variables(X, beta, sigma, x)
```

## Heteroskedasticity

Consider the case of bivariate regression with a single binary variable, in which each group has a different sample varaince:
$$
\begin{aligned}[t]
y_i &= \beta_0 + \beta_1 x_i + \epsilon_i \\
x_i &\in \{0, 1\} \\
\epsilon_i &\sim 
\begin{cases}
N(0, 1) & \text{if $x = 0$} \\
N(0, \sigma^2) & \text{if $x = 1$}
\end{cases}
\end{aligned}
$$


```r
sim_heteroskedasticity <- function(iter, x, beta, sigma) {
  mu <- cbind(1, x) %*% beta
  # variance varies by value of x
  sigma <- ifelse(as.logical(x), sigma, 1)
  epsilon <- rnorm(n, mean = 0, sd = sigma)
  # actual y's
  y <- yhat + epsilon
  # Estimate model
  mod <- lm(y ~ x)
  # Return results
  tidy(mod)
}
```


Estimate this with $\beta_0 = 0$, $\beta_1 = 1$, and varying values of sample size and $\sigma$? 

How do the following vary with $\sigma^2$ and $n$?

- bias and variance of `\beta_j`
- bias of `\se(\beta)_j`

## Non-random sample

This problem considers what happens when there is a truncated dependent variable.
This is also called sampling on the dependent variable, which is a research design problem not unknown to political science research.[^samplingdv]

The population is a linear normal model with homoskedastic errors.
$$
\begin{aligned}[t]
Y_1 &= \beta_0 + \beta_1 x_{1,i} + \dots + \beta_k x_{k,i} + \epsilon_i \\
\epsilon_i &\sim N(0, \sigma^2)
\end{aligned}
$$
However, in each sample, all $y_i$ which are less than a quantile $q$ are dropped before the regression is estimated.
$$
\begin{aligned}[t]
y_i = \beta_0 + \hat\beta_1 x_{1,i} + \dots + \hat\beta_k x_{k,i} + \hat\epsilon \\ \text{if $y_i \geq \quantile(y, q)$}
\end{aligned}
$$
where $\quantile(y, q)$ is the $q$th quantile of $y$.
For example, if $q = 0.5$, all $y_i$ that are less than the median of $y$ (the bottom 50%) are dropped.

The default value `truncation = 0.5` means all values of $y$ less than the median are dropped before running the regression.

Before running simulations, draw a single sample of a linear normal model with homoskedastic errors.
To do this, you should be able to adapt the code from `sim_lin_normal_truncated`.
Create a scatter plot with the OLS line for all $y$, and a another plot with only those $y$ less than the median of $y$.
How does the OLS line estimated on the truncated data differ from the one estimated on the full data.

Run several simulations with `sim_lin_normal_truncated` and vary the sample size.
How does the sample size affect the following:

- The bias of each $\hat{\beta}_j$?
- The variance of each $\hat{\beta}_j$?

In particular, if we gather more data but $y$ is truncated, does it decrease the bias in $\hat{\beta}$?

[^randomx]: Although the statistical theory of OLS works (thankfully) for random $X$,
    as long as certain conditions are met. See Fox (2nd ed.), Ch 9.6.

[^samplingdv]: See Ashworth, Scott, Joshua D. Clinton, Adam Meirowitz, and Kristopher W. Ramsay. 2008. ``Design, Inference, and the Strategic Logic of Suicide Terrorism.'' *American Political Science Review() 102(02): 269–73. <http://journals.cambridge.org/article_S0003055408080167>



```r
sim_truncated <- function(iter, .data, beta, sigma, q) {
  # X gives the number of observations in the data
  n <- nrow(X)
  # Draw data
  # This creates the X matrix
  X <- model.matrix(~., .data)
  # Create E(y | X)
  yhat <- X %*% beta
  # errors drawn from a normal distribution
  epsilon <- rnorm(n, mean = 0, sd = sigma)
  # actual y's
  y <- yhat + epsilon
  .data$y <- y
  # Remove all observations in which y is above the mean
  .data <- filter(.data, y > quantile(y, q))
  # Estimate model
  mod <- lm(y ~ ., data = .data)
  # Return results
  tidy(mod)
}
```


