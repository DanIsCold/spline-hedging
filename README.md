**7CCSMIEF Coursework II**
**Spline-Based Delta Hedging Optimisation**
This repository contains a Python implementation of a hedging strategy using B-Spline basis functions and PyTorch to optimise hedge ratios under transaction costs. The project compares a baseline discrete Black-Scholes-Merton (BSM) delta hedging strategy against an optimised affine spline model designed to minimise P&L variance.

**Overview**
In the presence of transaction costs, the continuous delta hedging suggested by Black-Scholes is no longer optimal. This program implements an "Affine Spline" model that learns an optimal hedge surface by:
1. Simulating asset price paths using Geometric Brownian Motion
2. Representing the Moneyness and Time-to-Expiry dimensions using B-Spline bases.
3. Optimising the weights of these bases using gradient descent (Adam) to minimise the variance of P&L.
