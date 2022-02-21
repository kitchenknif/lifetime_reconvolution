# lifetime_reconvolution
Reconvolution fitting for lifetime measurements

Inspired by [DLTReconvolution](https://github.com/dpscience/DLTReconvolution)

## Fitting procedure
0. Generate test data
   1. Generate Gaussian IRF
   2. Add noise to IRF
   3. Generate Bi-exponential decay
   4. Convolve IRF and Bi-exponential decay
   5. Add noise to convolved IRF and decay
1. Create gaussian IRF model & fit model to noisy IRF
2. Create convolved IRF and bi-exponential model 
3. Use parameters from IRF fitting and expected decay parameters as initial fitting, fit model to noisy decay data
4. Plot results

## Dependencies
- numpy
- lmfit
- matplotlib
