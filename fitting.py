import matplotlib.pyplot as plt
import numpy as np

from lmfit import CompositeModel, Model
from lmfit.lineshapes import gaussian, exponential

#
# Convolution function
#
def convolve(a, b):
    A = np.fft.fft(a)
    B = np.fft.fft(b)
    convAB = np.real(np.fft.ifft(A*B))
    return convAB

#
# Test data for fitting
#
x = np.linspace(0, 100, 2001)  # Time
y_exp = exponential(x, amplitude=10, decay=3) \
        + exponential(x, amplitude=1, decay=10)  # Bi-exponential decay data

#
# Test data IRF for convolution
#
gamp = 1
gsig = 0.1
y_gauss = gaussian(x, amplitude=gamp, center=10, sigma=gsig)  # Gaussian IRF
gamp = 1/np.sum(y_gauss)

y = convolve(y_exp, y_gauss)  # Convolved data

y_real = y
y_real = np.divide(y_real, np.max(y_real))  # Normalize data to 1


#
# Add noise to complicate fitting
#
np.random.seed(0)
y = np.abs(y + np.random.normal(scale=0.06, size=x.size))  # Noisy decay data
y = np.divide(y, np.max(y))  # Normalize data to 1

y_gaus_noise = np.abs(y_gauss + np.random.normal(scale=0.05, size=x.size))  # Noisy IRF data


#
# Fitting procedure
#

#
# Fit IRF
#
mod = Model(gaussian)  # gaussian model

#
pars = mod.make_params(amplitude=0.1, center=15, sigma=1)  # IRF Parameters

pars["amplitude"].vary = True
pars["center"].vary = True
pars["sigma"].vary = True

# fit this model to data array y
result = mod.fit(y_gaus_noise, params=pars, x=x)

# Fit parameters for IRF
gamp_fit = result.best_values["amplitude"]
gsig_fit = result.best_values["sigma"]
gcenter_fit = result.best_values["center"]

print(result.fit_report())

#
# Plot IRF Fitting
#
fig, axes = plt.subplots(2, 1, figsize=(4,8))

axes[0].plot(x, np.log(y_gaus_noise), 'bo', label="noisy data")
axes[0].plot(x, np.log(y_gauss), 'g+', label="data without noise")
axes[0].plot(x, np.log(result.init_fit), 'k--', label='initial fit')
axes[0].plot(x, np.log(result.best_fit), 'r-', label='best fit')
axes[0].legend()
axes[0].set_ylim([-6, 2])
axes[0].set_xlim([5, 25])

axes[0].set_ylabel("log(counts)")
axes[0].set_xlabel("time")
axes[0].set_title("IRF Fitting")

#
# Fit Decay
#

# Create Composite Model using the custom convolution operator
mod = CompositeModel(
        Model(exponential, prefix="a_") + Model(exponential, prefix="b_"),  # Biexponential decay
        Model(gaussian),  # IRF
            convolve)

#
pars = mod.make_params(amplitude=gamp_fit, center=gcenter_fit, sigma=gsig_fit,  # IRF parameters from fitting
                       a_amplitude=1, a_decay=1,  # Exponent 1 parameters
                       b_amplitude=1, b_decay=2)  # Exponent 2 parameters

pars["amplitude"].vary = True
pars["center"].vary = True
pars["sigma"].vary = False

fitWeightingSpec = 1.0/np.sqrt(y+1e-15)  # Fit weighting, 1e-15 added to prevent divide by zero


# fit this model to data array y
result = mod.fit(y, params=pars, x=x, weights=fitWeightingSpec)

print(result.fit_report())

#
# Plot results
#

axes[1].plot(x, np.log(y), 'bo', label="noisy data")
axes[1].plot(x, np.log(y_real), 'g+', label="data without noise")
axes[1].plot(x, np.log(result.init_fit), 'k--', label='initial fit')
axes[1].plot(x, np.log(result.best_fit), 'r-', label='best fit')
axes[1].legend()
axes[1].set_ylim([-10, 1])

axes[1].set_ylabel("log(counts)")
axes[1].set_xlabel("time")
axes[1].set_title("Decay Fitting")


plt.tight_layout()
plt.show()
