import numpy as np
from filter_math import *
from plotting_helpers import *
import matplotlib.pyplot as plt


def lorenz(u, sigma=10, rho=28, beta=8/3):
    """
    Evaluates lorenz attractor differential equations
    """
    x, y, z = u
    x_dot = sigma * (y - x)
    y_dot = x * (rho - z) - y
    z_dot = x * y - beta * z
    return np.array([x_dot, y_dot, z_dot])

# simulation parameters
dt = 0.01
num_steps = 500

# lorenz attractor params
sigma = 10
rho = 28
beta = 8/3

# initial states and ground truth signal
u_0 = np.array([0., 1., 1.05])
xs, ys, zs = generate_true_signal(lorenz, u_0, dt, num_steps, noise_sigma=[0.001, 0.01, 0.5], sigma=sigma, rho=rho, beta=beta)

# combine true state into one array 
true_signal = np.array([xs, ys, zs]).T
num_samples, N = true_signal.shape

# u_0_dot = lorenz(u_0, sigma=10, rho=28, beta=8/3)
# [10,-1,-2.8]
# approx state transition matrix F:
"""
linear approx: 
[[10, 0, 0],
 [0, -1, 0],
 [0, 0, -2.8]]

hold u[0] const and write as linear equation:
[[-sigma, sigma, 0],
 [rho, -1, -u_0[0]],
 [0, u_0[0], -beta]
]

F' = I + Fdt

du = Fu + Bn
du/dt = F(u, t, dt)u
du = F(u, t, dt)udt
u = u + du = Iu + Fdtu
u = (I + Fdt)u 
F' = (I + Fdt)
"""
# F = np.array([[-sigma, sigma, 0], [rho, -u_0[0], -1], [0, u_0[0], -beta]])
def calcF(u, t, dt):
    F = np.array([[-sigma, sigma, 0], [rho, -1, -u[0]], [0, u[0], -beta]])
    F_prime = np.eye(u.shape[0]) + dt * F
    return F_prime
F_0 = calcF(u_0, 0, dt)
print(f"Approx Initial State Derivative: {F_0 @ u_0}")

# initial state estimate
u = u_0

# initial covariance estimate
P = 0.01 * np.eye(3)

# process noise covariance matrix 
# Q = 1e1 * np.eye(3)
# simulate model only prediction for n steps
u_pred = np.empty((num_samples, N))
u_pred[0] = u_0
for i in range(num_samples - 1):
    u_pred[i+1] = calcF(u_pred[i], 0, dt) @ u_pred[i]
Q = calculate_noise_covariance(u_pred, true_signal) 

# create measurement vectors by adding noise to true state to simulate noisy observation
sigmas = np.array([2, 4, 0.3]).T
measurement_noise = np.random.normal(scale=sigmas, size=(num_samples, N))
measurements = true_signal + measurement_noise

# observation matrix
G = np.eye(3)

# get measurement noise covariance matrix 
# R = np.diag([0.001, 0.001, 0.001])
# R = np.diag(sigmas ** 2) # this is what it really is
R = calculate_noise_covariance(measurements, true_signal) # this is to simulate trying to find this

disp_mat([F_0, G, Q, u, P, R], [i for i in "FGQuPR"])

predicted_states, updated_states, _, _ = simulate_kalman_filter(calcF, G, Q, R, u, P, measurements, N, t_0=0, dt=dt, Fisfunc=True)
xp, yp, zp = (predicted_states[:,0], predicted_states[:,1], predicted_states[:,2]) # model prior from prediction stage
xk, yk, zk = (updated_states[:,0], updated_states[:,1], updated_states[:,2]) # kalman filter output
xv, yv, zv = (measurements[:,0], measurements[:,1], measurements[:,2]) # observations only
xm, ym, zm = (u_pred[:,0], u_pred[:,1], u_pred[:,2]) # only model (no use of kalman filter output)

fig1, slider1, leg1 = plot_3D_signal_slider([
        (xs, ys, zs, "truth"), 
        (xp, yp, zp, "model prior"),
        (xv, yv, zv, "observations"),
        (xk, yk, zk, "filtered signal"),
    ], "Lorenz Attractor")
leg1 = InteractiveLegend(legend=leg1)

leg2 = plot_signals(
    ([xs, xp, xv, xk], ["true", "model prior", "obs", "filter"], "X"), 
    ([ys, yp, yv, yk], ["true", "model prior", "obs", "filter"], "Y"), 
    ([zs, zp, zv, zk], ["true", "model prior", "obs", "filter"], "Z"),
    title="Signal Plot",
    xlabel="Time",
)
for i, l in enumerate(leg2):
    leg2[i] = InteractiveLegend(legend=l)

fig3, slider3, leg3 = plot_3D_signal_slider([
        (xs, ys, zs, "truth"), 
        (xm, ym, zm, "model only"),
    ], "Lorenz Attractor", 3)
leg3 = InteractiveLegend(legend=leg3)

leg4 = plot_signals(
    ([xs, xm], ["true", "model only"], "X"), 
    ([ys, ym], ["true", "model only"], "Y"), 
    ([zs, zm], ["true", "model only"], "Z"),
    title="Signal Plot",
    xlabel="Time",
    fignum=4
)
for i, l in enumerate(leg4):
    leg4[i] = InteractiveLegend(legend=l)

plt.show()