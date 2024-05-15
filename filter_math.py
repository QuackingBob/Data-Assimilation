import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as co
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

def plot_3D_signal(signals, name, fignum=1):
    """
    Plots multiple 3D signals together

    Parameters:
        signals: list of tuples with signal (x, y, z, "name")
        name: figure title
    """
    fig = plt.figure(fignum, figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    for (xs, ys, zs, label) in signals:
        ax.plot(xs, ys, zs, lw=0.5, label=label)
    ax.set_title(name)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    # plt.show()

def plot_3D_signal_slider(signals, name, fignum=1):
    """
    Plots multiple 3D signals together with a slider to control time

    Parameters:
        signals: list of tuples with signal (x, y, z, "name")
        name: figure title
    """
    fig = plt.figure(fignum, figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    slider_ax = fig.add_axes([0.1, 0.02, 0.8, 0.03])  # Position of the slider

    lines = []  # Store the plotted lines

    for (xs, ys, zs, label) in signals:
        line, = ax.plot(xs, ys, zs, lw=0.5, label=label)
        lines.append(line)
    ax.set_title(name)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    time_slider = Slider(slider_ax, 'Time', 0, len(signals[0][0])-1, valinit=0)  # Initialize slider

    def update(val):
        time_index = int(time_slider.val)
        for line, (xs, ys, zs, label) in zip(lines, signals):
            line.set_data(xs[:time_index+1], ys[:time_index+1])
            line.set_3d_properties(zs[:time_index+1])
        ax.set_title(name)
        fig.canvas.draw_idle()  # Redraw the plot

    time_slider.on_changed(update)
    
    return fig, time_slider

def plot_signals_ensemble(*args, title="Signal Plot", xlabel="Time", ylabel="Amplitude", fignum=3):
    """
    Plots multiple signals on the same figure

    Parameters:
        *args: variable list of tuple (signal, "name")
        title: Title of the plot
        xlabel: Label for the x-axis
        ylabel: Label for the y-axis
    """

    # plt.figure(fignum)
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (signal, name) in enumerate(args):
        rgb_color = co.hsv_to_rgb([i / len(args), 1, 1])
        ax.plot(signal, label=name, color=rgb_color)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    plt.grid(True)
    # plt.show()


def plot_signals(*args, title="Signal Plot", xlabel="Time", fignum=2):
    """
    Plots multiple signals on the same figure

    Parameters:
        *args: variable list of tuple ([signalX1, signalX2, ...], ["name1", "name2", ...], "title") ...
        title: Title of the plot
        xlabel: Label for the x-axis
    """

    # plt.figure(fignum)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for i, (signal_list, names, title) in enumerate(args):
        for j, signal in enumerate(signal_list):
            rgb_color = co.hsv_to_rgb([j / len(signal_list), 1, 1])
            axes[i].plot(signal, label=names[j], color=rgb_color)
            axes[i].set_title(title + " Component")
            axes[i].set_ylabel(title)
            axes[i].legend()

    axes[-1].set_xlabel(xlabel)

    plt.grid(True)
    plt.tight_layout()
    # plt.show()


# def rk4(u_m, u_mp1, t_m, dt):
#         k1 = dt * u_m
#         k2 = dt * f(t + 0.5 * dt, y + 0.5 * k1)
#         k3 = dt * f(t + 0.5 * dt, y + 0.5 * k2)
#         k4 = dt * f(t + dt, y + k3)

#         y += (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
#         t += dt

#         t_values.append(t)
#         y_values.append(y)

#     return t_values, y_values


def generate_true_signal(diff_eq, u_0, dt=0.01, num_steps=10000, noise_sigma=[0.4, 0.4, 0.4], **args):
    """
    Generates an artificial "true" signal for linear or nonlinear 3D turbulent system

    Parameters
        diff_eq: Function to evaluate for state derivative
        u_0: initial states
        dt: integration time
        noise_sigma: std_dev of noise to add to signal to simulate "unknown dynamics"
        *args: arg list for alt params to pass into diff_eq
    """
    assert u_0.shape[0] == 3

    u = np.empty((num_steps + 1, 3))

    u[0] = u_0

    for i in range(num_steps):
        u_dot = diff_eq(u[i], **args)
        noise = np.random.normal(0, noise_sigma, 3)
        u[i+1] = u[i] + u_dot * dt + noise

    return [u[:,0], u[:,1], u[:,2]]

def calculate_noise_covariance(x, true_signal):
    """
    Calculates the observation noise covariance matrix based on observation vector v and true signal.

    x: Estimation vector
    true_signal: True signal vector (contains xs, ys, zs)
    """
    residuals = np.array(x) - np.array(true_signal)
    covariance_matrix = np.cov(residuals, rowvar=False, bias=True)
    
    return covariance_matrix

def kalman_filter(F, G, Q, R, u, P, v, N, x=None, B=None, t=0, dt=0.01, Fisfunc=False, Bisfunc=False):
    """
    F: State transition matrix
    G: Observation matrix
    Q: Process noise covariance matrix
    R: Measurement noise covariance matrix
    u: Initial state estimate
    P: Current covariance estimate
    v: Measurement vector
    N: Dimension of state space
    x: The control vector (if None, not used)
    B: Control matrix (if None, not used)
    t: current time
    dt: simulation time step
    Fisfunc: if F is instead a function F(u, t, dt)
    Bisfunc: if B is instead a function B(u, t, dt)

    notation:
    mp1 is m+1
    g is | or given
    so mp1gm is m+1|m
    """

    # prediction step
    F = F if not Fisfunc else F(u, t, dt) 
    u_mp1gm = F @ u
    if x is not None and B is not None:
        u_mp1gm += B @ x if not Bisfunc else B(u, t, dt) @ x
    P_mp1gm = F @ P @ F.T + Q

    # analysis step
    K = P_mp1gm @ G.T @ np.linalg.inv(G @ P_mp1gm @ G.T + R)
    z = v
    u_mp1gmp1 = u_mp1gm + K @ (z - G @ u_mp1gm)
    P_mp1gmp1 = (np.eye(N) - K @ G) @ P_mp1gm

    return u_mp1gm, P_mp1gm, u_mp1gmp1, P_mp1gmp1

def simulate_kalman_filter(F, G, Q, R, u, P, measurements, N, controls=None, B=None, t_0=0, dt=0.01, Fisfunc=False, Bisfunc=False):
    """
    F: State transition matrix or function F(u, t, dt) to return matrix
    G: Observation matrix
    Q: Process noise covariance matrix
    R: Measurement noise covariance matrix
    u: Initial state estimate
    P: Initial covariance estimate
    measurements: List of measurement vectors
    N: Number of states
    controls: List of control vectors (if None, not used)
    B: Control matrix or function B(u, t, dt) to return matrix (if None, not used)
    t_0: initial time (if not needed, default 0)
    dt: simulation time step (default 0.01)
    Fisfunc: if F is instead a function F(u, t, dt) (default False)
    Bisfunc: if B is instead a function B(u, t, dt) (default False)
    """
    num_measurements = measurements.shape[0]
    
    predicted_states = np.zeros((num_measurements, N))
    updated_states = np.zeros((num_measurements, N))

    predicted_covariances = np.zeros((num_measurements, N, N))
    updated_covariances = np.zeros((num_measurements, N, N))
    
    # init states
    u_predicted = u
    P_predicted = P
    t = t_0
    
    for i in range(num_measurements):
        # apply kalman filter
        x = None
        if controls is not None:
            x = controls[i]
        u_predicted, P_predicted, u_updated, P_updated = kalman_filter(F, G, Q, R, u_predicted, P_predicted, measurements[i], N, x, B, t, dt, Fisfunc, Bisfunc)
        
        # record and update states
        predicted_states[i] = u_predicted
        updated_states[i] = u_updated
        predicted_covariances[i] = P_predicted
        updated_covariances[i] = P_updated

        u_predicted = u_updated
        P_predicted = P_updated

        t += dt
    
    return predicted_states, updated_states, predicted_covariances, updated_covariances

def disp_mat(mats, names):
    for (mat, n) in zip(mats, names):
        print(f"{n}={mat}")