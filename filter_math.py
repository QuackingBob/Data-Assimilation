import numpy as np

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
    """
    Shortcut function to print multiple matrices simulatneously
    
    Parameters:
        mats: list of matrices [m1, m2, ...]
        names: list of names to print ["Mat1", ...]
    """
    for (mat, n) in zip(mats, names):
        print(f"{n}={mat}")


def is_controllable(V, F):
    """
    This function checks the rank condition for controllability

    Parameters:
        V: The control matrix
        F: The state transition matrix
    """
    N = F.shape[0]
    rank_cond = True
    for i in range(N):
        if np.linalg.matrix_rank(np.linalg.matrix_power(F, i) @ V) != N:
            rank_cond = False
    return rank_cond


def is_observable(G, F):
    """
    This function checks the rank condition for observability

    Parameters:
        G: The observation matrix
        F: The state transition matrix
    """
    N = F.shape[0]
    G = G.T
    F = F.T
    rank_cond = True
    for i in range(N):
        if np.linalg.matrix_rank(np.linalg.matrix_power(F, i) @ G) != N:
            rank_cond = False
    return rank_cond

def is_stable(V, F, G):
    """
    Function to check stability of the system based on the given conditions
    True -> stable
    1 -> ||F|| < 1
    2 -> observable but not controllable
    3 -> both observable and controllable (optimal K_m exists for all t_m)

    Parameters:
        V: control matrix
        F: state transition matrix
        G: observation model matrix
    """
    
    # condition 1: Norm of F < 1
    if np.linalg.norm(F) < 1:
        return (True, 1)

    # condition 2: Observability
    if is_observable(G, F):
        if is_controllable(V, F):
            return (True, 3)
        return (True, 2)

    return (False, -1)


def alt_stability_condition(F, G, K_inf):
    """
    Check the asymptotic kalman gain stability condition

    Parameters:
        F: state transition matrix
        G: observation matrix
        K_inf: asymptotic kalman gain matrix
    """
    N = F.shape[0]
    return np.linalg.norm(F @ (np.eye(N) - K_inf @ G)) < 1

def calc_rmse(truth, x):
    """
    Calculate RMSE for signal (Accross N)

    Parameters:
        truth: NxM np array
        x: NxM np array
    """

    error = truth - x
    
# def generate_report(F, G, Q, R, u, P, v, N, K)