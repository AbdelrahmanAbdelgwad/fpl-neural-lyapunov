#%%
import numpy as np
from scipy.linalg import solve_continuous_are, inv
import matplotlib.pyplot as plt

# Define system dynamics
def unicycle_dynamics(state, control, dt):
    x, y, theta, omega = state
    v, u = control
    dx = v * np.cos(theta) * dt
    dy = v * np.sin(theta) * dt
    dtheta = omega * dt
    domega = u * dt
    return np.array([x + dx, y + dy, theta + dtheta, omega+domega])

# Linearize the dynamics around a current state and control input
def linearize_dynamics(state, control):
    theta = state[2]
    omega = state[3]
    v = control[0]
    A = np.array([[0, 0, -v * np.sin(theta), 0],
                  [0, 0,  v * np.cos(theta), 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]])
    B = np.array([[np.cos(theta), 0],
                  [np.sin(theta), 0],
                  [0, 0],
                  [0, 1]])
    return A, B

# Define LQR cost matrices
Q = np.diag([10, 10, 1, 1])
R = np.diag([1, 1])

# Compute LQR gain
def lqr(A, B, Q, R):
    P = solve_continuous_are(A, B, Q, R)
    K = inv(R) @ B.T @ P
    return K

# Generate a simple trajectory
def generate_trajectory(timesteps, dt):
    traj = []
    for t in range(timesteps):
        # x = 0.1 * t * dt
        # y = 0.1 * np.sin(0.1 * t * dt)
        # theta = 0.1 * t * dt
        x = 0.1 * t * dt
        y = 0.1 * t * dt
        theta = 0.2
        omega = 0.1
        traj.append(np.array([x, y, theta, omega]))
    return np.array(traj)

# Simulate the system with LQR control for trajectory tracking
def simulate_lqr_tracking(x0, trajectory, timesteps, dt):
    x = x0
    states = [x]
    for t in range(timesteps - 1):
        x_ref = trajectory[t]
        u_ref = [0.1, 0]  # Nominal control input
        A, B = linearize_dynamics(x, u_ref)
        K = lqr(A, B, Q, R)
        u = -K @ (x - x_ref) + u_ref
        x = unicycle_dynamics(x, u, dt)
        states.append(x)
    return np.array(states)

# Simulate point stabilization
def simulate_lqr_point_stabilization(x0, x_target, timesteps, dt):
    x = x0
    states = [x]
    for t in range(timesteps):
        u_ref = [0.1, 0]  # Small nominal control input for stabilization
        A, B = linearize_dynamics(x, u_ref)
        K = lqr(A, B, Q, R)
        u = -K @ (x - x_target) + u_ref
        x = unicycle_dynamics(x, u, dt)
        states.append(x)
    return np.array(states)

# Generate the trajectory
trajectory = generate_trajectory(1000, 0.01)

# Simulate trajectory tracking
x0 = np.array([0, 0, 0, 0])
timesteps = 1000
dt = 0.01
states_tracking = simulate_lqr_tracking(x0, trajectory, timesteps, dt)

# Plot the results for trajectory tracking
plt.figure()
plt.plot(trajectory[:, 0], trajectory[:, 1], 'g--', label='Reference trajectory')
plt.plot(states_tracking[:, 0], states_tracking[:, 1], 'b', label='LQR controlled trajectory')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Unicycle Model with LQR Control (Trajectory Tracking)')
plt.show()

# Desired final state
x_target = np.array([1, 1, 0, 0])

# Simulate point stabilization
states_stabilization = simulate_lqr_point_stabilization(x0, x_target, timesteps, dt)

# Plot the results for point stabilization
plt.figure()
plt.plot(states_stabilization[:, 0], states_stabilization[:, 1], 'b', label='LQR controlled trajectory')
plt.scatter(x_target[0], x_target[1], color='r', marker='x', label='Target point')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Unicycle Model with LQR Control (Point Stabilization)')
plt.show()



# %%
