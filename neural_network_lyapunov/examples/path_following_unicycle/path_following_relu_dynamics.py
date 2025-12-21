import torch
import numpy as np
import neural_network_lyapunov.utils as utils


def build_inv_relu_network(d_min, d_max, nb_points, dtype=torch.float64):
    """
    Builds a ReLU network approximating g(d) = 1 / (1 - d)
    over d ∈ [d_min, d_max].
    """
    d = np.linspace(d_min, d_max, nb_points)
    v = 1.0 / (1.0 - d)

    delta = (d_max - d_min) / (nb_points - 1)

    A1 = np.tile([1.0], (nb_points - 1, 1))
    b1 = d[:-1]

    slopes = np.diff(v) / delta
    A2 = slopes.reshape(1, -1)
    v_offset = v[0]

    A1 = torch.tensor(A1, dtype=dtype)
    b1 = torch.tensor(-b1, dtype=dtype)
    A2 = torch.tensor(A2, dtype=dtype)

    net = utils.setup_relu(
        (1, nb_points - 1, 1),
        negative_slope=0.0,
        bias=True,
        dtype=dtype,
    )

    net[0].weight.data[:] = A1
    net[0].bias.data[:] = b1

    net[2].weight.data[:] = A2
    net[2].bias.data[:] = v_offset

    return net


def build_trig_relu_matrices(nb_points: int, func: str):
    assert func in ("sin", "cos")

    theta = np.linspace(-np.pi, np.pi, nb_points)
    theta_max = np.pi
    nb_segments = nb_points - 1
    theta_delta = 2 * theta_max / nb_segments

    if func == "sin":
        v = np.sin(theta)
        sign = -1.0
    else:
        v = np.cos(theta)
        sign = +1.0

    A1a = np.tile([theta_delta, 0.0], (nb_segments, 1))
    A1 = np.vstack([A1a, A1a])

    A1b = np.tile([theta_delta, -1.0], (nb_segments, 1))
    A1b_sym = np.tile([theta_delta, +1.0], (nb_segments, 1))
    A1b = np.vstack([A1b, A1b_sym])

    b1b = theta_delta * np.arange(nb_segments) - theta_max
    b1b = np.hstack([b1b, b1b])

    A3_half = np.diff(v) / theta_delta / 2
    A3 = np.hstack([A3_half, sign * A3_half]).reshape(1, -1)

    v_offset = v[0]

    return A1, A1b, b1b, A3, v_offset


def build_trig_relu_network(nb_points, func, dtype=torch.float64):
    A1, A1b, b1b, A3, v_offset = build_trig_relu_matrices(nb_points, func)

    A1 = torch.tensor(A1, dtype=dtype)
    A1b = torch.tensor(A1b, dtype=dtype)
    b1b = torch.tensor(b1b, dtype=dtype)
    A3 = torch.tensor(A3, dtype=dtype)

    n_hid = A1.shape[0]

    net = utils.setup_relu(
        (2, n_hid, n_hid, 1),
        negative_slope=0.0,
        bias=True,
        dtype=dtype,
    )

    net[0].weight.data[:] = A1 - A1b
    net[0].bias.data[:] = -b1b

    net[2].weight.data[:] = torch.eye(n_hid, dtype=dtype)
    net[2].bias.data.zero_()

    net[4].weight.data[:] = A3
    net[4].bias.data.zero_()
    net[4].weight.data[:, 0] += v_offset

    return net


def build_path_following_dynamics_relu(nb_points, v, dtype=torch.float64):
    """
    Returns φ_dyn(x,u) where
    x = [d_e, theta_e], u = [u]
    output = [ḋ_e, θ̇_e]
    """

    sin_net = build_trig_relu_network(nb_points, "sin", dtype)
    cos_net = build_trig_relu_network(nb_points, "cos", dtype)

    n_trig = sin_net[0].out_features

    net = utils.setup_relu(
        (3, n_trig, n_trig, 2),
        negative_slope=0.0,
        bias=True,
        dtype=dtype,
    )

    # Layer 1: feed theta_e only
    net[0].weight.data.zero_()
    net[0].weight.data[:, 1] = 1.0
    net[0].bias.data.zero_()

    # Layer 2
    net[2].weight.data[:] = torch.eye(n_trig, dtype=dtype)
    net[2].bias.data.zero_()

    # Output layer
    net[4].weight.data.zero_()
    net[4].bias.data.zero_()

    # ḋ_e = v sin(theta_e)
    net[4].weight.data[0, :] = v * sin_net[4].weight.data

    # θ̇_e = u − v cos(theta_e)
    inv_net = build_inv_relu_network(
        d_min=-0.8,
        d_max=0.8,
        nb_points=17,
        dtype=dtype,
    )

    # θ̇_e = u − v * cos(theta_e) * inv(1-d_e)
    net[4].weight.data[1, :] = -v * (cos_net[4].weight.data * inv_net[2].weight.data)
    net[4].weight.data[1, 2] = 1.0

    return net
