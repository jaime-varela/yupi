"""
This contains filtering functions for the trajectories.
"""

from typing import Optional
from typing import Callable
import numpy as np

from yupi.trajectory import Trajectory

from yupi.trajectory import _THRESHOLD, Trajectory

def exp_convolutional_filter(
    traj: Trajectory, gamma: float, new_traj_id: Optional[str] = None
):
    """
    Returns a smoothed version of the trajectory `traj`
    by taking a weighted average over past values.

    Parameters
    ----------
    traj : Trajectory
        Input trajectory.
    gamma : float
        Inverse of the characteristic time window of
        the average.
    new_traj_id : Optional[str]
        New trajectory ID. By default None.

    Returns
    -------
    Trajectory
        Smoothed trajectory.
    """

    track_origin = traj.r[0]
    r = (traj-track_origin).r
    dt = np.ediff1d(traj.t)
    new_r = np.zeros_like(r)
    for i in range(len(traj) - 1):
        new_r[i + 1] = new_r[i] - gamma * (new_r[i] - r[i]) * dt[i]

    smooth_traj = Trajectory(
        points=new_r, t=traj.t, traj_id=new_traj_id, diff_est=traj.diff_est
    )
    return smooth_traj + track_origin



def exp_moving_average_filter(
        traj: Trajectory, alpha: float,tau:Optional[float] = None, new_traj_id: Optional[str] = None
):
    """
    Returns a smoothed version of the trajectory `traj`
    using the exponential moving average defined as

    s(0) = x(0)
    s(t_n) = alpha x(t_{n-1})  + (1-alpha) s(t_{n-1})

    If the the trajectory times are non-uniform then tau must be provided. The non-uniform time filter is
    computed as

    s(0) = x(0)
    alpha(t_n) = 1 - exp(-(t_n - t_{n-1}) / tau))
    s(t_n) = alpha(t_n) x(t_{n-1})  + (1-alpha(t_n)) s(t_{n-1})
    
    Parameters
    ----------
    traj : Trajectory
        Input trajectory.
    alpha : float
        Exponential smoothing paramter.
    tau: float [optional, default=None]
        Smoothing factor that must be provided if the trajectory timeseries is non-uniform.
    new_traj_id : Optional[str]
        New trajectory ID. By default None.

    Returns
    -------
    Trajectory
        Smoothed trajectory.
    """
    data = traj.r
    times = traj.t
    if tau is None and abs(traj.dt_std - 0) > _THRESHOLD:
        raise ValueError("All trajectories must be uniformly time spaced if tau is not provided")        
    n_times, _ = data.shape
    ema = np.zeros_like(data)    
    ema[0] = data[0]
    # The uniform time smoother can likely be simplified with a convolution
    # using scipy but I didn't want to bring in that dependency here
    for i in range(1, n_times):
        dt = times[i] - times[i - 1]
        if tau is not None:
            alpha = 1 - np.exp(-dt / tau)  # Adaptive smoothing factor
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    
    smooth_traj = Trajectory(
        points=ema, t=traj.t, traj_id=new_traj_id, diff_est=traj.diff_est
    )
    return smooth_traj

# The Kalman filters follow the convention of section 3.3.2 of 
# "State Estimation For Robotics" by Thomas D. Barafoot with v_k=0.
# These functions have been left general to incorporate additional dynamics,
#  beyond the random acceleration model.

def _predicted_covariance(previous_covariance: np.ndarray,
                          transition_matrix: np.ndarray,
                          process_covariance):
    predicted_state_covariance = (
        np.dot(transition_matrix, np.dot(previous_covariance, transition_matrix.T))
        + process_covariance
    )
    return predicted_state_covariance

def _predict_next_state(previous_state,transition_matrix):
    predicted_state_mean = (
        np.dot(transition_matrix, previous_state)
    )
    return predicted_state_mean

def _kalman_gain(predicted_covariance,observation_matrix,observation_covariance):
    predicted_observation_covariance = (
        np.dot(
            observation_matrix,
            np.dot(predicted_covariance, observation_matrix.T),
        )
        + observation_covariance
    )
    kalman_gain = np.dot(
        predicted_covariance,
        np.dot(observation_matrix.T, np.linalg.pinv(predicted_observation_covariance)),
    )
    return kalman_gain

def _state_corrector(observation,kalman_gain,predicted_state,observation_matrix):
    predicted_observation_mean = (
            np.dot(observation_matrix, predicted_state)
    )
    corrected_state_mean = predicted_state + np.dot(
            kalman_gain, observation - predicted_observation_mean
    )
    return corrected_state_mean


def _covariance_corrector(predicted_covariance,kalman_gain,observation_matrix):
    corrected_state_covariance = predicted_covariance - np.dot(
            kalman_gain, np.dot(observation_matrix, predicted_covariance)
        )
    return corrected_state_covariance


def _kalman_filter_from_generators(n_timesteps:int,
        initial_state_mean: np.ndarray,
        initial_state_covariance: np.ndarray,
        get_observation: Callable[[int], np.ndarray],
        get_observation_matrix: Callable[[int], np.ndarray],
        get_transition_matrix: Callable[[int], np.ndarray],
        get_process_covariance: Callable[[int], np.ndarray],
        get_measurement_covariance: Callable[[int], np.ndarray]
    ):
    
    n_dim_state = initial_state_mean.shape[0]
    filtered_state_means = np.zeros((n_timesteps, n_dim_state))
    filtered_state_covariances = np.zeros((n_timesteps, n_dim_state, n_dim_state))

    # Might be worth it to not pre-allocate these matrices
    previous_state_covariance = initial_state_covariance
    previous_state = initial_state_mean

    for t in range(n_timesteps):
        transition_matrix = get_transition_matrix(t)
        process_covariance = get_process_covariance(t)
        predicted_covariance = _predicted_covariance(previous_state_covariance,
                                                     transition_matrix,
                                                     process_covariance)
        predicted_state = _predict_next_state(previous_state,transition_matrix)

        observation_matrix = get_observation_matrix(t)
        observation_covariance = get_measurement_covariance(t)
        kalman_gain = _kalman_gain(predicted_covariance,
                                   observation_matrix,
                                   observation_covariance)
        
        observation = get_observation(t)
        filtered_state = _state_corrector(observation,
                                          kalman_gain,
                                          predicted_state,
                                          observation_matrix)

        filtered_covariance = _covariance_corrector(predicted_covariance,
                                                    kalman_gain,
                                                    observation_matrix)

        filtered_state_means[t] = filtered_state
        filtered_state_covariances[t] = filtered_covariance
        previous_state = filtered_state
        previous_state_covariance = filtered_covariance

    # maybe return the Kalman gains as well?
    return filtered_state_means, filtered_state_covariances


def kalman_random_acceleration_filter(
        traj: Trajectory, sigma_a: float,
        initial_state_mean : np.ndarray,
        new_traj_id: Optional[str] = None
):
    # TODO: set up the transition matrices, observation matrices, etc for
    # the random acceleration Kalman model
    return False



