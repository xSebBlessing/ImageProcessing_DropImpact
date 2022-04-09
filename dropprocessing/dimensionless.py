import numpy as np

def bond_number(d: float, g: float, rho: float, sigma: float) -> float:
    """Calculates the Bond number for the falling drop

    Args:
        d (float): diameter of falling drop [m]
        g (float): gravitational acceleration [m/s^2]
        rho (float): density of drop material [kg/m^3]
        sigma (float): surface tension of drop material [N/m = kg/s^2]

    Returns:
        float: Bond number of the drop [1]
    """
    return rho * (d/2)**2 * g / sigma


def weber_number(d: float, u: float, rho: float, sigma: float) -> float:
    """Calculates the Weber number for the falling drop

    Args:
        d (float): diameter of falling drop [m]
        u (float): falling velocity of the drop [m/s]
        rho (float): density of drop material [kg/m^3]
        sigma (float): surface tension of drop material [N/m = kg/s^2]

    Returns:
        float: Weber number of the drop [1]
    """
    return d/2 * u**2 * rho / sigma


def capillary_number(u: np.ndarray, mu: float,
                     sigma: float) -> np.ndarray:
    """Calculates the capillary number of each frame pair from the spreading
       velocity

    Args:
        u (np.ndarray): spreading velocity for each frame pair [m/s]
        mu (float): viscosity of drop material [Pa*s = kg/(m*s)]
        sigma (float): surface tension of drop material [N/m = kg/s^2]

    Returns:
        np.ndarray: Capillary number for each frame pair [1]
    """
    return mu * u / sigma


def spreading_factor(d0: float, x_left: np.ndarray, x_right: np.ndarray,
                     px_p_m: float) -> np.ndarray:
    """Calculates spreading factor for each frame with respect to the falling
       drop diameter

    Args:
        d0 (float): diameter of falling drop [m]
        x_left (np.ndarray): position of left end of contact line [px]
        x_right (np.ndarray): position of left end of contact line [px]
        px_p_m (float): physical spatial resolution derived from calibration
                        [px/m]

    Returns:
        np.ndarray: Spreading factor for each frame [1]
    """
    return (x_right - x_left) / (d0 * px_p_m)
