import math


def g() -> float:
    """returns gravitational acceleration

    Returns:
        g (float): gravitational acceleration
    """
    return 9.80665


def density_water(T: float) -> float:
    """
    Calculate the density of pure, airless water at a given temperature in degC.
    Formulation in accordance to Jones, F. E.,
    ITS-90 Density of Water Formulation for Volumetric Standards Calibration,
    J Res Natl Inst Stand Technol. 1992 May-Jun; 97(3): 335-340.
    doi: 10.6028/jres.097.013
    Valid between 5 and 40 °C

    Args:
        T (float): temperature [degC]

    Returns:
        rho (float): density [kg/m^3]
    """

    return 999.85308 + 6.32693e-2 * T - 8.523829e-3 * T ** 2 + 6.943248e-5 * T ** 3 \
        - 3.821216e-7 * T ** 4


def surface_tension_water(T: float) -> float:
    """
    Calculate the surface tension value of pure, airless water at a given
    temperature in degC.
    Utilizing the DIPPR106 Equation with parameters from Dortmund Data Bank
    http://ddbonline.ddbst.de/DIPPR106SFTCalculation/DIPPR106SFTCalculationCGI.exe
    Valid between -40 and 370 °C
    
    Args:
        T (float): temperature [degC]

    Returns:
        sigma(float): surface tension [N/m]
    """

    a = 134.15
    b = 1.6146
    c = -2.035
    d = 1.5598
    e = 0
    T_kelv_ref = 647.3

    T_kelv = T + 273.15
    T_rel = T_kelv / T_kelv_ref
    base = (1 - T_rel)
    exponent = (b + c * T_rel + d * T_rel ** 2 + e * T_rel ** 3)

    return a * (base ** exponent) / 1000


def dynamic_viscosity_water(T: float) -> float:
    """
    Calculate the dynamic viscosity of water at a given temperature.
    Utilizing the semi-empirical Vogel-Fulcher-Tammann equation.
    
    Args:
        T (float): temperature [degC]

    Returns:
        mu (float): dynamic viscosity [Pa*s = kg/(m*s)]
    """

    a = 0.02939
    b = 507.88
    c = 149.3

    T_kelv = T + 273.15

    return a * math.exp(b / (T_kelv - c)) / 1000
