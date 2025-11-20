"""
Constants for the Epydemics package.

This module contains all the constant values used throughout the package,
including rate names, compartment identifiers, and analysis parameters.

The constants are organized by their usage domain and properly typed
for better code clarity and IDE support.
"""

from typing import Final, List

# Epidemic model rate names
# These are the three key rates in the SIRD model
RATIOS: Final[List[str]] = ["alpha", "beta", "gamma"]

# Logit-transformed rate names for VAR modeling
# Each corresponds to logit(rate) transformation for time series analysis
LOGIT_RATIOS: Final[List[str]] = ["logit_alpha", "logit_beta", "logit_gamma"]

# SIRD compartment identifiers
# A=Affected, C=Cases, S=Susceptible, I=Infected, R=Recovered, D=Deaths
COMPARTMENTS: Final[List[str]] = ["A", "C", "S", "I", "R", "D"]

# Forecasting confidence levels
# Used for uncertainty quantification in Monte Carlo simulations
FORECASTING_LEVELS: Final[List[str]] = ["lower", "point", "upper"]

# Central tendency calculation methods
# Statistical measures for summarizing simulation results
CENTRAL_TENDENCY_METHODS: Final[List[str]] = ["mean", "median", "gmean", "hmean"]

# Visualization constants for plotting and display
COMPARTMENT_LABELS: Final[dict] = {
    "A": "Active",
    "C": "Confirmed",
    "S": "Susceptible",
    "I": "Infected",
    "R": "Recovered",
    "D": "Deaths",
}

METHOD_NAMES: Final[dict] = {
    "mean": "Mean",
    "median": "Median",
    "gmean": "Geometric Mean",
    "hmean": "Harmonic Mean",
}

METHOD_COLORS: Final[dict] = {
    "mean": "blue",
    "median": "orange",
    "gmean": "green",
    "hmean": "purple",
}

# Maintain original lowercase names for backward compatibility
ratios = RATIOS
logit_ratios = LOGIT_RATIOS
compartments = COMPARTMENTS
forecasting_levels = FORECASTING_LEVELS
central_tendency_methods = CENTRAL_TENDENCY_METHODS
compartment_labels = COMPARTMENT_LABELS
method_names = METHOD_NAMES
method_colors = METHOD_COLORS

# Export all constants for easy importing
__all__ = [
    "RATIOS",
    "LOGIT_RATIOS",
    "COMPARTMENTS",
    "FORECASTING_LEVELS",
    "CENTRAL_TENDENCY_METHODS",
    "COMPARTMENT_LABELS",
    "METHOD_NAMES",
    "METHOD_COLORS",
    # Backward compatibility exports
    "ratios",
    "logit_ratios",
    "compartments",
    "forecasting_levels",
    "central_tendency_methods",
    "compartment_labels",
    "method_names",
    "method_colors",
]
