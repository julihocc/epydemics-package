"""
Visualization functions for epidemiological models and forecasts.

This module provides functions to visualize epidemic data, model results,
and forecasts with various plotting options.
"""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd

from ..core.constants import (
    central_tendency_methods,
    compartment_labels,
    method_colors,
    method_names,
)


def visualize_results(
    results: Dict[str, Any],
    compartment_code: str,
    testing_data: Optional[pd.DataFrame] = None,
    log_response: bool = False,
    alpha: float = 0.3,
) -> None:
    """
    Visualize forecast results for a specific compartment.

    Args:
        results: Dictionary containing forecast results for each compartment
        compartment_code: Code of the compartment to visualize (e.g., 'C', 'D', 'I')
        testing_data: Optional DataFrame with actual values for comparison
        log_response: Whether to use logarithmic scale for y-axis
        alpha: Transparency for simulation paths

    Raises:
        KeyError: If compartment_code is not found in results
    """
    if compartment_code not in results:
        raise KeyError(f"Compartment '{compartment_code}' not found in results")

    compartment = results[compartment_code]

    # Plot individual simulation paths with low alpha
    for alpha_key in compartment.keys():
        if alpha_key not in central_tendency_methods:  # Skip central tendency methods
            for beta_key in compartment[alpha_key].keys():
                for gamma_key in compartment[alpha_key][beta_key].keys():
                    simulation = compartment[alpha_key][beta_key][gamma_key]
                    plt.plot(
                        simulation.index,
                        simulation.values,
                        color="gray",
                        alpha=alpha,
                        linestyle="--",
                    )

    # Plot central tendency methods
    for i, method in enumerate(central_tendency_methods):
        if method in compartment:
            central_tendency = compartment[method]
            plt.plot(
                central_tendency.index,
                central_tendency.values,
                color=method_colors[method],
                label=method_names[method],
                linewidth=2,
            )

    # Plot actual testing data if provided
    if testing_data is not None and compartment_code in testing_data.columns:
        plt.plot(
            testing_data.index,
            testing_data[compartment_code],
            color="red",
            label="Actual",
            linewidth=2,
        )

    plt.xlabel("Date")
    plt.ylabel(f"{compartment_labels[compartment_code]} Cases")
    plt.title(f"Forecast for {compartment_labels[compartment_code]}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if log_response:
        plt.yscale("log")

    plt.tight_layout()
    plt.show()
