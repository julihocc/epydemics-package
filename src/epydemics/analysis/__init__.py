"""
Analysis module for epydemics package.

This module provides visualization and evaluation functionality
for epidemiological models and forecasts.

Modules:
    visualization: Functions for plotting and visualizing epidemic data and forecasts
    evaluation: Metrics and evaluation functions for model performance assessment
"""

from .evaluation import evaluate_forecast, evaluate_model
from .visualization import visualize_results

__all__ = [
    "evaluate_forecast",
    "evaluate_model",
    "visualize_results",
]
