"""
Base classes for epidemiological models.

This module provides abstract base classes and interfaces for epidemiological
modeling implementations in the epydemics package.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd


class BaseModel(ABC):
    """
    Abstract base class for epidemiological models.

    This class defines the interface that all epidemiological models
    should implement, including data handling, model fitting, forecasting,
    and evaluation capabilities.
    """

    def __init__(
        self,
        data_container,
        start: Optional[str] = None,
        stop: Optional[str] = None,
        days_to_forecast: Optional[int] = None,
    ):
        """
        Initialize the base model.

        Args:
            data_container: DataContainer with processed epidemiological data
            start: Start date for modeling period (ISO format)
            stop: Stop date for modeling period (ISO format)
            days_to_forecast: Number of days to forecast ahead
        """
        self.data_container = data_container
        self.start = start
        self.stop = stop
        self.days_to_forecast = days_to_forecast

        # Initialize common attributes that will be set during model lifecycle
        self.data: Optional[pd.DataFrame] = None
        self.fitted_model: Optional[Any] = None
        self.forecasting_results: Optional[Dict[str, Any]] = None
        self.simulation_results: Optional[Dict[str, Any]] = None

    @abstractmethod
    def create_model(self, *args, **kwargs) -> None:
        """Create the underlying statistical/mathematical model."""
        pass

    @abstractmethod
    def fit_model(self, *args, **kwargs) -> None:
        """Fit the model to the data."""
        pass

    @abstractmethod
    def forecast(self, steps: Optional[int] = None, **kwargs) -> None:
        """Generate forecasts from the fitted model."""
        pass

    @abstractmethod
    def run_simulations(self) -> None:
        """Run epidemic simulations based on model forecasts."""
        pass

    @abstractmethod
    def evaluate_forecast(
        self,
        testing_data: pd.DataFrame,
        compartment_codes: tuple[str, ...] = ("C", "D", "I"),
        save_evaluation: bool = False,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate forecast accuracy against testing data."""
        pass

    @abstractmethod
    def visualize_results(
        self,
        compartment_code: str,
        testing_data: Optional[pd.DataFrame] = None,
        log_response: bool = True,
    ) -> None:
        """Visualize model results and forecasts."""
        pass


class SIRDModelMixin:
    """
    Mixin class providing SIRD-specific functionality.

    This mixin provides methods and utilities specific to SIRD
    (Susceptible-Infected-Recovered-Deaths) epidemiological models.
    """

    def get_sird_compartments(self) -> list[str]:
        """Get the list of SIRD compartment names."""
        return ["S", "I", "R", "D", "C", "A"]

    def get_sird_rates(self) -> list[str]:
        """Get the list of SIRD rate names."""
        return ["alpha", "beta", "gamma"]

    def get_logit_rates(self) -> list[str]:
        """Get the list of logit-transformed rate names."""
        return ["logit_alpha", "logit_beta", "logit_gamma"]

    def validate_sird_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that data contains required SIRD columns.

        Args:
            data: DataFrame to validate

        Returns:
            True if data contains all required SIRD columns
        """
        required_cols = self.get_sird_compartments() + self.get_sird_rates()
        missing_cols = [col for col in required_cols if col not in data.columns]

        if missing_cols:
            raise ValueError(f"Missing required SIRD columns: {missing_cols}")

        return True
