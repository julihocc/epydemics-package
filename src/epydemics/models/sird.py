"""SIRD epidemiological model with VAR time series forecasting."""

import itertools
import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from box import Box
from scipy.stats import gmean, hmean
from statsmodels.tsa.api import VAR

from ..analysis.evaluation import evaluate_forecast as _evaluate_forecast
from ..analysis.visualization import visualize_results as _visualize_results
from ..core.constants import compartments, forecasting_levels, logit_ratios
from ..data.container import reindex_data
from ..utils.transformations import logistic_function
from .base import BaseModel, SIRDModelMixin


class Model(BaseModel, SIRDModelMixin):
    """
    SIRD epidemiological model with VAR time series forecasting.

    This model implements the SIRD (Susceptible-Infected-Recovered-Deaths)
    compartmental model with time-varying rates modeled using Vector Autoregression
    on logit-transformed infection, recovery, and mortality rates.
    """

    def __init__(
        self,
        data_container,
        start: Optional[str] = None,
        stop: Optional[str] = None,
        days_to_forecast: Optional[int] = None,
    ):
        """
        Initialize the SIRD Model.

        Args:
            data_container: DataContainer instance with preprocessed data
            start: Start date for model training (YYYY-MM-DD format)
            stop: Stop date for model training (YYYY-MM-DD format)
            days_to_forecast: Number of days to forecast ahead
        """
        # Data and model attributes
        self.data: Optional[pd.DataFrame] = None
        self.data_container = data_container
        self.window = data_container.window
        self.start = start
        self.stop = stop

        # Results and simulation attributes (set during model execution)
        self.results: Optional[Box] = None
        self.simulation: Optional[Box] = None
        self.forecasting_box: Optional[Dict[str, pd.DataFrame]] = None
        self.forecasted_logit_ratios_tuple_arrays: Optional[Any] = None
        self.forecasting_interval: Optional[pd.DatetimeIndex] = None
        self.forecast_index_stop: Optional[pd.Timestamp] = None
        self.forecast_index_start: Optional[pd.Timestamp] = None

        # Model parameters
        self.days_to_forecast = days_to_forecast

        # VAR model attributes
        self.logit_ratios_model: Optional[VAR] = None
        self.logit_ratios_model_fitted: Optional[Any] = None
        self.forecasted_logit_ratios: Optional[pd.DataFrame] = None

        self.data = reindex_data(data_container.data, start, stop)
        self.logit_ratios_values = self.data[logit_ratios].values

    def create_model(self, *args, **kwargs) -> None:
        """Create the VAR model for logit-transformed rates."""
        self.create_logit_ratios_model(*args, **kwargs)

    def fit_model(self, *args, **kwargs) -> None:
        """Fit the VAR model to the data."""
        self.fit_logit_ratios_model(*args, **kwargs)

    def forecast(self, steps: Optional[int] = None, **kwargs) -> None:
        """Generate forecasts for the specified number of steps."""
        self.forecast_logit_ratios(steps, **kwargs)

    def simulate_epidemic(self) -> None:
        """Run epidemic simulations based on forecasted rates."""
        self.run_simulations()

    def evaluate_model(self, testing_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Evaluate model performance against test data."""
        return self.evaluate_forecast(testing_data, **kwargs)

    def create_logit_ratios_model(self, *args, **kwargs) -> None:
        """
        Create VAR model for logit-transformed rates.

        Args:
            *args: Positional arguments for VAR constructor
            **kwargs: Keyword arguments for VAR constructor
        """
        self.logit_ratios_model = VAR(self.logit_ratios_values, *args, **kwargs)

    def fit_logit_ratios_model(self, *args, **kwargs) -> None:
        """
        Fit the VAR model to logit-transformed rates.

        Args:
            *args: Positional arguments for VAR.fit()
            **kwargs: Keyword arguments for VAR.fit()
        """
        self.logit_ratios_model_fitted = self.logit_ratios_model.fit(*args, **kwargs)
        if self.days_to_forecast is None:
            self.days_to_forecast = self.logit_ratios_model_fitted.k_ar + self.window

    def forecast_logit_ratios(self, steps: Optional[int] = None, **kwargs) -> None:
        """
        Generate forecasts for logit-transformed rates.

        Args:
            steps: Number of steps to forecast (overrides days_to_forecast)
            **kwargs: Keyword arguments for forecast_interval()
        """
        if steps:
            self.days_to_forecast = steps
        last_date = self.data.index[-1]
        self.forecast_index_start = last_date + pd.Timedelta(days=1)
        self.forecast_index_stop = last_date + pd.Timedelta(days=self.days_to_forecast)
        self.forecasting_interval = pd.date_range(
            start=self.forecast_index_start,
            end=self.forecast_index_stop,
            freq="D",
        )
        try:
            self.forecasted_logit_ratios_tuple_arrays = (
                self.logit_ratios_model_fitted.forecast_interval(
                    self.logit_ratios_values, self.days_to_forecast, **kwargs
                )
            )
        except Exception as e:
            raise Exception(e)

        self.forecasting_box = {
            logit_ratios[0]: pd.DataFrame(
                self.forecasted_logit_ratios_tuple_arrays[0],
                index=self.forecasting_interval,
                columns=forecasting_levels,
            ),
            logit_ratios[1]: pd.DataFrame(
                self.forecasted_logit_ratios_tuple_arrays[1],
                index=self.forecasting_interval,
                columns=forecasting_levels,
            ),
            logit_ratios[2]: pd.DataFrame(
                self.forecasted_logit_ratios_tuple_arrays[2],
                index=self.forecasting_interval,
                columns=forecasting_levels,
            ),
        }

        self.forecasting_box["alpha"] = self.forecasting_box["logit_alpha"].apply(
            logistic_function
        )
        self.forecasting_box["beta"] = self.forecasting_box["logit_beta"].apply(
            logistic_function
        )
        self.forecasting_box["gamma"] = self.forecasting_box["logit_gamma"].apply(
            logistic_function
        )

        self.forecasting_box = Box(self.forecasting_box)

    def simulate_for_given_levels(
        self, simulation_levels: Tuple[str, str, str]
    ) -> pd.DataFrame:
        """
        Simulate epidemic dynamics for given rate confidence levels.

        Args:
            simulation_levels: Tuple of (alpha_level, beta_level, gamma_level)

        Returns:
            DataFrame with simulated compartment values
        """
        simulation = (
            self.data[["A", "C", "S", "I", "R", "D", "alpha", "beta", "gamma"]]
            .iloc[-1:]
            .copy()
        )

        for t1 in self.forecasting_interval:
            t0 = t1 - pd.Timedelta(days=1)
            previous = simulation.loc[t0]
            S = previous.S - previous.I * previous.alpha * previous.S / previous.A
            I = (
                previous.I
                + previous.I * previous.alpha * previous.S / previous.A
                - previous.beta * previous.I
                - previous.gamma * previous.I
            )
            R = previous.R + previous.beta * previous.I
            D = previous.D + previous.gamma * previous.I
            C = I + R + D
            A = previous.A

            simulation.loc[t1] = [
                A,
                C,
                S,
                I,
                R,
                D,
                self.forecasting_box["alpha"][simulation_levels[0]].loc[t1],
                self.forecasting_box["beta"][simulation_levels[1]].loc[t1],
                self.forecasting_box["gamma"][simulation_levels[2]].loc[t1],
            ]

        simulation = simulation.iloc[1:]
        try:
            simulation.index = self.forecasting_interval
        except Exception as e:
            raise Exception(e)

        return simulation

    def create_simulation_box(self) -> None:
        """Create nested Box structure for storing simulation results."""
        self.simulation = Box()
        for logit_alpha_level in forecasting_levels:
            self.simulation[logit_alpha_level] = Box()
            for logit_beta_level in forecasting_levels:
                self.simulation[logit_alpha_level][logit_beta_level] = Box()
                for logit_gamma_level in forecasting_levels:
                    self.simulation[logit_alpha_level][logit_beta_level][
                        logit_gamma_level
                    ] = None

    def run_simulations(self) -> None:
        """Run epidemic simulations for all combinations of rate confidence levels."""
        self.create_simulation_box()
        for current_levels in itertools.product(
            forecasting_levels, forecasting_levels, forecasting_levels
        ):
            logit_alpha_level, logit_beta_level, logit_gamma_level = current_levels
            current_simulation = self.simulate_for_given_levels(current_levels)
            self.simulation[logit_alpha_level][logit_beta_level][
                logit_gamma_level
            ] = current_simulation

    def create_results_dataframe(self, compartment: str) -> pd.DataFrame:
        """
        Create results DataFrame for a specific compartment.

        Args:
            compartment: Compartment code (A, C, S, I, R, D)

        Returns:
            DataFrame with simulation results and central tendencies
        """
        results_dataframe = pd.DataFrame()
        logging.debug(results_dataframe.head())

        levels_interactions = itertools.product(
            forecasting_levels, forecasting_levels, forecasting_levels
        )

        for (
            logit_alpha_level,
            logit_beta_level,
            logit_gamma_level,
        ) in levels_interactions:
            column_name = f"{logit_alpha_level}|{logit_beta_level}|{logit_gamma_level}"
            simulation = self.simulation[logit_alpha_level][logit_beta_level][
                logit_gamma_level
            ]
            results_dataframe[column_name] = simulation[compartment].values

        results_dataframe["mean"] = results_dataframe.mean(axis=1)
        results_dataframe["median"] = results_dataframe.median(axis=1)
        results_dataframe["gmean"] = results_dataframe.apply(gmean, axis=1)
        results_dataframe["hmean"] = results_dataframe.apply(hmean, axis=1)

        results_dataframe.index = self.forecasting_interval

        return results_dataframe

    def generate_result(self) -> None:
        """Generate results for all compartments."""
        self.results = Box()

        for compartment in compartments:
            self.results[compartment] = self.create_results_dataframe(compartment)

    def visualize_results(
        self,
        compartment_code: str,
        testing_data: Optional[pd.DataFrame] = None,
        log_response: bool = True,
    ) -> None:
        """
        Visualize forecast results for a specific compartment.

        Args:
            compartment_code: Compartment to visualize (A, C, S, I, R, D)
            testing_data: Optional test data for comparison
            log_response: Whether to use logarithmic scale
        """
        _visualize_results(
            results=self.results,
            compartment_code=compartment_code,
            testing_data=testing_data,
            log_response=log_response,
        )

    def evaluate_forecast(
        self,
        testing_data: pd.DataFrame,
        compartment_codes: Tuple[str, ...] = ("C", "D", "I"),
        save_evaluation: bool = False,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate forecast performance against test data.

        Args:
            testing_data: DataFrame with actual values for comparison
            compartment_codes: Tuple of compartment codes to evaluate
            save_evaluation: Whether to save results to JSON file
            filename: Optional filename for saving (auto-generated if None)

        Returns:
            Dictionary with evaluation metrics for each compartment and method
        """
        return _evaluate_forecast(
            results=self.results,
            testing_data=testing_data,
            compartment_codes=compartment_codes,
            save_evaluation=save_evaluation,
            filename=filename,
        )
