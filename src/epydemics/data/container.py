"""
DataContainer class and related data processing functionality.

This module contains the DataContainer class extracted from the main
epydemics.py file. The DataContainer handles data preprocessing, validation,
and feature engineering for epidemiological modeling.
"""

import logging
from typing import Optional, Union

import pandas as pd

# Import constants and exceptions from the core module
from ..core.constants import LOGIT_RATIOS, RATIOS
from ..core.exceptions import NotDataFrameError


def validate_data(training_data: pd.DataFrame) -> None:
    """
    Validate that the input data is a pandas DataFrame.

    Args:
        training_data: The data to validate

    Raises:
        NotDataFrameError: If the data is not a pandas DataFrame
    """
    if not isinstance(training_data, pd.DataFrame):
        raise NotDataFrameError("raw data must be a pandas DataFrame")


def prepare_for_logit_function(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare rate data for logit transformation by ensuring values are in (0,1).

    Args:
        data: DataFrame containing rate columns

    Returns:
        DataFrame with rates bounded between 0 and 1
    """
    data = data.copy()

    # Ensure rates are within (0,1) bounds for logit transformation
    for ratio in RATIOS:
        if ratio in data.columns:
            # Replace NaN and infinite values
            data[ratio] = data[ratio].replace(
                [float("inf"), -float("inf")], float("nan")
            )

            # Bound values between small epsilon and (1-epsilon)
            epsilon = 1e-10
            data[ratio] = data[ratio].clip(lower=epsilon, upper=1 - epsilon)

    return data


def logit_function(x: Union[float, pd.Series]) -> Union[float, pd.Series]:
    """
    Compute the logit transformation: log(x/(1-x)).

    Args:
        x: Value(s) to transform, must be in (0,1)

    Returns:
        Logit-transformed value(s)
    """
    import numpy as np

    return np.log(x / (1 - x))


def logistic_function(x: Union[float, pd.Series]) -> Union[float, pd.Series]:
    """
    Compute the logistic (inverse logit) transformation: 1/(1+exp(-x)).

    Args:
        x: Value(s) to transform

    Returns:
        Logistic-transformed value(s)
    """
    import numpy as np

    return 1 / (1 + np.exp(-x))


def add_logit_ratios(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add logit-transformed ratio columns to the DataFrame.

    Args:
        data: DataFrame containing rate columns

    Returns:
        DataFrame with additional logit rate columns
    """
    data = data.copy()

    # Add logit transformations for each ratio
    ratio_pairs = list(zip(RATIOS, LOGIT_RATIOS))
    for ratio, logit_ratio in ratio_pairs:
        if ratio in data.columns:
            try:
                data[logit_ratio] = logit_function(data[ratio])
            except Exception as e:
                logging.warning(f"Could not compute logit for {ratio}: {e}")
                data[logit_ratio] = float("nan")

    return data


def preprocess_data(data: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Preprocess raw data by applying rolling window smoothing and reindexing.

    Args:
        data: Raw input DataFrame
        window: Rolling window size for smoothing

    Returns:
        Preprocessed DataFrame
    """
    # Apply rolling window smoothing
    smoothed_data = data.rolling(window=window).mean()[window:]

    # Reindex to ensure consistent date range
    reindexed_data = reindex_data(smoothed_data)

    return reindexed_data


def reindex_data(
    data: pd.DataFrame, start: Optional[str] = None, stop: Optional[str] = None
) -> pd.DataFrame:
    """
    Reindex DataFrame to a consistent daily date range and forward fill missing values.

    Args:
        data: DataFrame to reindex
        start: Start date (ISO format string), defaults to data minimum
        stop: Stop date (ISO format string), defaults to data maximum

    Returns:
        Reindexed DataFrame with daily frequency

    Raises:
        ValueError: If start > stop or dates are outside data range
    """
    # Handle case where data has no rows
    if len(data) == 0:
        return data

    # Convert dates and set defaults
    start_date = pd.to_datetime(start) if start is not None else data.index.min()
    stop_date = pd.to_datetime(stop) if stop is not None else data.index.max()

    # Validate date range
    if start_date > stop_date:
        raise ValueError("Start date is after stop date")

    if start_date < data.index[0]:
        raise ValueError("Start date is before first date on confirmed cases")

    if stop_date > data.index[-1]:
        raise ValueError("Stop date is after last date of updated cases")

    try:
        logging.debug(
            f"Reindex data from {start_date} to {stop_date} shape: {data.shape}"
        )
        reindex = pd.date_range(start=start_date, end=stop_date, freq="D")
        reindexed_data = data.reindex(reindex)
    except Exception as e:
        raise Exception(f"Could not reindex data: {e}")

    try:
        # Use forward fill for missing values
        reindexed_data = reindexed_data.ffill()
    except Exception as e:
        raise Exception(f"Could not fill missing values: {e}")

    return reindexed_data


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering to create SIRD compartments and rate calculations.

    This function calculates:
    - SIRD compartments (S, I, R, D)
    - Difference values (dC, dI, dR, dD, etc.)
    - Epidemiological rates (alpha, beta, gamma)
    - R0 calculation
    - Logit transformations of rates

    Args:
        data: Preprocessed DataFrame with basic columns C, D, N

    Returns:
        DataFrame with full feature set for epidemiological modeling
    """
    logging.debug(f"When starting feature engineering, columns are {data.columns}")

    # Create a copy to avoid modifying original
    engineered_data = data.copy()

    # Calculate SIRD compartments
    # R: Recovered (using 14-day lag approximation)
    engineered_data = engineered_data.assign(
        R=engineered_data["C"].shift(14).fillna(0) - engineered_data["D"]
    )

    # I: Currently infected (active cases)
    engineered_data = engineered_data.assign(
        I=engineered_data["C"] - engineered_data["R"] - engineered_data["D"]
    )

    # S: Susceptible population
    engineered_data = engineered_data.assign(
        S=engineered_data["N"] - engineered_data["C"]
    )

    # A: At-risk population (S + I)
    engineered_data = engineered_data.assign(
        A=engineered_data["S"] + engineered_data["I"]
    )

    # Calculate differences (daily changes)
    engineered_data = engineered_data.assign(dC=-engineered_data["C"].diff(periods=-1))
    engineered_data = engineered_data.assign(dA=-engineered_data["A"].diff(periods=-1))
    engineered_data = engineered_data.assign(dS=-engineered_data["S"].diff(periods=-1))
    engineered_data = engineered_data.assign(dI=-engineered_data["I"].diff(periods=-1))
    engineered_data = engineered_data.assign(dR=-engineered_data["R"].diff(periods=-1))
    engineered_data = engineered_data.assign(dD=-engineered_data["D"].diff(periods=-1))

    # Calculate epidemiological rates
    # Alpha: infection rate
    engineered_data = engineered_data.assign(
        alpha=(engineered_data.A * engineered_data.dC)
        / (engineered_data.I * engineered_data.S)
    )

    # Beta: recovery rate
    engineered_data = engineered_data.assign(
        beta=engineered_data.dR / engineered_data.I
    )

    # Gamma: mortality rate
    engineered_data = engineered_data.assign(
        gamma=engineered_data.dD / engineered_data.I
    )

    # R0: Basic reproduction number
    engineered_data = engineered_data.assign(
        R0=engineered_data["alpha"]
        / (engineered_data["beta"] + engineered_data["gamma"])
    )

    logging.debug(f"When completing assignments, columns are {engineered_data.columns}")

    # Prepare rates for logit transformation and apply it
    engineered_data = prepare_for_logit_function(engineered_data)
    engineered_data = add_logit_ratios(engineered_data)

    # Final cleanup: forward fill then zero fill any remaining NaN values
    engineered_data = engineered_data.ffill().fillna(0)

    logging.debug(
        f"When completing feature engineering, columns are {engineered_data.columns}"
    )

    return engineered_data


class DataContainer:
    """
    Container for epidemiological data with preprocessing and feature engineering.

    The DataContainer class handles the transformation of raw epidemiological data
    into a format suitable for SIRD (Susceptible-Infected-Recovered-Deaths) modeling.
    It performs data validation, preprocessing with rolling window smoothing,
    and comprehensive feature engineering to create all necessary epidemiological
    variables and rates.

    Attributes:
        raw_data: Original input DataFrame
        window: Rolling window size for smoothing operations
        data: Processed DataFrame with full feature engineering
    """

    def __init__(self, raw_data: pd.DataFrame, window: int = 7) -> None:
        """
        Initialize DataContainer with raw epidemiological data.

        Args:
            raw_data: DataFrame with columns ['C', 'D', 'N'] representing
                     cumulative cases, deaths, and population
            window: Rolling window size for smoothing (default: 7 days)

        Raises:
            NotDataFrameError: If raw_data is not a pandas DataFrame
        """
        self.raw_data = raw_data
        self.window = window

        # Validate input data
        validate_data(self.raw_data)

        # Process data through the pipeline
        self.data = preprocess_data(self.raw_data, window=window)
        logging.debug(f"Preprocessed data columns: {self.data.columns}")

        # Apply feature engineering
        self.data = feature_engineering(self.data)
        logging.debug(f"Feature engineered data columns: {self.data.columns}")
        logging.debug(f"Data shape: {self.data.shape}")
