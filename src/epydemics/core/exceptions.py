"""
Custom exceptions for the Epydemics package.

This module defines a hierarchy of custom exceptions for better error handling
and debugging throughout the package. All custom exceptions inherit from a
base EpydemicsError class for consistent error handling.
"""

from typing import Optional, Any


class EpydemicsError(Exception):
    """
    Base exception class for all Epydemics-specific errors.

    All custom exceptions in the package should inherit from this class
    to provide consistent error handling and to make it easy to catch
    all package-specific errors.

    Args:
        message: The error message
        *args: Additional arguments passed to the base Exception
    """

    def __init__(self, message: Optional[str] = None, *args: Any) -> None:
        (
            super().__init__(message, *args)
            if message is not None
            else super().__init__(*args)
        )


class NotDataFrameError(EpydemicsError):
    """
    Exception raised when input data is not a pandas DataFrame.

    This exception is raised during data validation when the input
    is expected to be a pandas DataFrame but is of a different type.

    Args:
        message: The error message (default: descriptive message about DataFrame requirement)
        *args: Additional arguments passed to the base exception

    Example:
        >>> raise NotDataFrameError("raw data must be a pandas DataFrame")
    """

    def __init__(self, message: Optional[str] = None, *args: Any) -> None:
        if message is None:
            message = "Input data must be a pandas DataFrame"
        super().__init__(message, *args)


class DataValidationError(EpydemicsError):
    """
    Exception raised when data validation fails.

    This exception is used for various data validation errors such as
    missing required columns, invalid data formats, or inconsistent data.

    Args:
        message: The error message describing the validation failure
        *args: Additional arguments passed to the base exception

    Example:
        >>> raise DataValidationError("Missing required columns: ['total_cases', 'total_deaths']")
    """

    def __init__(self, message: Optional[str] = None, *args: Any) -> None:
        if message is None:
            message = "Data validation failed"
        super().__init__(message, *args)


class DateRangeError(EpydemicsError):
    """
    Exception raised when there are issues with date ranges.

    This exception is used when start/stop dates are invalid,
    out of range, or inconsistent with the available data.

    Args:
        message: The error message describing the date range issue
        *args: Additional arguments passed to the base exception

    Example:
        >>> raise DateRangeError("Start date is after stop date")
    """

    def __init__(self, message: Optional[str] = None, *args: Any) -> None:
        if message is None:
            message = "Invalid date range"
        super().__init__(message, *args)


# Export all exceptions for easy importing
__all__ = [
    "EpydemicsError",
    "NotDataFrameError",
    "DataValidationError",
    "DateRangeError",
]
