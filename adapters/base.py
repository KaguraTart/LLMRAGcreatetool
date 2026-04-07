"""Common adapter errors and helpers."""


class AdapterError(RuntimeError):
    """Base adapter exception."""


class AdapterConfigError(AdapterError):
    """Raised when adapter configuration is invalid."""


class AdapterAuthError(AdapterError):
    """Raised when adapter authentication fails."""


class AdapterRequestError(AdapterError):
    """Raised when adapter request handling fails."""
