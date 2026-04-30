class AppError(Exception):
    error_code = "application_error"
    status_code = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidRequestError(AppError):
    error_code = "invalid_request"
    status_code = 400


class IndexMissingError(AppError):
    error_code = "index_missing"
    status_code = 409


class GenerationError(AppError):
    error_code = "generation_error"
    status_code = 500


class IndexingError(AppError):
    error_code = "indexing_error"
    status_code = 500
