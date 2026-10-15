"""Run the FinGraph API locally."""

import uvicorn

from fingraph.config import (
    get_settings,
)


def main() -> None:
    """Start the development API server."""

    settings = get_settings()

    uvicorn.run(
        "fingraph.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_config=None,
    )


if __name__ == "__main__":
    main()
