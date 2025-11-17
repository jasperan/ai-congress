#!/usr/bin/env python3
"""
Uvicorn Server Launcher with Verbose Logging
Runs the AI Congress API with detailed request/response logging
"""
import logging
import sys
import uvicorn

# Configure root logger for maximum verbosity
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)

# Set all relevant loggers to DEBUG
logging.getLogger("uvicorn").setLevel(logging.DEBUG)
logging.getLogger("uvicorn.error").setLevel(logging.DEBUG)
logging.getLogger("uvicorn.access").setLevel(logging.DEBUG)
logging.getLogger("fastapi").setLevel(logging.DEBUG)
logging.getLogger("ai_congress").setLevel(logging.DEBUG)

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 Starting AI Congress API with VERBOSE logging")
    print("=" * 80)
    print()
    
    uvicorn.run(
        "src.ai_congress.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug",  # Set uvicorn log level to debug
        access_log=True,    # Enable access logs
        use_colors=True,    # Colorized output
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "access": {
                    "format": '%(asctime)s | ACCESS    | %(message)s',
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
                "access": {
                    "formatter": "access",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                "uvicorn": {
                    "handlers": ["default"],
                    "level": "DEBUG",
                    "propagate": False
                },
                "uvicorn.error": {
                    "handlers": ["default"],
                    "level": "DEBUG",
                    "propagate": False
                },
                "uvicorn.access": {
                    "handlers": ["access"],
                    "level": "DEBUG",
                    "propagate": False
                },
                "fastapi": {
                    "handlers": ["default"],
                    "level": "DEBUG",
                    "propagate": False
                },
                "ai_congress": {
                    "handlers": ["default"],
                    "level": "DEBUG",
                    "propagate": False
                },
            },
        }
    )

