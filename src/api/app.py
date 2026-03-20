"""FastAPI application factory for CV Midterm project.

Mounts all Part A/B/C API routers and configures CORS
middleware for future web UI integration.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routes.filtering_routes import (
    router as filtering_router,
)
from src.api.routes.filtering_compare_routes import (
    router as filtering_compare_router,
)
from src.api.routes.stereo_routes import (
    router as stereo_router,
)
from src.api.routes.epipolar_routes import (
    router as epipolar_router,
)
from src.api.routes.stitching_routes import (
    router as stitching_router,
)
from src.api.routes.stitching_stitch_routes import (
    router as stitching_stitch_router,
)
from src.api.routes.model_routes import (
    router as model_router,
)
from src.api.routes.reconstruct_routes import (
    router as reconstruct_router,
)
from src.api.routes.monocular_routes import (
    router as monocular_router,
)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured application instance.
    """
    application = FastAPI(
        title="CV Midterm API",
        description="REST API for Image Filtering, 3D Reconstruction, "
                    "and Image Stitching (INS3155)",
        version="1.0.0",
    )

    _add_cors(application)
    _register_routers(application)
    _mount_static(application)

    return application


def _add_cors(application: FastAPI) -> None:
    """Add CORS middleware for web UI access.

    Args:
        application: FastAPI instance.
    """
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _register_routers(application: FastAPI) -> None:
    """Register all API routers.

    Args:
        application: FastAPI instance.
    """
    application.include_router(filtering_router)
    application.include_router(filtering_compare_router)
    application.include_router(stereo_router)
    application.include_router(epipolar_router)
    application.include_router(stitching_router)
    application.include_router(stitching_stitch_router)
    application.include_router(model_router)
    application.include_router(reconstruct_router)
    application.include_router(monocular_router)


def _mount_static(application: FastAPI) -> None:
    """Mount static file directories for output access.

    Args:
        application: FastAPI instance.
    """
    import os
    os.makedirs("outputs", exist_ok=True)
    application.mount(
        "/outputs",
        StaticFiles(directory="outputs"),
        name="outputs",
    )


app = create_app()
