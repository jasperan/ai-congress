"""Image generation routes."""
import logging

from fastapi import APIRouter, HTTPException

from ..schemas import ImageGenRequest
from ..state import config, event_logger, image_generator
from ...integrations.image_gen import get_image_generator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["images"])


@router.post("/images/generate")
async def generate_image(request: ImageGenRequest):
    """Generate image from text prompt"""
    global image_generator

    try:
        if image_generator is None:
            image_generator = get_image_generator(
                model=config.image_gen.model,
                output_dir=config.image_gen.output_dir,
                device=config.image_gen.device
            )

        result = await image_generator.generate_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            steps=request.steps,
            width=request.width,
            height=request.height,
            seed=request.seed
        )

        event_logger.log("image_generate", prompt_length=len(request.prompt))
        return result

    except Exception as e:
        logger.error(f"Image generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
