"""
Image Generation Module
Provides image generation using Stable Diffusion via Hugging Face Diffusers
"""
import logging
from typing import Optional, Dict, Any
import os
from pathlib import Path
import asyncio
from datetime import datetime

# Hugging Face Diffusers for local image generation
try:
    from diffusers import StableDiffusionPipeline
    import torch
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logger.warning("diffusers not installed. Install with: pip install diffusers torch accelerate")

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Generate images using Stable Diffusion via Hugging Face Diffusers"""

    def __init__(
        self,
        model: str = "runwayml/stable-diffusion-v1-5",
        output_dir: str = "static/generated_images",
        default_steps: int = 20,  # Reduced for CPU
        width: int = 512,
        height: int = 512,
        device: str = "auto"  # auto, cpu, cuda
    ):
        """
        Initialize image generator

        Args:
            model: Hugging Face model name for Stable Diffusion
            output_dir: Directory to save generated images
            default_steps: Default number of generation steps
            width: Default image width
            height: Default image height
            device: Device to use (auto, cpu, cuda)
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers library not available. Install with: pip install diffusers torch accelerate")

        self.model_name = model
        self.output_dir = output_dir
        self.default_steps = default_steps
        self.width = width
        self.height = height

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Loading Stable Diffusion model: {model} on {self.device}")

        try:
            # Load model (this may take time on first run)
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)

            logger.info(f"Image generator initialized with model: {model}")
        except Exception as e:
            logger.error(f"Failed to load Stable Diffusion model: {e}")
            raise
    
    async def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        steps: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate an image from text prompt using Stable Diffusion

        Args:
            prompt: Text description of the image
            negative_prompt: Things to avoid in the image
            steps: Number of generation steps
            width: Image width
            height: Image height
            seed: Random seed for reproducibility

        Returns:
            Dictionary with image path and metadata
        """
        try:
            if steps is None:
                steps = self.default_steps
            if width is None:
                width = self.width
            if height is None:
                height = self.height

            logger.info(f"Generating image: {prompt[:50]}...")

            # Prepare generator for reproducibility if seed provided
            generator = None
            if seed is not None:
                generator = torch.manual_seed(seed)

            # Generate image using Diffusers (run in thread to avoid blocking)
            def _generate():
                with torch.no_grad():
                    return self.pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=steps,
                        width=width,
                        height=height,
                        generator=generator
                    ).images[0]

            image = await asyncio.to_thread(_generate)

            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"image_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)

            # Save the PIL image
            image.save(filepath)

            logger.info(f"Image generated and saved: {filepath}")

            return {
                'success': True,
                'image_path': filepath,
                'filename': filename,
                'url': f"/static/generated_images/{filename}",
                'prompt': prompt,
                'metadata': {
                    'model': self.model_name,
                    'steps': steps,
                    'width': width,
                    'height': height,
                    'seed': seed,
                    'negative_prompt': negative_prompt,
                    'device': self.device
                }
            }

        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return {
                'success': False,
                'error': str(e),
                'prompt': prompt
            }
    
    async def generate_image_variation(
        self,
        base_image_path: str,
        prompt: str,
        steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate image variation based on existing image

        Args:
            base_image_path: Path to base image
            prompt: Modification prompt
            steps: Number of generation steps

        Returns:
            Dictionary with image path and metadata
        """
        try:
            if steps is None:
                steps = self.default_steps

            logger.info(f"Generating image variation: {prompt[:50]}...")

            # For variations, we could use img2img in the future
            # For now, generate new image with combined prompt
            combined_prompt = f"{prompt}, based on existing image"

            # Generate new image
            result = await self.generate_image(
                prompt=combined_prompt,
                steps=steps
            )

            if result['success']:
                result['base_image'] = base_image_path
                result['variation_prompt'] = prompt
                logger.info(f"Image variation generated: {result['filename']}")

            return result

        except Exception as e:
            logger.error(f"Error generating image variation: {e}")
            return {
                'success': False,
                'error': str(e),
                'prompt': prompt
            }


# Global singleton instance
_image_generator = None


def get_image_generator(
    model: str = "runwayml/stable-diffusion-v1-5",
    output_dir: str = "static/generated_images",
    device: str = "auto"
) -> ImageGenerator:
    """
    Get or create singleton image generator

    Args:
        model: Hugging Face model name
        output_dir: Output directory
        device: Device to use (auto, cpu, cuda)

    Returns:
        ImageGenerator instance
    """
    global _image_generator

    if _image_generator is None:
        _image_generator = ImageGenerator(model, output_dir, device=device)

    return _image_generator
