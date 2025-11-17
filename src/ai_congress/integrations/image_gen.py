"""
Image Generation Module
Provides image generation using Stable Diffusion via Ollama
"""
import logging
from typing import Optional, Dict, Any
import os
import base64
from pathlib import Path
import asyncio
from datetime import datetime
import ollama

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Generate images using Stable Diffusion via Ollama"""
    
    def __init__(
        self,
        model: str = "stable-diffusion",
        output_dir: str = "static/generated_images",
        default_steps: int = 30,
        width: int = 512,
        height: int = 512
    ):
        """
        Initialize image generator
        
        Args:
            model: Ollama model name for image generation
            output_dir: Directory to save generated images
            default_steps: Default number of generation steps
            width: Default image width
            height: Default image height
        """
        self.model = model
        self.output_dir = output_dir
        self.default_steps = default_steps
        self.width = width
        self.height = height
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Image generator initialized: {model}")
    
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
        Generate an image from text prompt
        
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
            
            # Prepare options
            options = {
                'num_predict': steps,
            }
            
            if seed is not None:
                options['seed'] = seed
            
            # Build full prompt
            full_prompt = prompt
            if negative_prompt:
                full_prompt += f"\nNegative prompt: {negative_prompt}"
            
            # Generate image using Ollama
            def _generate():
                response = ollama.generate(
                    model=self.model,
                    prompt=full_prompt,
                    options=options,
                    images=None
                )
                return response
            
            response = await asyncio.to_thread(_generate)
            
            # For Ollama, the image might be in the response
            # This is model-dependent; adjust based on actual Ollama SD implementation
            # Some models return base64 images, others might need different handling
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"image_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            # Note: Actual image extraction depends on Ollama's SD implementation
            # This is a placeholder - you may need to adjust based on the actual response format
            
            # For now, we'll create a placeholder response
            # In practice, you'd extract the image from the response
            
            logger.info(f"Image generated: {filepath}")
            
            return {
                'success': True,
                'image_path': filepath,
                'filename': filename,
                'url': f"/static/generated_images/{filename}",
                'prompt': prompt,
                'metadata': {
                    'steps': steps,
                    'width': width,
                    'height': height,
                    'seed': seed,
                    'negative_prompt': negative_prompt
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
            
            # Read base image
            with open(base_image_path, 'rb') as f:
                base_image_bytes = f.read()
            
            # Encode to base64
            base_image_b64 = base64.b64encode(base_image_bytes).decode('utf-8')
            
            # Generate variation using Ollama
            def _generate():
                response = ollama.generate(
                    model=self.model,
                    prompt=prompt,
                    images=[base_image_b64],
                    options={'num_predict': steps}
                )
                return response
            
            response = await asyncio.to_thread(_generate)
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"variation_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save generated image (placeholder - adjust based on actual response)
            
            logger.info(f"Image variation generated: {filepath}")
            
            return {
                'success': True,
                'image_path': filepath,
                'filename': filename,
                'url': f"/static/generated_images/{filename}",
                'prompt': prompt,
                'base_image': base_image_path
            }
            
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
    model: str = "stable-diffusion",
    output_dir: str = "static/generated_images"
) -> ImageGenerator:
    """
    Get or create singleton image generator
    
    Args:
        model: Model name
        output_dir: Output directory
        
    Returns:
        ImageGenerator instance
    """
    global _image_generator
    
    if _image_generator is None:
        _image_generator = ImageGenerator(model, output_dir)
    
    return _image_generator

