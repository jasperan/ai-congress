"""Personality routes."""
import logging
import json
import os
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from ..schemas import Personality, PersonalityCreate
from ..state import load_personalities, security_ctx, event_logger

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["personalities"])


@router.get("/personalities", response_model=List[Personality])
async def list_personalities():
    """List all available personalities (predefined and custom)"""
    personalities = await load_personalities()
    return [Personality(**p) for p in personalities]


@router.post("/personalities", response_model=Personality, dependencies=[Depends(security_ctx.require_api_key)])
async def create_personality(personality: PersonalityCreate):
    """Create a new custom personality (4.9.4: optional API key)."""
    event_logger.log("audit.personality_create", name=personality.name)
    custom_file = "personalities/custom_personalities.json"

    # Ensure directory exists
    os.makedirs("personalities", exist_ok=True)

    # Load existing custom personalities
    custom_personalities = []
    if os.path.exists(custom_file):
        try:
            with open(custom_file, 'r') as f:
                custom_personalities = json.load(f)
        except Exception as e:
            logger.error(f"Error loading custom personalities: {e}")

    # Check for duplicate name
    if any(p['name'] == personality.name for p in custom_personalities):
        raise HTTPException(status_code=400, detail="Personality name already exists")

    # Add new personality
    new_personality = {"name": personality.name, "system_prompt": personality.system_prompt}
    custom_personalities.append(new_personality)

    # Save back to file
    try:
        with open(custom_file, 'w') as f:
            json.dump(custom_personalities, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving custom personalities: {e}")
        raise HTTPException(status_code=500, detail="Failed to save personality")

    return Personality(**new_personality)


@router.get("/personality-lists", response_model=List[str])
async def list_personality_lists():
    """List available personality lists"""
    return ["hollywood", "us_congress", "youtubers"]


@router.get("/personality-list/{list_name}", response_model=List[Personality])
async def get_personality_list(list_name: str):
    """Get personalities from a specific list"""
    # Handle filename mapping: hollywood has _personalities suffix, others don't
    if list_name == "hollywood":
        file_path = f"config/{list_name}_personalities.json"
    else:
        file_path = f"config/{list_name}.json"

    if not os.path.exists(file_path):
        return []

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading personality list {list_name}: {e}")
        return []

    # Normalize to Personality format (name, system_prompt)
    personalities = []
    for item in data:
        if "name" in item and "system_prompt" in item:
            personalities.append(Personality(
                name=item["name"],
                system_prompt=item["system_prompt"]
            ))

    return personalities
