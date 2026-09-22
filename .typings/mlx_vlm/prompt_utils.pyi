from enum import Enum
from typing import Any

class MessageFormat(Enum):
    LIST_WITH_IMAGE_FIRST = "list_with_image_first"

MODEL_CONFIG: dict[str, MessageFormat]

def get_message_json(
    model_name: str,
    prompt: str,
    role: str = "user",
    skip_image_token: bool = False,
    skip_audio_token: bool = False,
    num_images: int = 0,
    num_audios: int = 0,
    **kwargs: Any,
) -> dict[str, Any]: ...
