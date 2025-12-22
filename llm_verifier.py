import os
import cv2
import base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv() # Load env vars from .env file

class LlmVerifier:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            print("Warning: OPENAI_API_KEY not found. LLM features will be disabled.")
            self.client = None
            return

        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o-mini" 
        print(f"LlmVerifier initialized with OpenAI {self.model}.")

    def _encode_image(self, frame):
        """Encodes a CV2 frame to base64 string."""
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    def verify_event(self, frames):
        """
        Sends a sequence of frames to the LLM to ask if a theft occurred.
        frames: List of cv2 images (numpy arrays)
        """
        if not self.client:
            return "LLM Disabled (No API Key)"

        print("Sending frames to OpenAI for verification...")
        
        # Prepare images for OpenAI API
        # We'll take a subsample if there are too many frames to save tokens/cost
        # e.g., take up to 5 evenly spaced frames
        num_frames = len(frames)
        if num_frames > 5:
            step = num_frames // 5
            selected_frames = frames[::step][:5]
        else:
            selected_frames = frames
            
        content_images = []
        for frame in selected_frames:
            base64_image = self._encode_image(frame)
            content_images.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

        prompt_text = (
            "You are a security assistant. Analyze this sequence of images from a doorbell camera. "
            "Did a person take a package? Answer with 'YES' or 'NO' first, then provide a brief description of what happened."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            *content_images
                        ],
                    }
                ],
                max_tokens=300,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return "Error during verification"
