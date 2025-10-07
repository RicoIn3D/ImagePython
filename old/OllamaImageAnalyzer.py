#!/usr/bin/env python3
"""
Ollama Image Analysis Script
Analyzes brick wall images for cracks and mortar issues using vision models
"""

import requests
import base64
import json
from typing import Dict, Any, Optional
from pathlib import Path


class OllamaImageAnalyzer:
    """Client for analyzing images using Ollama vision models"""
    
    def __init__(self, base_url: str = "http://192.168.87.207:11434"):
        self.base_url = base_url
        self.chat_endpoint = f"{base_url}/api/chat"
        self.generate_endpoint = f"{base_url}/api/generate"
    
    def image_url_to_base64(self, image_url: str) -> str:
        """
        Download image from URL and convert to base64
        
        Args:
            image_url: URL of the image to download
            
        Returns:
            Base64 encoded string of the image
        """
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            return base64.b64encode(response.content).decode('utf-8')
        except requests.RequestException as e:
            raise Exception(f"Failed to download image: {e}")
    
    def image_file_to_base64(self, file_path: str) -> str:
        """
        Read local image file and convert to base64
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Base64 encoded string of the image
        """
        try:
            with open(file_path, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except IOError as e:
            raise Exception(f"Failed to read image file: {e}")
    
    def analyze_brick_wall(
        self,
        image_source: str,
        model: str = "llava",
        is_url: bool = True,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze a brick wall image for structural issues
        
        Args:
            image_source: URL or file path to the image
            model: Ollama model to use (default: llava)
            is_url: True if image_source is URL, False if file path
            stream: Whether to stream the response
            
        Returns:
            Dictionary containing the analysis results
        """
        # Convert image to base64
        if is_url:
            image_base64 = self.image_url_to_base64(image_source)
        else:
            image_base64 = self.image_file_to_base64(image_source)
        
        # Prepare the request payload
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an experienced architect inspecting brick walls for structural issues. "
                        "Analyze images and provide detailed findings about cracks, mortar problems, "
                        "discoloration, and any structural concerns. Be specific about locations and severity."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Inspect this brick wall image for:\n"
                        "1. Cracks (location, size, direction)\n"
                        "2. Mortar issues (missing, deteriorated, gaps)\n"
                        "3. Discoloration or water damage\n"
                        "4. Structural concerns\n\n"
                        "Provide your findings in JSON format with these fields:\n"
                        "- location: where the issue is found\n"
                        "- severity: low, medium, high, critical\n"
                        "- type: crack, mortar_deterioration, water_damage, etc.\n"
                        "- description: detailed description of the issue\n"
                        "- recommendations: suggested actions"
                    ),
                    "images": [image_base64]
                }
            ],
            "format": "json",
            "stream": stream,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9
            }
        }
        
        # Make the request
        try:
            response = requests.post(
                self.chat_endpoint,
                json=payload,
                timeout=120,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_stream_response(response)
            else:
                return response.json()
                
        except requests.RequestException as e:
            raise Exception(f"API request failed: {e}")
    
    def _handle_stream_response(self, response) -> Dict[str, Any]:
        """Handle streaming response from Ollama"""
        full_content = ""
        
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    if "message" in chunk and "content" in chunk["message"]:
                        content = chunk["message"]["content"]
                        full_content += content
                        print(content, end="", flush=True)
                    
                    if chunk.get("done", False):
                        print()  # New line after streaming
                        return {
                            "message": {"content": full_content},
                            "done": True
                        }
                except json.JSONDecodeError:
                    continue
        
        return {"message": {"content": full_content}, "done": True}
    
    def simple_analyze(
        self,
        image_source: str,
        prompt: str,
        model: str = "llava",
        is_url: bool = True
    ) -> str:
        """
        Simple image analysis with custom prompt
        
        Args:
            image_source: URL or file path to the image
            prompt: Custom prompt for analysis
            model: Ollama model to use
            is_url: True if image_source is URL, False if file path
            
        Returns:
            Analysis result as string
        """
        if is_url:
            image_base64 = self.image_url_to_base64(image_source)
        else:
            image_base64 = self.image_file_to_base64(image_source)
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_base64]
                }
            ],
            "stream": False
        }
        
        try:
            response = requests.post(self.chat_endpoint, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "")
        except requests.RequestException as e:
            raise Exception(f"API request failed: {e}")


def main():
    """Example usage"""
    
    # Initialize the analyzer
    analyzer = OllamaImageAnalyzer(base_url="http://localhost:11434")
    
    print("=" * 70)
    print("Ollama Image Analyzer - Brick Wall Inspection")
    print("=" * 70)
    
    # Example 1: Analyze from URL
    print("\n[Example 1] Analyzing brick wall from URL...")
    print("-" * 70)
    
    try:
        image_url = "https://obj3423.public-dk6.clu4.obj.storagefactory.io/dev-poc-drone-images/Chat/Testpulje/uploaded/DJI_0942.JPG"
        
        result = analyzer.analyze_brick_wall(
            image_source=image_url,
            model="llava",
            is_url=True,
            stream=False
        )
        
        # Extract and parse the response
        content = result.get("message", {}).get("content", "")
        print("\nAnalysis Result:")
        print(content)
        
        # Try to parse as JSON if formatted correctly
        try:
            findings = json.loads(content)
            print("\n\nParsed Findings:")
            print(json.dumps(findings, indent=2))
        except json.JSONDecodeError:
            print("\n(Response was not in JSON format)")
            
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Analyze from local file
    print("\n" + "=" * 70)
    print("[Example 2] Analyzing brick wall from local file...")
    print("-" * 70)
    
    try:
        local_image = "brick_wall.jpg"  # Replace with your local file
        
        result = analyzer.analyze_brick_wall(
            image_source=local_image,
            model="llava",
            is_url=False,
            stream=False
        )
        
        content = result.get("message", {}).get("content", "")
        print("\nAnalysis Result:")
        print(content)
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 3: Simple custom analysis with streaming
    print("\n" + "=" * 70)
    print("[Example 3] Simple analysis with custom prompt (streaming)...")
    print("-" * 70)
    
    try:
        analyzer_stream = OllamaImageAnalyzer()
        
        result = analyzer_stream.simple_analyze(
            image_source="https://example.com/wall.jpg",  # Replace with your URL
            prompt="Describe this image in detail. What materials and conditions do you see?",
            model="llava",
            is_url=True
        )
        
        print("\nSimple Analysis:")
        print(result)
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()