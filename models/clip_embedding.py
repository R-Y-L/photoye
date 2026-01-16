#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLIP Embedding encoder for semantic search using ONNX Runtime.

This module provides image and text embedding extraction using OpenCLIP ViT-B/32.
The embeddings can be used for semantic similarity search.

V2.3 升级:
- Prompt Ensemble: 多模板平均提升文本 embedding 质量
- Multi-Crop: 多裁剪融合捕获更完整的图像信息
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

MODEL_DIR = Path(__file__).resolve().parent / "models"

# Prompt Ensemble 模板 (参考 OpenAI CLIP 论文)
PROMPT_TEMPLATES = [
    "a photo of {}",
    "a photograph of {}",
    "an image of {}",
    "{} in a photo",
    "a picture of {}",
    "a good photo of {}",
    "a photo showing {}",
]

# Multi-Crop 配置
CROP_POSITIONS = ["center", "top_left", "top_right", "bottom_left", "bottom_right"]
CROP_WEIGHTS = [2.0, 1.0, 1.0, 1.0, 1.0]  # 中心权重更高


class CLIPEmbeddingEncoder:
    """CLIP encoder that extracts 512-dim embeddings for images and text."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        tokenizer_path: Optional[Path] = None,
    ) -> None:
        """Initialize the CLIP encoder.
        
        Args:
            model_path: Path to the fused ONNX model (vision + text).
            tokenizer_path: Path to the tokenizer JSON file.
        """
        self.model_path = Path(model_path) if model_path else MODEL_DIR / "openclip_vitb32_vision.onnx"
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else MODEL_DIR / "openclip_tokenizer.json"
        
        self.context_length = 77
        self.embedding_dim = 512
        self.session: Optional[ort.InferenceSession] = None
        self.tokenizer: Optional[Tokenizer] = None
        
        # Image preprocessing constants (OpenCLIP)
        self.image_size = 224
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        self.std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        
        self._init_model()

    def _init_model(self) -> None:
        """Load ONNX model and tokenizer."""
        if ort is None:
            print("⚠️ onnxruntime not installed, CLIP encoder disabled")
            return
            
        if not self.model_path.exists():
            print(f"⚠️ CLIP model not found: {self.model_path}")
            return
            
        try:
            self.session = ort.InferenceSession(
                str(self.model_path), 
                providers=["CPUExecutionProvider"]
            )
        except Exception as e:
            print(f"⚠️ Failed to load CLIP model: {e}")
            return
            
        if Tokenizer is None:
            print("⚠️ tokenizers not installed, text encoding disabled")
        elif not self.tokenizer_path.exists():
            print(f"⚠️ Tokenizer not found: {self.tokenizer_path}")
        else:
            try:
                self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
            except Exception as e:
                print(f"⚠️ Failed to load tokenizer: {e}")

    def is_available(self) -> bool:
        """Check if the encoder is ready."""
        return self.session is not None

    # ------------------------------------------------------------------
    # Image Embedding
    # ------------------------------------------------------------------
    
    def encode_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Optional[np.ndarray]:
        """Encode a single image to 512-dim embedding.
        
        Args:
            image: Image path, PIL Image, or numpy array (H, W, C).
            
        Returns:
            Normalized 512-dim numpy array, or None if failed.
        """
        if not self.session:
            return None
            
        try:
            pixel_values = self._preprocess_image(image)
            if pixel_values is None:
                return None
                
            # Need dummy text input for fused model
            dummy_ids, dummy_mask = self._get_dummy_text_input()
            
            outputs = self.session.run(None, {
                "pixel_values": pixel_values,
                "input_ids": dummy_ids,
                "attention_mask": dummy_mask,
            })
            
            # outputs[3] = image_embeds: [1, 512]
            image_embed = outputs[3][0]  # [512]
            
            # L2 normalize
            norm = np.linalg.norm(image_embed)
            if norm > 0:
                image_embed = image_embed / norm
                
            return image_embed.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ Image encoding failed: {e}")
            return None

    def encode_images_batch(
        self, 
        images: List[Union[str, Path, Image.Image]], 
        batch_size: int = 16
    ) -> List[Optional[np.ndarray]]:
        """Encode multiple images in batches.
        
        Args:
            images: List of image paths or PIL Images.
            batch_size: Number of images to process at once.
            
        Returns:
            List of 512-dim embeddings (None for failed images).
        """
        results = []
        for img in images:
            embedding = self.encode_image(img)
            results.append(embedding)
        return results

    def _preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Optional[np.ndarray]:
        """Preprocess image for CLIP model.
        
        Returns:
            Preprocessed image tensor [1, 3, 224, 224].
        """
        try:
            if isinstance(image, (str, Path)):
                img = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                img = Image.fromarray(image).convert("RGB")
            else:
                img = image.convert("RGB")
                
            # Resize with center crop
            img = self._resize_and_center_crop(img, self.image_size)
            
            # Convert to numpy and normalize
            arr = np.array(img).astype(np.float32) / 255.0
            arr = (arr - self.mean) / self.std
            arr = arr.transpose(2, 0, 1)  # [3, 224, 224]
            arr = arr[None, ...]  # [1, 3, 224, 224]
            
            return arr.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ Image preprocessing failed: {e}")
            return None

    def _resize_and_center_crop(self, img: Image.Image, size: int) -> Image.Image:
        """Resize and center crop to square."""
        w, h = img.size
        scale = size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        
        left = (new_w - size) // 2
        top = (new_h - size) // 2
        return img.crop((left, top, left + size, top + size))

    def _get_dummy_text_input(self) -> tuple[np.ndarray, np.ndarray]:
        """Create dummy text input for fused model."""
        # Single empty prompt
        ids = np.zeros((1, self.context_length), dtype=np.int64)
        mask = np.zeros((1, self.context_length), dtype=np.int64)
        ids[0, 0] = 49406  # <|startoftext|>
        ids[0, 1] = 49407  # <|endoftext|>
        mask[0, :2] = 1
        return ids, mask

    # ------------------------------------------------------------------
    # Text Embedding
    # ------------------------------------------------------------------
    
    def encode_text(self, text: str) -> Optional[np.ndarray]:
        """Encode text to 512-dim embedding.
        
        Args:
            text: Query text string.
            
        Returns:
            Normalized 512-dim numpy array, or None if failed.
        """
        if not self.session or not self.tokenizer:
            return None
            
        try:
            input_ids, attention_mask = self._tokenize(text)
            
            # Need dummy image input for fused model
            dummy_image = np.zeros((1, 3, 224, 224), dtype=np.float32)
            
            outputs = self.session.run(None, {
                "pixel_values": dummy_image,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            })
            
            # outputs[2] = text_embeds: [1, 512]
            text_embed = outputs[2][0]  # [512]
            
            # L2 normalize
            norm = np.linalg.norm(text_embed)
            if norm > 0:
                text_embed = text_embed / norm
                
            return text_embed.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ Text encoding failed: {e}")
            return None

    def encode_texts_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Encode multiple texts."""
        return [self.encode_text(t) for t in texts]

    def _tokenize(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize text for CLIP model.
        
        Returns:
            (input_ids, attention_mask) each of shape [1, 77].
        """
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids[:self.context_length]
        
        # Build attention mask
        mask = [1] * len(ids)
        
        # Pad to context_length
        if len(ids) < self.context_length:
            pad_len = self.context_length - len(ids)
            ids = ids + [0] * pad_len
            mask = mask + [0] * pad_len
            
        return (
            np.array([ids], dtype=np.int64),
            np.array([mask], dtype=np.int64)
        )

    # ------------------------------------------------------------------
    # Similarity Search
    # ------------------------------------------------------------------
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(a, b))

    def search(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Find top-k most similar embeddings.
        
        Args:
            query_embedding: Query vector [512].
            embeddings: Database vectors [N, 512].
            top_k: Number of results to return.
            
        Returns:
            List of (index, similarity_score) tuples, sorted by score descending.
        """
        if len(embeddings) == 0:
            return []
            
        # Compute cosine similarities (embeddings are already normalized)
        similarities = embeddings @ query_embedding
        
        # Get top-k indices
        if top_k >= len(similarities):
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
            
        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    # ------------------------------------------------------------------
    # V2.3: Prompt Ensemble
    # ------------------------------------------------------------------
    
    def encode_text_ensemble(self, concept: str) -> Optional[np.ndarray]:
        """Encode text using prompt ensemble for better quality.
        
        Uses multiple prompt templates and averages the embeddings.
        
        Args:
            concept: The concept to encode (e.g., "sunset beach", "family dinner")
            
        Returns:
            Normalized 512-dim embedding averaged from all templates.
        """
        if not self.session or not self.tokenizer:
            return None
            
        embeddings = []
        for template in PROMPT_TEMPLATES:
            prompt = template.format(concept)
            emb = self.encode_text(prompt)
            if emb is not None:
                embeddings.append(emb)
                
        if not embeddings:
            return None
            
        # Average and re-normalize
        mean_emb = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm
            
        return mean_emb.astype(np.float32)

    # ------------------------------------------------------------------
    # V2.3: Multi-Crop Image Encoding
    # ------------------------------------------------------------------
    
    def encode_image_multicrop(
        self, 
        image: Union[str, Path, Image.Image],
        n_crops: int = 5
    ) -> Optional[np.ndarray]:
        """Encode image using multi-crop ensemble for better coverage.
        
        Extracts embeddings from multiple crop positions and combines them
        with weighted average, giving center crop higher weight.
        
        Args:
            image: Image path or PIL Image.
            n_crops: Number of crops (1-5). Default 5 uses all positions.
            
        Returns:
            Normalized 512-dim embedding from weighted crop average.
        """
        if not self.session:
            return None
            
        try:
            # Load image
            if isinstance(image, (str, Path)):
                img = Image.open(image).convert("RGB")
            else:
                img = image.convert("RGB")
                
            # Generate crops
            crops = self._generate_crops(img, n_crops)
            
            # Encode each crop
            embeddings = []
            weights = []
            for i, crop in enumerate(crops):
                emb = self._encode_pil_image(crop)
                if emb is not None:
                    embeddings.append(emb)
                    weights.append(CROP_WEIGHTS[i] if i < len(CROP_WEIGHTS) else 1.0)
                    
            if not embeddings:
                return None
                
            # Weighted average
            weights = np.array(weights)
            weights = weights / weights.sum()  # normalize weights
            weighted_emb = sum(w * e for w, e in zip(weights, embeddings))
            
            # Re-normalize
            norm = np.linalg.norm(weighted_emb)
            if norm > 0:
                weighted_emb = weighted_emb / norm
                
            return weighted_emb.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ Multi-crop encoding failed: {e}")
            return None

    def _generate_crops(self, img: Image.Image, n_crops: int = 5) -> List[Image.Image]:
        """Generate multiple crops from image.
        
        Args:
            img: PIL Image
            n_crops: Number of crops (1-5)
            
        Returns:
            List of cropped PIL Images, each 224x224
        """
        w, h = img.size
        min_dim = min(w, h)
        
        crops = []
        positions = CROP_POSITIONS[:n_crops]
        
        for pos in positions:
            if pos == "center":
                # Center crop at full scale
                crop = self._crop_at_position(img, w, h, min_dim, "center")
            else:
                # Corner crops at 80% scale for more context overlap
                crop_size = int(min_dim * 0.8)
                crop = self._crop_at_position(img, w, h, crop_size, pos)
                
            # Resize to 224x224
            crop = crop.resize((self.image_size, self.image_size), Image.BILINEAR)
            crops.append(crop)
            
        return crops

    def _crop_at_position(
        self, 
        img: Image.Image, 
        w: int, 
        h: int, 
        crop_size: int, 
        position: str
    ) -> Image.Image:
        """Crop image at specified position.
        
        Args:
            img: PIL Image
            w, h: Image dimensions
            crop_size: Size of square crop
            position: One of center, top_left, top_right, bottom_left, bottom_right
            
        Returns:
            Cropped PIL Image
        """
        if position == "center":
            left = (w - crop_size) // 2
            top = (h - crop_size) // 2
        elif position == "top_left":
            left = 0
            top = 0
        elif position == "top_right":
            left = w - crop_size
            top = 0
        elif position == "bottom_left":
            left = 0
            top = h - crop_size
        elif position == "bottom_right":
            left = w - crop_size
            top = h - crop_size
        else:
            # Default to center
            left = (w - crop_size) // 2
            top = (h - crop_size) // 2
            
        # Clamp values
        left = max(0, left)
        top = max(0, top)
        right = min(w, left + crop_size)
        bottom = min(h, top + crop_size)
        
        return img.crop((left, top, right, bottom))

    def _encode_pil_image(self, img: Image.Image) -> Optional[np.ndarray]:
        """Encode a PIL Image directly (already preprocessed size).
        
        Args:
            img: PIL Image, should be 224x224 RGB
            
        Returns:
            Normalized 512-dim embedding
        """
        try:
            # Convert to numpy and normalize
            arr = np.array(img).astype(np.float32) / 255.0
            arr = (arr - self.mean) / self.std
            arr = arr.transpose(2, 0, 1)  # [3, 224, 224]
            arr = arr[None, ...]  # [1, 3, 224, 224]
            
            # Need dummy text input for fused model
            dummy_ids, dummy_mask = self._get_dummy_text_input()
            
            outputs = self.session.run(None, {
                "pixel_values": arr.astype(np.float32),
                "input_ids": dummy_ids,
                "attention_mask": dummy_mask,
            })
            
            # outputs[3] = image_embeds: [1, 512]
            image_embed = outputs[3][0]  # [512]
            
            # L2 normalize
            norm = np.linalg.norm(image_embed)
            if norm > 0:
                image_embed = image_embed / norm
                
            return image_embed.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ PIL image encoding failed: {e}")
            return None


# Convenience function
def create_clip_encoder() -> CLIPEmbeddingEncoder:
    """Create a CLIP encoder with default settings."""
    return CLIPEmbeddingEncoder()
