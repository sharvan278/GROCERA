"""
Computer Vision Expiry Detection Service

Uses OCR and image processing to detect expiry dates from product images.
"""

from PIL import Image
import re
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import os


class ExpiryDetectionService:
    """Computer vision service for expiry date detection"""
    
    def __init__(self):
        self.date_patterns = [
            r'(\d{2})/(\d{2})/(\d{4})',  # DD/MM/YYYY
            r'(\d{2})-(\d{2})-(\d{4})',  # DD-MM-YYYY
            r'(\d{4})/(\d{2})/(\d{2})',  # YYYY/MM/DD
            r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD
            r'EXP:?\s*(\d{2})/(\d{2})/(\d{4})',  # EXP: DD/MM/YYYY
            r'BEST\s*BEFORE:?\s*(\d{2})/(\d{2})/(\d{4})',  # BEST BEFORE DD/MM/YYYY
            r'USE\s*BY:?\s*(\d{2})/(\d{2})/(\d{4})',  # USE BY DD/MM/YYYY
        ]
    
    def _extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from image using OCR.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Extracted text
        """
        try:
            import pytesseract
            from PIL import Image
            
            img = Image.open(image_path)
            
            # Preprocess image for better OCR
            img = img.convert('L')  # Convert to grayscale
            
            # Extract text
            text = pytesseract.image_to_string(img)
            return text
            
        except ImportError:
            print("pytesseract not installed. Install with: uv pip install pytesseract")
            return ""
        except Exception as e:
            print(f"OCR error: {str(e)}")
            return ""
    
    def _parse_date(self, text: str) -> Optional[datetime]:
        """
        Parse date from text using regex patterns.
        
        Args:
            text: Text containing date
        
        Returns:
            Parsed datetime or None
        """
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            
            for match in matches:
                try:
                    if len(match) == 3:
                        # Try different date formats
                        formats = [
                            '%d/%m/%Y', '%m/%d/%Y',  # DD/MM/YYYY or MM/DD/YYYY
                            '%Y/%m/%d', '%Y-%m-%d',  # YYYY/MM/DD
                            '%d-%m-%Y', '%m-%d-%Y'   # DD-MM-YYYY or MM-DD-YYYY
                        ]
                        
                        date_str = '/'.join(match) if '/' in pattern else '-'.join(match)
                        
                        for fmt in formats:
                            try:
                                date = datetime.strptime(date_str, fmt)
                                # Validate date is in reasonable range
                                if 2020 <= date.year <= 2030:
                                    return date
                            except:
                                continue
                except Exception as e:
                    continue
        
        return None
    
    def detect_expiry_from_image(self, image_path: str) -> Dict:
        """
        Detect expiry date from product image.
        
        Args:
            image_path: Path to product image
        
        Returns:
            Detection results
        """
        if not os.path.exists(image_path):
            return {'success': False, 'error': 'Image not found'}
        
        # Extract text from image
        text = self._extract_text_from_image(image_path)
        
        if not text:
            return {'success': False, 'error': 'Could not extract text from image'}
        
        # Parse expiry date
        expiry_date = self._parse_date(text)
        
        if expiry_date:
            days_until_expiry = (expiry_date.date() - datetime.utcnow().date()).days
            
            return {
                'success': True,
                'expiry_date': expiry_date.strftime('%Y-%m-%d'),
                'days_until_expiry': days_until_expiry,
                'is_expired': days_until_expiry < 0,
                'urgency': 'high' if days_until_expiry < 7 else 'medium' if days_until_expiry < 30 else 'low',
                'extracted_text': text[:200]  # First 200 chars
            }
        else:
            return {
                'success': False,
                'error': 'No expiry date found in image',
                'extracted_text': text[:200]
            }
    
    def batch_detect_expiry(self, image_dir: str) -> List[Dict]:
        """
        Detect expiry dates from multiple images.
        
        Args:
            image_dir: Directory containing product images
        
        Returns:
            List of detection results
        """
        if not os.path.exists(image_dir):
            return []
        
        results = []
        
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_dir, filename)
                result = self.detect_expiry_from_image(image_path)
                result['filename'] = filename
                results.append(result)
        
        return results
    
    def analyze_product_freshness(self, image_path: str) -> Dict:
        """
        Analyze product freshness from image using color analysis.
        
        Args:
            image_path: Path to product image
        
        Returns:
            Freshness analysis
        """
        try:
            from PIL import Image
            import numpy as np
            
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # Calculate average RGB values
            avg_r = np.mean(img_array[:, :, 0])
            avg_g = np.mean(img_array[:, :, 1])
            avg_b = np.mean(img_array[:, :, 2])
            
            # Simple freshness heuristic based on green content
            # (More green = fresher for produce)
            freshness_score = (avg_g / (avg_r + avg_g + avg_b)) * 100
            
            # Calculate color variation (lower = more uniform = potentially older)
            color_std = np.std(img_array)
            
            return {
                'success': True,
                'freshness_score': round(freshness_score, 2),
                'color_variation': round(color_std, 2),
                'freshness_level': 'high' if freshness_score > 40 else 'medium' if freshness_score > 30 else 'low',
                'avg_rgb': {
                    'r': round(avg_r, 2),
                    'g': round(avg_g, 2),
                    'b': round(avg_b, 2)
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


# Singleton instance
expiry_detection_service = ExpiryDetectionService()
