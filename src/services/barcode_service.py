"""
Barcode Service

Handles barcode/QR code generation, scanning, and product lookup.
"""

import barcode
from barcode.writer import ImageWriter
import qrcode
from PIL import Image
import io
import os
from typing import Dict, Optional, Tuple
import requests
from src.models.models import db, InventoryItem


class BarcodeService:
    """Service for barcode generation, scanning, and lookup"""
    
    def __init__(self):
        self.barcode_dir = 'static/barcodes'
        self.qrcode_dir = 'static/qrcodes'
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create barcode directories if they don't exist"""
        os.makedirs(self.barcode_dir, exist_ok=True)
        os.makedirs(self.qrcode_dir, exist_ok=True)
    
    def generate_barcode(self, item_id: int, barcode_value: Optional[str] = None) -> str:
        """
        Generate barcode for an inventory item.
        
        Args:
            item_id: Inventory item ID
            barcode_value: Optional custom barcode value (uses item_id if not provided)
        
        Returns:
            Path to generated barcode image
        """
        item = InventoryItem.query.get(item_id)
        if not item:
            raise ValueError(f"Item {item_id} not found")
        
        # Use custom value or generate from item_id
        if not barcode_value:
            # Generate 12-digit barcode (EAN13 format requires 12 digits + 1 checksum)
            barcode_value = f"{item_id:012d}"
        
        # Create barcode (EAN13 format)
        try:
            ean = barcode.get('ean13', barcode_value, writer=ImageWriter())
            filename = f"barcode_{item_id}"
            filepath = os.path.join(self.barcode_dir, filename)
            
            # Save barcode image
            ean.save(filepath, options={
                'module_width': 0.3,
                'module_height': 10.0,
                'font_size': 10,
                'text_distance': 5,
                'quiet_zone': 6.5
            })
            
            # Update item with barcode value
            item.barcode = barcode_value
            db.session.commit()
            
            return f"{filepath}.png"
            
        except Exception as e:
            print(f"Barcode generation error: {str(e)}")
            # Fallback to CODE128 which is more flexible
            try:
                code128 = barcode.get('code128', str(item_id), writer=ImageWriter())
                filename = f"barcode_{item_id}"
                filepath = os.path.join(self.barcode_dir, filename)
                code128.save(filepath)
                
                item.barcode = str(item_id)
                db.session.commit()
                
                return f"{filepath}.png"
            except Exception as e2:
                raise ValueError(f"Failed to generate barcode: {str(e2)}")
    
    def generate_qr_code(self, item_id: int, include_details: bool = True) -> str:
        """
        Generate QR code for an inventory item.
        
        Args:
            item_id: Inventory item ID
            include_details: Include item details in QR code
        
        Returns:
            Path to generated QR code image
        """
        item = InventoryItem.query.get(item_id)
        if not item:
            raise ValueError(f"Item {item_id} not found")
        
        # Create QR code data
        if include_details:
            qr_data = f"""
Item: {item.item_name}
Category: {item.category}
Price: ${item.price:.2f}
Stock: {item.quantity}
Expiry: {item.expiry_date.strftime('%Y-%m-%d') if item.expiry_date else 'N/A'}
            """.strip()
        else:
            qr_data = f"ITEM:{item_id}"
        
        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(qr_data)
        qr.make(fit=True)
        
        # Create image
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Save image
        filename = f"qr_{item_id}.png"
        filepath = os.path.join(self.qrcode_dir, filename)
        img.save(filepath)
        
        return filepath
    
    def lookup_barcode(self, barcode_value: str) -> Optional[InventoryItem]:
        """
        Look up item by barcode in inventory.
        
        Args:
            barcode_value: Barcode value to search
        
        Returns:
            InventoryItem if found, None otherwise
        """
        return InventoryItem.query.filter_by(barcode=barcode_value).first()
    
    def lookup_barcode_external(self, barcode_value: str) -> Optional[Dict]:
        """
        Look up product information from external barcode databases.
        Uses Open Food Facts API (free, open source).
        
        Args:
            barcode_value: Barcode/UPC/EAN code
        
        Returns:
            Product information dictionary or None
        """
        try:
            # Try Open Food Facts API
            url = f"https://world.openfoodfacts.org/api/v0/product/{barcode_value}.json"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 1:  # Product found
                    product = data.get('product', {})
                    
                    return {
                        'name': product.get('product_name', 'Unknown'),
                        'brand': product.get('brands', ''),
                        'category': product.get('categories', ''),
                        'quantity': product.get('quantity', ''),
                        'image_url': product.get('image_url', ''),
                        'source': 'Open Food Facts'
                    }
            
            # Try UPC Item DB API (alternative)
            url = f"https://api.upcitemdb.com/prod/trial/lookup?upc={barcode_value}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('items', [])
                
                if items:
                    item = items[0]
                    return {
                        'name': item.get('title', 'Unknown'),
                        'brand': item.get('brand', ''),
                        'category': item.get('category', ''),
                        'image_url': item.get('images', [''])[0],
                        'source': 'UPC Item DB'
                    }
            
            return None
            
        except Exception as e:
            print(f"External barcode lookup error: {str(e)}")
            return None
    
    def create_item_from_barcode(self, user_id: int, barcode_value: str, 
                                 external_lookup: bool = True) -> Optional[InventoryItem]:
        """
        Create inventory item from barcode scan.
        
        Args:
            user_id: User ID
            barcode_value: Scanned barcode value
            external_lookup: Try external APIs for product info
        
        Returns:
            Created InventoryItem or None
        """
        # Check if item already exists
        existing = self.lookup_barcode(barcode_value)
        if existing:
            return existing
        
        # Try external lookup
        product_info = None
        if external_lookup:
            product_info = self.lookup_barcode_external(barcode_value)
        
        # Create new item
        if product_info:
            item = InventoryItem(
                user_id=user_id,
                item_name=product_info['name'],
                category=product_info.get('category', 'Uncategorized')[:50],
                quantity=1,  # Default quantity
                price=0.0,  # User must set price
                barcode=barcode_value
            )
        else:
            # Create basic item
            item = InventoryItem(
                user_id=user_id,
                item_name=f"Item {barcode_value}",
                category="Uncategorized",
                quantity=1,
                price=0.0,
                barcode=barcode_value
            )
        
        db.session.add(item)
        db.session.commit()
        
        return item
    
    def scan_barcode_from_image(self, image_path: str) -> Optional[str]:
        """
        Scan barcode from uploaded image.
        Note: Requires pyzbar library (install: pip install pyzbar)
        
        Args:
            image_path: Path to image file
        
        Returns:
            Decoded barcode value or None
        """
        try:
            from pyzbar import pyzbar
            
            # Open image
            image = Image.open(image_path)
            
            # Decode barcodes
            barcodes = pyzbar.decode(image)
            
            if barcodes:
                # Return first barcode found
                return barcodes[0].data.decode('utf-8')
            
            return None
            
        except ImportError:
            print("pyzbar library not installed. Install with: pip install pyzbar")
            return None
        except Exception as e:
            print(f"Barcode scanning error: {str(e)}")
            return None
    
    def bulk_generate_barcodes(self, user_id: int) -> Dict:
        """
        Generate barcodes for all items without barcodes.
        
        Args:
            user_id: User ID
        
        Returns:
            Results summary
        """
        items = InventoryItem.query.filter_by(user_id=user_id).filter(
            (InventoryItem.barcode == None) | (InventoryItem.barcode == '')
        ).all()
        
        results = {
            'total': len(items),
            'success': 0,
            'failed': 0,
            'items': []
        }
        
        for item in items:
            try:
                filepath = self.generate_barcode(item.id)
                results['success'] += 1
                results['items'].append({
                    'item_id': item.id,
                    'item_name': item.item_name,
                    'barcode_path': filepath
                })
            except Exception as e:
                results['failed'] += 1
                print(f"Failed to generate barcode for {item.item_name}: {str(e)}")
        
        return results
    
    def get_printable_labels(self, item_ids: list) -> list:
        """
        Generate printable labels with barcode and item info.
        
        Args:
            item_ids: List of inventory item IDs
        
        Returns:
            List of label data dictionaries
        """
        labels = []
        
        for item_id in item_ids:
            item = InventoryItem.query.get(item_id)
            if not item:
                continue
            
            # Generate barcode if not exists
            if not item.barcode:
                try:
                    barcode_path = self.generate_barcode(item_id)
                except:
                    barcode_path = None
            else:
                barcode_path = f"{self.barcode_dir}/barcode_{item_id}.png"
            
            labels.append({
                'item_id': item.id,
                'item_name': item.item_name,
                'category': item.category,
                'price': item.price,
                'barcode': item.barcode,
                'barcode_image': barcode_path,
                'expiry_date': item.expiry_date.strftime('%Y-%m-%d') if item.expiry_date else None
            })
        
        return labels


# Singleton instance
barcode_service = BarcodeService()
