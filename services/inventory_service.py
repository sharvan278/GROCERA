"""
Inventory Service - Business logic for inventory management
Single Responsibility: Handle all inventory-related operations
"""
from models import db, InventoryItem, PriceHistory
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd


class InventoryService:
    """Service class for inventory operations"""
    
    @staticmethod
    def get_user_inventory(user_id: int, page: int = 1, per_page: int = 20) -> tuple:
        """
        Get paginated inventory for a user
        
        Returns:
            (items, total_count) tuple
        """
        query = InventoryItem.query.filter_by(user_id=user_id)
        total = query.count()
        items = query.paginate(page=page, per_page=per_page, error_out=False)
        return items.items, total
    
    @staticmethod
    def create_item(user_id: int, data: Dict) -> InventoryItem:
        """Create a new inventory item"""
        item = InventoryItem(
            user_id=user_id,
            item_name=data.get('item_name'),
            category=data.get('category'),
            price=float(data.get('price', 0)),
            quantity=int(data.get('quantity', 0)),
            unit=data.get('unit', 'units'),
            store_name=data.get('store_name'),
            stock_status=data.get('stock_status', 'In Stock'),
            expiry_date=data.get('expiry_date'),
            base_price=data.get('base_price'),
            is_discounted=bool(data.get('is_discounted', False))
        )
        
        db.session.add(item)
        db.session.commit()
        
        # Record initial price
        InventoryService.record_price(item.id, item.price, item.store_name)
        
        return item
    
    @staticmethod
    def update_item(item_id: int, user_id: int, data: Dict) -> Optional[InventoryItem]:
        """Update an inventory item"""
        item = InventoryItem.query.filter_by(id=item_id, user_id=user_id).first()
        
        if not item:
            return None
        
        old_price = item.price
        
        # Update fields
        for key, value in data.items():
            if hasattr(item, key) and key not in ['id', 'user_id', 'created_at']:
                setattr(item, key, value)
        
        item.updated_at = datetime.utcnow()
        db.session.commit()
        
        # Record price change
        if old_price != item.price:
            InventoryService.record_price(item.id, item.price, item.store_name)
        
        return item
    
    @staticmethod
    def delete_item(item_id: int, user_id: int) -> bool:
        """Delete an inventory item"""
        item = InventoryItem.query.filter_by(id=item_id, user_id=user_id).first()
        
        if not item:
            return False
        
        db.session.delete(item)
        db.session.commit()
        return True
    
    @staticmethod
    def record_price(item_id: int, price: float, store_name: str = None):
        """Record price history"""
        price_record = PriceHistory(
            item_id=item_id,
            price=price,
            store_name=store_name
        )
        db.session.add(price_record)
        db.session.commit()
    
    @staticmethod
    def bulk_import_from_dataframe(user_id: int, df: pd.DataFrame) -> Dict:
        """
        Import inventory items from DataFrame with intelligent column mapping
        Handles different store formats with flexible column synonyms
        
        Returns:
            {'success': count, 'errors': count, 'details': []}
        """
        results = {'success': 0, 'errors': 0, 'details': []}
        
        # Comprehensive column synonym mapping for various store formats
        column_synonyms = {
            'item_name': ['product_name', 'name', 'item', 'product', 'description', 'item_description'],
            'category': ['cat', 'type', 'group', 'product_category', 'item_category'],
            'price': ['selling_price', 'sale_price', 'cost', 'amount', 'retail_price', 'unit_price', 'mrp'],
            'quantity': ['qty', 'stock', 'quantity_in_stock', 'amount', 'available', 'stock_qty', 'in_stock'],
            'unit': ['uom', 'measure', 'unit_of_measure', 'measurement', 'package'],
            'store_name': ['store', 'shop', 'supplier', 'vendor', 'location', 'branch'],
            'expiry_date': ['expiry', 'exp_date', 'expires', 'expiration_date', 'best_before', 'use_by'],
            'base_price': ['cost_price', 'original_price', 'list_price', 'purchase_price', 'wholesale_price']
        }
        
        # Map columns
        mapped_columns = {}
        df_columns_lower = {col.lower(): col for col in df.columns}
        
        for target_col, synonyms in column_synonyms.items():
            # Check for exact match first
            if target_col in df_columns_lower:
                mapped_columns[target_col] = df_columns_lower[target_col]
            else:
                # Check synonyms
                for synonym in synonyms:
                    if synonym in df_columns_lower:
                        mapped_columns[target_col] = df_columns_lower[synonym]
                        break
        
        for idx, row in df.iterrows():
            try:
                # Parse expiry date with proper format
                expiry_date = None
                expiry_col = mapped_columns.get('expiry_date')
                if expiry_col and pd.notna(row[expiry_col]):
                    try:
                        expiry_date = pd.to_datetime(row[expiry_col], dayfirst=True, format='%d-%m-%Y').date()
                    except:
                        try:
                            expiry_date = pd.to_datetime(row[expiry_col], dayfirst=True).date()
                        except:
                            pass
                
                # Extract item data using mapped columns
                item_name = str(row.get(mapped_columns.get('item_name', 'item_name'), f'Item_{idx}'))
                category = str(row.get(mapped_columns.get('category', 'category'), 'Uncategorized'))
                price = float(row.get(mapped_columns.get('price', 'price'), 0))
                quantity = int(row.get(mapped_columns.get('quantity', 'quantity'), 0))
                unit = str(row.get(mapped_columns.get('unit', 'unit'), 'units'))
                store_name = str(row.get(mapped_columns.get('store_name', 'store_name'), ''))
                base_price = float(row.get(mapped_columns.get('base_price', 'base_price'), price))
                
                item_data = {
                    'item_name': item_name,
                    'category': category,
                    'price': price,
                    'quantity': quantity,
                    'unit': unit,
                    'store_name': store_name,
                    'stock_status': 'Low Stock' if quantity < 5 else 'In Stock',
                    'expiry_date': expiry_date,
                    'base_price': base_price,
                    'is_discounted': price < base_price if base_price > 0 else False
                }
                
                InventoryService.create_item(user_id, item_data)
                results['success'] += 1
                
            except Exception as e:
                results['errors'] += 1
                results['details'].append(f"Row {idx + 1}: {str(e)}")
        
        return results
    
    @staticmethod
    def get_inventory_summary(user_id: int) -> Dict:
        """Get inventory statistics summary"""
        items = InventoryItem.query.filter_by(user_id=user_id).all()
        
        if not items:
            return {
                'total_items': 0,
                'total_value': 0,
                'low_stock_count': 0,
                'categories': {}
            }
        
        total_value = sum(item.total_value for item in items)
        low_stock = sum(1 for item in items if item.quantity < 5)
        
        categories = {}
        for item in items:
            cat = item.category or 'Uncategorized'
            if cat not in categories:
                categories[cat] = {'count': 0, 'value': 0}
            categories[cat]['count'] += 1
            categories[cat]['value'] += item.total_value
        
        return {
            'total_items': len(items),
            'total_value': round(total_value, 2),
            'low_stock_count': low_stock,
            'categories': categories
        }
    
    @staticmethod
    def search_items(user_id: int, query: str) -> List[InventoryItem]:
        """Search inventory items by name or category"""
        return InventoryItem.query.filter_by(user_id=user_id).filter(
            db.or_(
                InventoryItem.item_name.ilike(f'%{query}%'),
                InventoryItem.category.ilike(f'%{query}%')
            )
        ).all()


# Singleton instance
inventory_service = InventoryService()
