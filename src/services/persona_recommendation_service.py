"""
Persona-Based Recommendation Service

User segmentation and personalized product recommendations.
"""

from typing import Dict, List, Optional
from src.models.models import db, User, InventoryItem
from sqlalchemy import text
from collections import Counter


class PersonaRecommendationService:
    """Service for persona-based recommendations"""

    def __init__(self):
            0: 'Budget Conscious',
            1: 'Premium Shopper',
            2: 'Balanced Buyer'
        }
    
    def _get_user_purchase_behavior(self, user_id: int) -> Dict:
        """
        Analyze user purchase behavior.
        
        Args:
            user_id: User ID
        
        Returns:
            Behavior metrics
        """
        query = text("""
            SELECT 
                COUNT(*) as total_purchases,
                AVG(price) as avg_price,
                MAX(price) as max_price,
                MIN(price) as min_price,
                COUNT(DISTINCT category) as unique_categories,
                SUM(quantity) as total_quantity
            FROM inventory
            WHERE user_id = :user_id
        """)
        
        result = db.session.execute(query, {'user_id': user_id}).fetchone()
        
        if result and result[0] > 0:
            return {
                'total_purchases': result[0],
                'avg_price': float(result[1]) if result[1] else 0,
                'max_price': float(result[2]) if result[2] else 0,
                'min_price': float(result[3]) if result[3] else 0,
                'unique_categories': result[4],
                'total_quantity': result[5],
                'avg_quantity_per_purchase': result[5] / result[0] if result[0] > 0 else 0
            }
        
        return {
            'total_purchases': 0,
            'avg_price': 0,
            'max_price': 0,
            'min_price': 0,
            'unique_categories': 0,
            'total_quantity': 0,
            'avg_quantity_per_purchase': 0
        }
    
    def _get_category_preferences(self, user_id: int) -> List[tuple]:
        """
        Get user's category preferences.
        
        Args:
            user_id: User ID
        
        Returns:
            List of (category, count) tuples
        """
        query = text("""
            SELECT category, COUNT(*) as count
            FROM inventory
            WHERE user_id = :user_id
            GROUP BY category
            ORDER BY count DESC
            LIMIT 5
        """)
        
        result = db.session.execute(query, {'user_id': user_id}).fetchall()
        return [(row[0], row[1]) for row in result]
    
    def identify_persona(self, user_id: int) -> Dict:
        """
        Identify user persona based on behavior.
        
        Args:
            user_id: User ID
        
        Returns:
            Persona identification
        """
        behavior = self._get_user_purchase_behavior(user_id)
        categories = self._get_category_preferences(user_id)
        
        if behavior['total_purchases'] == 0:
            return {
                'success': True,
                'persona': 'New User',
                'persona_id': -1,
                'confidence': 0,
                'characteristics': ['No purchase history yet']
            }
        
        # Determine persona based on behavior
        avg_price = behavior['avg_price']
        variety = behavior['unique_categories']
        
        if avg_price < 50 and variety < 5:
            persona = 'Budget Conscious'
            persona_id = 0
            characteristics = [
                'Prefers lower-priced items',
                'Focused on specific categories',
                'Value-oriented shopping'
            ]
        elif avg_price > 100 and variety > 5:
            persona = 'Premium Shopper'
            persona_id = 1
            characteristics = [
                'Willing to pay premium prices',
                'Diverse product interests',
                'Quality-focused'
            ]
        else:
            persona = 'Balanced Buyer'
            persona_id = 2
            characteristics = [
                'Moderate pricing preferences',
                'Balanced product selection',
                'Mix of value and quality'
            ]
        
        return {
            'success': True,
            'user_id': user_id,
            'persona': persona,
            'persona_id': persona_id,
            'confidence': min(behavior['total_purchases'] / 20.0, 1.0),
            'characteristics': characteristics,
            'behavior': behavior,
            'top_categories': [cat[0] for cat in categories[:3]]
        }
    
    def recommend_products(self, user_id: int, limit: int = 10) -> Dict:
        """
        Recommend products based on user persona.
        
        Args:
            user_id: User ID
            limit: Number of recommendations
        
        Returns:
            Product recommendations
        """
        persona_data = self.identify_persona(user_id)
        persona_id = persona_data['persona_id']
        
        # Get user's purchase history
        user_items = InventoryItem.query.filter_by(user_id=user_id).all()
        user_categories = [item.category for item in user_items]
        category_counts = Counter(user_categories)
        top_categories = [cat for cat, _ in category_counts.most_common(3)]
        
        # Get all items (simulating marketplace)
        all_items = InventoryItem.query.limit(100).all()
        
        recommendations = []
        for item in all_items:
            if item.user_id == user_id:
                continue  # Skip user's own items
            
            score = 0
            
            # Category match
            if item.category in top_categories:
                score += 5
            
            # Price match based on persona
            if persona_id == 0:  # Budget Conscious
                if item.price < 50:
                    score += 3
            elif persona_id == 1:  # Premium Shopper
                if item.price > 100:
                    score += 3
            else:  # Balanced
                if 50 <= item.price <= 100:
                    score += 3
            
            # Stock availability
            if item.quantity > 0:
                score += 2
            
            # Discount bonus
            if item.is_discounted:
                score += 1
            
            if score > 0:
                recommendations.append({
                    'item_id': item.id,
                    'item_name': item.item_name,
                    'category': item.category,
                    'price': item.price,
                    'is_discounted': item.is_discounted,
                    'score': score,
                    'reason': self._get_recommendation_reason(item, persona_id, top_categories)
                })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'success': True,
            'persona': persona_data['persona'],
            'recommendations': recommendations[:limit],
            'total_available': len(recommendations)
        }
    
    def _get_recommendation_reason(self, item: InventoryItem, persona_id: int, 
                                  top_categories: List[str]) -> str:
        """Generate recommendation reason"""
        reasons = []
        
        if item.category in top_categories:
            reasons.append(f"Matches your interest in {item.category}")
        
        if persona_id == 0 and item.price < 50:
            reasons.append("Great value for budget-conscious shopping")
        elif persona_id == 1 and item.price > 100:
            reasons.append("Premium quality product")
        
        if item.is_discounted:
            reasons.append("Currently on discount")
        
        return ' • '.join(reasons) if reasons else "Recommended for you"
    
    def get_cross_sell_recommendations(self, item_id: int, limit: int = 5) -> Dict:
        """
        Get cross-sell recommendations for an item.
        
        Args:
            item_id: Item ID
            limit: Number of recommendations
        
        Returns:
            Cross-sell recommendations
        """
        item = InventoryItem.query.get(item_id)
        if not item:
            return {'success': False, 'error': 'Item not found'}
        
        # Find items in same category
        similar_items = InventoryItem.query.filter(
            InventoryItem.category == item.category,
            InventoryItem.id != item_id,
            InventoryItem.quantity > 0
        ).limit(limit).all()
        
        recommendations = []
        for similar in similar_items:
            recommendations.append({
                'item_id': similar.id,
                'item_name': similar.item_name,
                'category': similar.category,
                'price': similar.price,
                'reason': f"Frequently bought with {item.item_name}"
            })
        
        return {
            'success': True,
            'base_item': {
                'id': item.id,
                'name': item.item_name,
                'category': item.category
            },
            'recommendations': recommendations
        }
    
    def get_trending_for_persona(self, persona_id: int, limit: int = 10) -> Dict:
        """
        Get trending items for specific persona.
        
        Args:
            persona_id: Persona ID (0, 1, 2)
            limit: Number of items
        
        Returns:
            Trending items
        """
        # Price ranges for personas
        price_ranges = {
            0: (0, 50),      # Budget Conscious
            1: (100, 999),   # Premium Shopper
            2: (50, 100)     # Balanced Buyer
        }
        
        min_price, max_price = price_ranges.get(persona_id, (0, 999))
        
        items = InventoryItem.query.filter(
            InventoryItem.price >= min_price,
            InventoryItem.price <= max_price,
            InventoryItem.quantity > 0
        ).order_by(InventoryItem.created_at.desc()).limit(limit).all()
        
        return {
            'success': True,
            'persona': self.personas.get(persona_id, 'Unknown'),
            'items': [{
                'item_id': item.id,
                'item_name': item.item_name,
                'category': item.category,
                'price': item.price,
                'is_discounted': item.is_discounted
            } for item in items]
        }


# Singleton instance
persona_recommendation_service = PersonaRecommendationService()
