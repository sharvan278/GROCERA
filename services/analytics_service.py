"""
Analytics Service - Price comparisons, alerts, recommendations
Single Responsibility: Handle all analytics and insights
"""
from models import db, InventoryItem, Alert
from datetime import datetime, timedelta
from typing import List, Dict
from config import Config


class AnalyticsService:
    """Service class for analytics operations"""
    
    @staticmethod
    def get_stock_alerts(user_id: int) -> List[Dict]:
        """Generate stock alerts for low inventory and expiring items"""
        alerts = []
        items = InventoryItem.query.filter_by(user_id=user_id).all()
        today = datetime.utcnow().date()
        
        for item in items:
            # Low stock alert
            if item.quantity <= Config.STOCK_ALERT_THRESHOLD:
                alerts.append({
                    'type': 'low_stock',
                    'severity': 'high' if item.quantity < 2 else 'medium',
                    'item': item.item_name,
                    'current_stock': item.quantity,
                    'message': f'Only {item.quantity} {item.unit} remaining',
                    'item_id': item.id
                })
            
            # Expiry alert
            if item.expiry_date:
                days_left = (item.expiry_date - today).days
                if days_left <= Config.EXPIRY_ALERT_DAYS:
                    alerts.append({
                        'type': 'expiry',
                        'severity': 'high' if days_left <= 3 else 'medium',
                        'item': item.item_name,
                        'days_left': days_left,
                        'expiry_date': item.expiry_date.isoformat(),
                        'message': f'Expires in {days_left} days',
                        'item_id': item.id
                    })
        
        return sorted(alerts, key=lambda x: x['severity'], reverse=True)
    
    @staticmethod
    def create_alert(user_id: int, alert_type: str, item_name: str, 
                    message: str, severity: str = 'medium'):
        """Create and save an alert"""
        alert = Alert(
            user_id=user_id,
            alert_type=alert_type,
            item_name=item_name,
            message=message,
            severity=severity
        )
        db.session.add(alert)
        db.session.commit()
        return alert
    
    @staticmethod
    def get_trending_items(user_id: int, limit: int = 10) -> List[Dict]:
        """Get trending items based on value and velocity"""
        items = InventoryItem.query.filter_by(user_id=user_id).all()
        
        if not items:
            return []
        
        # Calculate velocity (quantity / price ratio)
        trending = []
        for item in items:
            velocity = item.quantity / (item.price + 0.001)
            trending.append({
                'item_name': item.item_name,
                'price': item.price,
                'quantity': item.quantity,
                'total_value': item.total_value,
                'velocity': velocity,
                'category': item.category
            })
        
        # Sort by velocity and total value
        trending.sort(key=lambda x: (x['velocity'], x['total_value']), reverse=True)
        return trending[:limit]
    
    @staticmethod
    def get_restock_recommendations(user_id: int) -> List[Dict]:
        """Generate restocking recommendations"""
        items = InventoryItem.query.filter_by(user_id=user_id).all()
        recommendations = []
        
        for item in items:
            if item.quantity < 5:
                suggested_order = max(20 - item.quantity, 5)
                recommendations.append({
                    'type': 'restock',
                    'item': item.item_name,
                    'current_stock': item.quantity,
                    'suggested_order': suggested_order,
                    'unit': item.unit,
                    'priority': 'high' if item.quantity < 2 else 'medium',
                    'estimated_cost': item.price * suggested_order
                })
        
        return sorted(recommendations, key=lambda x: x['priority'], reverse=True)
    
    @staticmethod
    def get_expiry_recommendations(user_id: int) -> List[Dict]:
        """Generate recommendations for expiring items"""
        items = InventoryItem.query.filter_by(user_id=user_id).all()
        recommendations = []
        today = datetime.utcnow().date()
        
        for item in items:
            if item.expiry_date:
                days_left = (item.expiry_date - today).days
                if days_left <= 7:
                    action = "discount 30%" if days_left <= 3 else "promote"
                    discount_pct = min(30, (8 - days_left) * 5)
                    
                    recommendations.append({
                        'type': 'expiry',
                        'item': item.item_name,
                        'days_left': days_left,
                        'quantity': item.quantity,
                        'current_price': item.price,
                        'suggested_action': action,
                        'discount_pct': discount_pct,
                        'new_price': round(item.price * (1 - discount_pct/100), 2)
                    })
        
        return sorted(recommendations, key=lambda x: x['days_left'])
    
    @staticmethod
    def calculate_price_recommendations(user_id: int) -> List[Dict]:
        """Calculate pricing recommendations with market analysis"""
        items = InventoryItem.query.filter_by(user_id=user_id).all()
        recommendations = []
        
        # Elasticity data for common products
        ELASTICITY = {
            'potatoes': -1.8, 'salt': -1.2, 'bananas': -1.5,
            'milk': -1.4, 'bread': -1.7, 'eggs': -1.3,
            'default': -1.5
        }
        
        for item in items:
            # Get elasticity
            item_lower = item.item_name.lower()
            elasticity = ELASTICITY.get(item_lower, ELASTICITY['default'])
            
            # Simulate market price (in real app, call external API)
            market_avg = item.price * 1.05  # Assume 5% higher market price
            
            # Calculate recommended price
            if elasticity < -1.5:  # Highly elastic
                recommended = market_avg * 0.95  # 5% below market
            elif elasticity < -1:  # Moderately elastic
                recommended = market_avg * 0.98  # 2% below market
            else:  # Inelastic
                recommended = market_avg * 1.05  # 5% above market
            
            # Adjust for inventory
            if item.quantity > 50:
                recommended = min(recommended, item.price * 0.85)  # Discount overstock
            elif item.quantity < 5:
                recommended = max(recommended, item.price * 1.15)  # Premium for low stock
            
            price_change_pct = ((recommended - item.price) / item.price * 100) if item.price > 0 else 0
            
            if abs(price_change_pct) > 1:  # Only recommend if change > 1%
                recommendations.append({
                    'item': item.item_name,
                    'current_price': item.price,
                    'recommended_price': round(recommended, 2),
                    'market_avg': round(market_avg, 2),
                    'price_change_pct': round(price_change_pct, 1),
                    'reason': AnalyticsService._get_price_reason(price_change_pct, item.quantity)
                })
        
        return sorted(recommendations, key=lambda x: abs(x['price_change_pct']), reverse=True)
    
    @staticmethod
    def _get_price_reason(price_change_pct: float, quantity: int) -> str:
        """Generate reason for price recommendation"""
        reasons = []
        
        if price_change_pct < 0:
            reasons.append("Reduce price to increase sales")
        else:
            reasons.append("Increase price for better margins")
        
        if quantity > 50:
            reasons.append("High inventory - discount to clear")
        elif quantity < 5:
            reasons.append("Low stock - premium pricing justified")
        
        return ". ".join(reasons)
    
    @staticmethod
    def get_inventory_status(user_id: int) -> List[Dict]:
        """Get detailed inventory status"""
        items = InventoryItem.query.filter_by(user_id=user_id).all()
        
        status_list = []
        for item in items:
            if item.quantity < 3:
                status = 'critical'
            elif item.quantity < 10:
                status = 'low'
            else:
                status = 'adequate'
            
            status_list.append({
                'id': item.id,
                'name': item.item_name,
                'quantity': item.quantity,
                'unit': item.unit,
                'price': item.price,
                'total_value': item.total_value,
                'stock_level': status,
                'category': item.category,
                'progress': min(100, item.quantity * 5)  # Visual progress bar
            })
        
        return status_list


# Singleton instance
analytics_service = AnalyticsService()
