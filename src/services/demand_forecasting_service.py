"""
Demand Forecasting Service

Uses time series analysis and machine learning to predict future demand.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from src.models.models import db, InventoryItem, PriceHistory
from sqlalchemy import text


class DemandForecastingService:
    """Service for predicting product demand using ML"""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False
    
    def _get_historical_data(self, user_id: int, days: int = 90) -> pd.DataFrame:
        """
        Get historical sales and inventory data.
        
        Args:
            user_id: User ID
            days: Number of days of history
        
        Returns:
            DataFrame with historical data
        """
        query = text("""
            SELECT 
                i.id as item_id,
                i.item_name,
                i.category,
                i.price,
                i.quantity,
                i.created_at,
                i.updated_at,
                ph.price as historical_price,
                ph.changed_at
            FROM inventory i
            LEFT JOIN price_history ph ON i.id = ph.item_id
            WHERE i.user_id = :user_id
            AND i.created_at >= DATE_SUB(NOW(), INTERVAL :days DAY)
            ORDER BY i.id, ph.changed_at
        """)
        
        result = db.session.execute(query, {'user_id': user_id, 'days': days})
        data = result.fetchall()
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data, columns=result.keys())
        return df
    
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for demand forecasting.
        
        Args:
            df: Raw historical data
        
        Returns:
            DataFrame with engineered features
        """
        if df.empty:
            return df
        
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Time-based features
        df['day_of_week'] = df['created_at'].dt.dayofweek
        df['day_of_month'] = df['created_at'].dt.day
        df['month'] = df['created_at'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Price features
        df['price_change'] = df.groupby('item_id')['price'].pct_change().fillna(0)
        df['price_volatility'] = df.groupby('item_id')['price'].rolling(7, min_periods=1).std().reset_index(0, drop=True).fillna(0)
        
        # Inventory features
        df['quantity_change'] = df.groupby('item_id')['quantity'].diff().fillna(0)
        df['stock_velocity'] = df['quantity_change'].rolling(7, min_periods=1).mean().fillna(0)
        
        # Lag features
        df['quantity_lag_1'] = df.groupby('item_id')['quantity'].shift(1).fillna(0)
        df['quantity_lag_7'] = df.groupby('item_id')['quantity'].shift(7).fillna(0)
        
        # Rolling statistics
        df['quantity_mean_7d'] = df.groupby('item_id')['quantity'].rolling(7, min_periods=1).mean().reset_index(0, drop=True).fillna(0)
        df['quantity_std_7d'] = df.groupby('item_id')['quantity'].rolling(7, min_periods=1).std().reset_index(0, drop=True).fillna(0)
        
        return df
    
    def train_model(self, user_id: int) -> Dict:
        """
        Train demand forecasting model.
        
        Args:
            user_id: User ID
        
        Returns:
            Training results
        """
        df = self._get_historical_data(user_id)
        
        if df.empty or len(df) < 10:
            return {'success': False, 'error': 'Insufficient historical data'}
        
        df = self._extract_features(df)
        
        # Feature columns
        feature_cols = [
            'price', 'day_of_week', 'day_of_month', 'month', 'is_weekend',
            'price_change', 'price_volatility', 'quantity_lag_1', 'quantity_lag_7',
            'quantity_mean_7d', 'quantity_std_7d', 'stock_velocity'
        ]
        
        # Remove rows with NaN in features
        df_clean = df.dropna(subset=feature_cols)
        
        if len(df_clean) < 10:
            return {'success': False, 'error': 'Insufficient clean data'}
        
        X = df_clean[feature_cols]
        y = df_clean['quantity']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.trained = True
        
        # Calculate training metrics
        train_score = self.model.score(X_scaled, y)
        
        return {
            'success': True,
            'train_score': round(train_score, 4),
            'samples': len(df_clean),
            'features': len(feature_cols)
        }
    
    def predict_demand(self, item_id: int, days_ahead: int = 7) -> Dict:
        """
        Predict demand for an item.
        
        Args:
            item_id: Item ID
            days_ahead: Days to forecast
        
        Returns:
            Demand predictions
        """
        if not self.trained:
            return {'success': False, 'error': 'Model not trained'}
        
        item = InventoryItem.query.get(item_id)
        if not item:
            return {'success': False, 'error': 'Item not found'}
        
        # Get recent data for context
        recent_data = self._get_historical_data(item.user_id, days=30)
        
        if recent_data.empty:
            return {'success': False, 'error': 'No historical data'}
        
        recent_data = self._extract_features(recent_data)
        item_data = recent_data[recent_data['item_id'] == item_id]
        
        if item_data.empty:
            # Use default features
            predictions = []
            for day in range(days_ahead):
                pred_date = datetime.utcnow() + timedelta(days=day+1)
                predictions.append({
                    'date': pred_date.strftime('%Y-%m-%d'),
                    'predicted_demand': int(item.quantity * 0.9),  # Conservative estimate
                    'confidence': 'low'
                })
            
            return {
                'success': True,
                'item_id': item_id,
                'item_name': item.item_name,
                'current_stock': item.quantity,
                'predictions': predictions
            }
        
        # Get latest values
        latest = item_data.iloc[-1]
        
        predictions = []
        for day in range(days_ahead):
            pred_date = datetime.utcnow() + timedelta(days=day+1)
            
            # Create feature vector
            features = {
                'price': item.price,
                'day_of_week': pred_date.weekday(),
                'day_of_month': pred_date.day,
                'month': pred_date.month,
                'is_weekend': int(pred_date.weekday() in [5, 6]),
                'price_change': latest['price_change'],
                'price_volatility': latest['price_volatility'],
                'quantity_lag_1': latest['quantity'],
                'quantity_lag_7': latest['quantity_lag_7'],
                'quantity_mean_7d': latest['quantity_mean_7d'],
                'quantity_std_7d': latest['quantity_std_7d'],
                'stock_velocity': latest['stock_velocity']
            }
            
            X = pd.DataFrame([features])
            X_scaled = self.scaler.transform(X)
            
            pred = self.model.predict(X_scaled)[0]
            pred = max(0, int(pred))  # Ensure non-negative
            
            # Estimate confidence based on historical volatility
            confidence = 'high' if latest['quantity_std_7d'] < 5 else 'medium' if latest['quantity_std_7d'] < 10 else 'low'
            
            predictions.append({
                'date': pred_date.strftime('%Y-%m-%d'),
                'predicted_demand': pred,
                'confidence': confidence
            })
        
        # Calculate reorder point
        avg_daily_demand = np.mean([p['predicted_demand'] for p in predictions])
        lead_time = 3  # days
        safety_stock = 2 * latest['quantity_std_7d']
        reorder_point = int(avg_daily_demand * lead_time + safety_stock)
        
        return {
            'success': True,
            'item_id': item_id,
            'item_name': item.item_name,
            'current_stock': item.quantity,
            'predictions': predictions,
            'analytics': {
                'avg_daily_demand': round(avg_daily_demand, 2),
                'reorder_point': reorder_point,
                'days_until_stockout': int(item.quantity / max(avg_daily_demand, 1)) if avg_daily_demand > 0 else 999,
                'recommended_order_quantity': max(0, reorder_point - item.quantity)
            }
        }
    
    def get_low_stock_alerts(self, user_id: int) -> List[Dict]:
        """
        Get items that need reordering based on predictions.
        
        Args:
            user_id: User ID
        
        Returns:
            List of items needing reorder
        """
        items = InventoryItem.query.filter_by(user_id=user_id).all()
        alerts = []
        
        for item in items:
            prediction = self.predict_demand(item.id, days_ahead=7)
            
            if prediction['success']:
                analytics = prediction.get('analytics', {})
                days_until_stockout = analytics.get('days_until_stockout', 999)
                
                if days_until_stockout < 7:
                    alerts.append({
                        'item_id': item.id,
                        'item_name': item.item_name,
                        'current_stock': item.quantity,
                        'days_until_stockout': days_until_stockout,
                        'recommended_order': analytics.get('recommended_order_quantity', 0),
                        'urgency': 'high' if days_until_stockout < 3 else 'medium'
                    })
        
        return sorted(alerts, key=lambda x: x['days_until_stockout'])


# Singleton instance
demand_forecasting_service = DemandForecastingService()
