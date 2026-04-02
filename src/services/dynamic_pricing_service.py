"""
Dynamic Pricing Engine

Uses Q-Learning (Reinforcement Learning) to optimize pricing dynamically.
"""

import numpy as np
import json
import os
from typing import Dict, Optional, List
from src.models.models import db, InventoryItem, CompetitorPrice
from datetime import datetime, timedelta


class DynamicPricingEngine:
    """Q-Learning based dynamic pricing engine"""
    
    def __init__(self):
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.2
        
        # State: (stock_level, competitor_position, demand_level)
        # Action: (decrease_price, keep_price, increase_price)
        self.q_table = {}
        self.price_actions = [-0.05, 0, 0.05]  # -5%, 0%, +5%
        
        self.load_q_table()
    
    def load_q_table(self):
        """Load Q-table from disk if exists"""
        path = 'processed/q_table.json'
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    self.q_table = json.load(f)
            except:
                self.q_table = {}
    
    def save_q_table(self):
        """Save Q-table to disk"""
        os.makedirs('processed', exist_ok=True)
        with open('processed/q_table.json', 'w') as f:
            json.dump(self.q_table, f)
    
    def _get_state(self, item: InventoryItem) -> tuple:
        """
        Get current state for an item.
        
        Args:
            item: Inventory item
        
        Returns:
            State tuple (stock_level, competitor_position, demand_level)
        """
        # Stock level: low (0), medium (1), high (2)
        if item.quantity < 10:
            stock_level = 0
        elif item.quantity < 50:
            stock_level = 1
        else:
            stock_level = 2
        
        # Competitor position: cheaper (0), same (1), expensive (2)
        competitors = CompetitorPrice.query.filter_by(item_id=item.id).all()
        if competitors:
            avg_competitor_price = np.mean([c.price for c in competitors])
            if item.price < avg_competitor_price * 0.95:
                competitor_position = 0
            elif item.price > avg_competitor_price * 1.05:
                competitor_position = 2
            else:
                competitor_position = 1
        else:
            competitor_position = 1
        
        # Demand level: low (0), medium (1), high (2)
        # Estimate from stock velocity
        demand_level = 1  # Default to medium
        
        return (stock_level, competitor_position, demand_level)
    
    def _get_q_value(self, state: tuple, action: int) -> float:
        """Get Q-value for state-action pair"""
        key = f"{state}_{action}"
        return self.q_table.get(key, 0.0)
    
    def _set_q_value(self, state: tuple, action: int, value: float):
        """Set Q-value for state-action pair"""
        key = f"{state}_{action}"
        self.q_table[key] = value
    
    def _calculate_reward(self, item: InventoryItem, old_price: float, 
                         new_price: float, sales: int) -> float:
        """
        Calculate reward for pricing action.
        
        Args:
            item: Item
            old_price: Previous price
            new_price: New price
            sales: Number of sales
        
        Returns:
            Reward value
        """
        # Revenue reward
        revenue = new_price * sales
        
        # Inventory penalty (holding cost)
        holding_penalty = -0.01 * item.quantity
        
        # Stockout penalty
        stockout_penalty = -10 if item.quantity == 0 else 0
        
        # Competitor advantage reward
        competitors = CompetitorPrice.query.filter_by(item_id=item.id).all()
        if competitors:
            avg_competitor_price = np.mean([c.price for c in competitors])
            if new_price < avg_competitor_price:
                competitor_reward = 5
            else:
                competitor_reward = -2
        else:
            competitor_reward = 0
        
        return revenue + holding_penalty + stockout_penalty + competitor_reward
    
    def select_action(self, state: tuple, explore: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            explore: Whether to explore
        
        Returns:
            Action index
        """
        if explore and np.random.random() < self.exploration_rate:
            # Explore: random action
            return np.random.randint(0, len(self.price_actions))
        else:
            # Exploit: best action
            q_values = [self._get_q_value(state, a) for a in range(len(self.price_actions))]
            return int(np.argmax(q_values))
    
    def update_q_value(self, state: tuple, action: int, reward: float, 
                      next_state: tuple):
        """
        Update Q-value using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        current_q = self._get_q_value(state, action)
        max_next_q = max([self._get_q_value(next_state, a) for a in range(len(self.price_actions))])
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self._set_q_value(state, action, new_q)
    
    def suggest_price(self, item_id: int) -> Dict:
        """
        Suggest optimal price for an item.
        
        Args:
            item_id: Item ID
        
        Returns:
            Price suggestion
        """
        item = InventoryItem.query.get(item_id)
        if not item:
            return {'success': False, 'error': 'Item not found'}
        
        state = self._get_state(item)
        action = self.select_action(state, explore=False)
        
        # Calculate new price
        price_change = self.price_actions[action]
        suggested_price = item.price * (1 + price_change)
        
        # Ensure reasonable bounds
        min_price = item.price * 0.7  # Max 30% discount
        max_price = item.price * 1.3  # Max 30% increase
        suggested_price = max(min_price, min(max_price, suggested_price))
        
        # Get competitor context
        competitors = CompetitorPrice.query.filter_by(item_id=item_id).all()
        competitor_prices = [c.price for c in competitors] if competitors else []
        
        return {
            'success': True,
            'item_id': item_id,
            'item_name': item.item_name,
            'current_price': round(item.price, 2),
            'suggested_price': round(suggested_price, 2),
            'price_change_percent': round(price_change * 100, 2),
            'state': {
                'stock_level': ['low', 'medium', 'high'][state[0]],
                'competitor_position': ['cheaper', 'competitive', 'expensive'][state[1]],
                'demand_level': ['low', 'medium', 'high'][state[2]]
            },
            'competitor_prices': competitor_prices,
            'reasoning': self._get_reasoning(state, action)
        }
    
    def _get_reasoning(self, state: tuple, action: int) -> str:
        """Generate human-readable reasoning for pricing decision"""
        stock_level, competitor_position, demand_level = state
        action_name = ['Decrease', 'Maintain', 'Increase'][action]
        
        reasons = []
        
        if stock_level == 0:
            reasons.append("Low stock levels")
        elif stock_level == 2:
            reasons.append("High inventory holding costs")
        
        if competitor_position == 0:
            reasons.append("Already competitive pricing")
        elif competitor_position == 2:
            reasons.append("Price above competitors")
        
        if demand_level == 2:
            reasons.append("High demand period")
        elif demand_level == 0:
            reasons.append("Low demand period")
        
        return f"{action_name} price based on: {', '.join(reasons)}"
    
    def apply_dynamic_pricing(self, user_id: int, auto_apply: bool = False) -> Dict:
        """
        Apply dynamic pricing to all items.
        
        Args:
            user_id: User ID
            auto_apply: Automatically apply suggested prices
        
        Returns:
            Pricing suggestions
        """
        items = InventoryItem.query.filter_by(user_id=user_id).all()
        suggestions = []
        
        for item in items:
            suggestion = self.suggest_price(item.id)
            
            if suggestion['success']:
                if auto_apply and suggestion['suggested_price'] != item.price:
                    # Save old price to history
                    old_price = item.price
                    item.price = suggestion['suggested_price']
                    
                    # Update base price if discounted
                    if not item.base_price or item.base_price < item.price:
                        item.base_price = old_price
                    
                    db.session.commit()
                    suggestion['applied'] = True
                else:
                    suggestion['applied'] = False
                
                suggestions.append(suggestion)
        
        return {
            'success': True,
            'total_items': len(items),
            'suggestions': suggestions
        }
    
    def train_from_history(self, user_id: int) -> Dict:
        """
        Train Q-learning model from historical data.
        
        Args:
            user_id: User ID
        
        Returns:
            Training results
        """
        # Simulate training with historical price changes
        items = InventoryItem.query.filter_by(user_id=user_id).all()
        
        updates = 0
        for item in items:
            state = self._get_state(item)
            
            # Simulate different actions and outcomes
            for action in range(len(self.price_actions)):
                # Estimate reward based on current state
                estimated_sales = 10 if state[2] == 2 else 5 if state[2] == 1 else 2
                reward = self._calculate_reward(item, item.price, 
                                              item.price * (1 + self.price_actions[action]),
                                              estimated_sales)
                
                # Update Q-value
                next_state = state  # Simplified
                self.update_q_value(state, action, reward, next_state)
                updates += 1
        
        self.save_q_table()
        
        return {
            'success': True,
            'updates': updates,
            'q_table_size': len(self.q_table)
        }


# Singleton instance
dynamic_pricing_engine = DynamicPricingEngine()
