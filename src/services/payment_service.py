"""
Payment Service - Stripe Integration

Handles payment processing, checkout sessions, and webhooks for Stripe.
"""

import stripe
from typing import Dict, Optional, List
from src.config import Config
from src.models.models import db, Cart, InventoryItem
from src.models.multi_tenant import Order, OrderItem
from datetime import datetime
import os

# Initialize Stripe
stripe.api_key = os.getenv('STRIPE_SECRET_KEY', '')


class PaymentService:
    """Service for handling payment operations with Stripe"""
    
    @staticmethod
    def create_checkout_session(user_id: int, cart_items: List[Cart], 
                                success_url: str, cancel_url: str) -> Optional[Dict]:
        """
        Create a Stripe checkout session for cart items.
        
        Args:
            user_id: User ID making the purchase
            cart_items: List of cart items
            success_url: URL to redirect after successful payment
            cancel_url: URL to redirect if payment cancelled
        
        Returns:
            Dictionary with session details or None if error
        """
        try:
            # Calculate total and prepare line items
            line_items = []
            total_amount = 0
            
            for cart_item in cart_items:
                if not cart_item.is_available:
                    continue
                
                line_items.append({
                    'price_data': {
                        'currency': 'usd',
                        'unit_amount': int(cart_item.item.price * 100),  # Convert to cents
                        'product_data': {
                            'name': cart_item.item.item_name,
                            'description': f"Category: {cart_item.item.category or 'N/A'}",
                        },
                    },
                    'quantity': cart_item.quantity,
                })
                total_amount += cart_item.subtotal
            
            if not line_items:
                return None
            
            # Create Stripe checkout session
            session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=line_items,
                mode='payment',
                success_url=success_url + '?session_id={CHECKOUT_SESSION_ID}',
                cancel_url=cancel_url,
                client_reference_id=str(user_id),
                metadata={
                    'user_id': user_id,
                    'total_amount': total_amount
                }
            )
            
            # Create pending order
            order = PaymentService._create_pending_order(
                user_id=user_id,
                cart_items=cart_items,
                total_amount=total_amount,
                stripe_session_id=session.id
            )
            
            return {
                'session_id': session.id,
                'session_url': session.url,
                'order_id': order.id,
                'order_number': order.order_number
            }
            
        except stripe.error.StripeError as e:
            print(f"Stripe error: {str(e)}")
            return None
        except Exception as e:
            print(f"Payment error: {str(e)}")
            return None
    
    @staticmethod
    def _create_pending_order(user_id: int, cart_items: List[Cart], 
                            total_amount: float, stripe_session_id: str) -> Order:
        """Create a pending order from cart items."""
        store_id = None
        if cart_items and cart_items[0].item:
            store_id = getattr(cart_items[0].item, 'store_id', None)

        order = Order(
            store_id=store_id,
            customer_id=user_id,
            order_number=Order.generate_order_number(),
            total_amount=total_amount,
            final_amount=total_amount,
            status='placed',
            payment_method='card',
            payment_status='pending',
            payment_transaction_id=stripe_session_id,
            placed_at=datetime.utcnow()
        )
        db.session.add(order)
        db.session.flush()  # Get order ID
        
        # Create order items
        for cart_item in cart_items:
            order_item = OrderItem(
                order_id=order.id,
                item_id=cart_item.item_id,
                item_name=cart_item.item.item_name,
                category=cart_item.item.category,
                quantity=cart_item.quantity,
                unit='unit',
                unit_price=cart_item.item.price,
                subtotal=cart_item.subtotal
            )
            db.session.add(order_item)
        
        db.session.commit()
        return order
    
    @staticmethod
    def complete_payment(session_id: str) -> Optional[Order]:
        """
        Complete payment after successful Stripe checkout.
        
        Args:
            session_id: Stripe session ID
        
        Returns:
            Updated order or None if error
        """
        try:
            # Retrieve session from Stripe
            session = stripe.checkout.Session.retrieve(session_id)
            
            # Find order
            order = Order.query.filter_by(payment_transaction_id=session_id).first()
            if not order:
                return None
            
            # Update order status
            order.payment_status = 'paid'
            order.status = 'confirmed'
            order.payment_transaction_id = session.payment_intent or session_id
            order.confirmed_at = datetime.utcnow()
            
            # Update inventory quantities
            for order_item in order.items:
                if order_item.inventory_item:
                    order_item.inventory_item.quantity -= order_item.quantity
                    order_item.inventory_item.update_stock_status()
            
            # Clear user's cart
            Cart.query.filter_by(user_id=order.customer_id).delete()
            
            db.session.commit()
            return order
            
        except stripe.error.StripeError as e:
            print(f"Stripe error: {str(e)}")
            db.session.rollback()
            return None
        except Exception as e:
            print(f"Payment completion error: {str(e)}")
            db.session.rollback()
            return None
    
    @staticmethod
    def cancel_payment(session_id: str) -> bool:
        """
        Cancel a pending payment.
        
        Args:
            session_id: Stripe session ID
        
        Returns:
            True if successful, False otherwise
        """
        try:
            order = Order.query.filter_by(payment_transaction_id=session_id).first()
            if order:
                order.status = 'cancelled'
                order.payment_status = 'failed'
                order.cancelled_at = datetime.utcnow()
                db.session.commit()
            return True
        except Exception as e:
            print(f"Cancel payment error: {str(e)}")
            db.session.rollback()
            return False
    
    @staticmethod
    def get_order_history(user_id: int, page: int = 1, per_page: int = 20) -> tuple:
        """
        Get user's order history.
        
        Args:
            user_id: User ID
            page: Page number
            per_page: Items per page
        
        Returns:
            Tuple of (orders list, total count)
        """
        query = Order.query.filter_by(customer_id=user_id).order_by(Order.placed_at.desc())
        total = query.count()
        orders = query.limit(per_page).offset((page - 1) * per_page).all()
        return orders, total
    
    @staticmethod
    def get_order_by_number(order_number: str, user_id: int) -> Optional[Order]:
        """Get order by order number for specific user."""
        return Order.query.filter_by(order_number=order_number, customer_id=user_id).first()
    
    @staticmethod
    def refund_order(order_id: int) -> bool:
        """
        Process refund for an order.
        
        Args:
            order_id: Order ID to refund
        
        Returns:
            True if successful, False otherwise
        """
        try:
            order = Order.query.get(order_id)
            if not order or not order.payment_transaction_id:
                return False
            
            # Create refund in Stripe
            refund = stripe.Refund.create(
                payment_intent=order.payment_transaction_id
            )
            
            if refund.status == 'succeeded':
                order.payment_status = 'refunded'
                order.status = 'cancelled'
                order.cancelled_at = datetime.utcnow()
                
                # Restore inventory
                for order_item in order.items:
                    if order_item.inventory_item:
                        order_item.inventory_item.quantity += order_item.quantity
                        order_item.inventory_item.update_stock_status()
                
                db.session.commit()
                return True
            
            return False
            
        except stripe.error.StripeError as e:
            print(f"Refund error: {str(e)}")
            db.session.rollback()
            return False
        except Exception as e:
            print(f"Refund processing error: {str(e)}")
            db.session.rollback()
            return False


# Singleton instance
payment_service = PaymentService()
