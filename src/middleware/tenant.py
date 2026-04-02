"""
Multi-tenant context middleware for GROCERA

Handles store context for all requests, ensuring proper tenant isolation.
Follows Single Responsibility Principle - only manages tenant context.
"""
from flask import g, abort
from flask_login import current_user
from functools import wraps


class TenantContext:
    """Manages current request's tenant (store) context"""
    
    @staticmethod
    def get_current_store():
        """
        Get store from current request context
        
        Returns:
            Store: Current store object or None
        """
        if hasattr(g, 'store'):
            return g.store
        
        if not current_user.is_authenticated:
            return None
        
        # Import here to avoid circular dependency
        from src.models.models import Store
        
        if current_user.user_type == 'store_owner':
            g.store = Store.query.filter_by(owner_id=current_user.id, is_active=True).first()
        elif current_user.user_type == 'customer' and current_user.linked_store_id:
            g.store = Store.query.filter_by(id=current_user.linked_store_id, is_active=True).first()
        else:
            g.store = None
        
        return g.store
    
    @staticmethod
    def get_customer_stores():
        """
        Get all stores linked to current customer
        
        Returns:
            list: List of Store objects
        """
        if not current_user.is_authenticated or current_user.user_type != 'customer':
            return []
        
        from src.models.models import Store, StoreCustomer
        
        linked_stores = Store.query.join(StoreCustomer).filter(
            StoreCustomer.customer_id == current_user.id,
            Store.is_active == True
        ).all()
        
        return linked_stores


def store_required(f):
    """
    Decorator to ensure store context exists
    
    Usage:
        @store_required
        def my_route():
            store = TenantContext.get_current_store()
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        store = TenantContext.get_current_store()
        if not store:
            abort(403, description="No store context available. Please link to a store.")
        return f(*args, **kwargs)
    return decorated_function


def store_owner_required(f):
    """
    Decorator to ensure user is a store owner with active store
    
    Usage:
        @store_owner_required
        def manage_inventory():
            # Only store owners can access
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            abort(401, description="Authentication required")
        
        if current_user.user_type != 'store_owner':
            abort(403, description="Store owner access required")
        
        store = TenantContext.get_current_store()
        if not store:
            abort(403, description="No active store found")
        
        return f(*args, **kwargs)
    return decorated_function


def customer_required(f):
    """
    Decorator to ensure user is a customer
    
    Usage:
        @customer_required
        def place_order():
            # Only customers can access
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            abort(401, description="Authentication required")
        
        if current_user.user_type != 'customer':
            abort(403, description="Customer access required")
        
        return f(*args, **kwargs)
    return decorated_function
