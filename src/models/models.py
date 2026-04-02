"""
Database Models for Grocera Inventory Management System

This module defines all SQLAlchemy models including:
- User: Authentication and user management
- InventoryItem: Product inventory tracking
- PriceHistory: Historical price data
- Alert: Notification system
- Cart: Shopping cart functionality
"""
from datetime import datetime
from typing import Optional, List
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import re

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """
    User model for authentication and authorization.
    
    Attributes:
        id: Primary key
        username: Unique username (3-80 chars)
        email: Unique email address
        password_hash: Bcrypt hashed password
        role: User role (admin, manager, viewer)
        created_at: Account creation timestamp
        is_active: Account status flag
    """
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default='viewer')  # admin, manager, viewer
    user_type = db.Column(db.String(20), default='customer')  # store_owner, customer, admin
    linked_store_id = db.Column(db.Integer, db.ForeignKey('stores.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    inventory_items = db.relationship('InventoryItem', backref='owner', lazy='dynamic', cascade='all, delete-orphan')
    alerts = db.relationship('Alert', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    owned_store = db.relationship('Store', foreign_keys='Store.owner_id', backref='owner', uselist=False)
    linked_store = db.relationship('Store', foreign_keys='[User.linked_store_id]', backref='linked_customers')
    
    def set_password(self, password: str) -> None:
        """Hash and set user password."""
        if len(password) < 6:
            raise ValueError("Password must be at least 6 characters")
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password: str) -> bool:
        """Verify password against stored hash."""
        return check_password_hash(self.password_hash, password)
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_username(username: str) -> bool:
        """Validate username (3-80 chars, alphanumeric and underscore only)."""
        return 3 <= len(username) <= 80 and username.replace('_', '').isalnum()
    
    def get_inventory_count(self) -> int:
        """Get total number of inventory items for this user."""
        return self.inventory_items.count()
    
    def get_unread_alerts(self) -> int:
        """Get count of unread alerts."""
        return self.alerts.filter_by(is_read=False).count()
    
    def __repr__(self) -> str:
        return f'<User {self.username}>'


class InventoryItem(db.Model):
    """
    Inventory item model for tracking products.
    
    Attributes:
        id: Primary key
        user_id: Foreign key to User
        item_name: Product name
        category: Product category
        price: Current selling price
        quantity: Stock quantity
        unit: Unit of measurement
        store_name: Store/supplier name
        stock_status: In Stock, Low Stock, Out of Stock
        expiry_date: Product expiration date
        base_price: Original/cost price
        is_discounted: Discount flag
    """
    __tablename__ = 'inventory'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    store_id = db.Column(db.Integer, db.ForeignKey('stores.id'), nullable=True, index=True)  # Multi-tenant support
    item_name = db.Column(db.String(200), nullable=False, index=True)
    category = db.Column(db.String(100), index=True)
    price = db.Column(db.Float, nullable=False)
    quantity = db.Column(db.Integer, default=0)
    unit = db.Column(db.String(50), default='units')
    store_name = db.Column(db.String(200))
    stock_status = db.Column(db.String(50), default='In Stock', index=True)
    expiry_date = db.Column(db.Date, index=True)
    base_price = db.Column(db.Float)
    is_discounted = db.Column(db.Boolean, default=False)
    barcode = db.Column(db.String(100), unique=True, index=True)  # Barcode/UPC/EAN code
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    price_history = db.relationship('PriceHistory', backref='item', lazy='dynamic', cascade='all, delete-orphan')
    store = db.relationship('Store', foreign_keys=[store_id], backref='inventory')
    
    @property
    def total_value(self) -> float:
        """Calculate total inventory value (price × quantity)."""
        return round(self.price * self.quantity, 2)
    
    @property
    def days_to_expiry(self) -> Optional[int]:
        """Calculate days until expiration."""
        if self.expiry_date:
            return (self.expiry_date - datetime.utcnow().date()).days
        return None
    
    @property
    def is_expired(self) -> bool:
        """Check if item has expired."""
        if self.expiry_date:
            return datetime.utcnow().date() > self.expiry_date
        return False
    
    @property
    def is_low_stock(self) -> bool:
        """Check if item is low in stock (< 5 units)."""
        return self.quantity < 5
    
    @property
    def discount_percentage(self) -> Optional[float]:
        """Calculate discount percentage if discounted."""
        if self.is_discounted and self.base_price and self.base_price > 0:
            return round(((self.base_price - self.price) / self.base_price) * 100, 2)
        return None
    
    def update_stock_status(self) -> None:
        """Auto-update stock status based on quantity."""
        if self.quantity == 0:
            self.stock_status = 'Out of Stock'
        elif self.quantity < 5:
            self.stock_status = 'Low Stock'
        else:
            self.stock_status = 'In Stock'
    
    def to_dict(self) -> dict:
        """Convert model to dictionary for API responses."""
        return {
            'id': self.id,
            'item_name': self.item_name,
            'category': self.category,
            'price': self.price,
            'quantity': self.quantity,
            'unit': self.unit,
            'total_value': self.total_value,
            'stock_status': self.stock_status,
            'days_to_expiry': self.days_to_expiry,
            'is_discounted': self.is_discounted,
            'discount_percentage': self.discount_percentage
        }
    
    def __repr__(self) -> str:
        return f'<InventoryItem {self.item_name}>'


class PriceHistory(db.Model):
    __tablename__ = 'price_history'
    
    id = db.Column(db.Integer, primary_key=True)
    item_id = db.Column(db.Integer, db.ForeignKey('inventory.id'), nullable=False, index=True)
    price = db.Column(db.Float, nullable=False)
    store_name = db.Column(db.String(200))
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f'<PriceHistory {self.item_id} - ${self.price}>'


class Alert(db.Model):
    __tablename__ = 'alerts'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    alert_type = db.Column(db.String(50), nullable=False)  # low_stock, expiry, price_change
    severity = db.Column(db.String(20), default='medium')  # low, medium, high
    item_name = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text)
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f'<Alert {self.alert_type} - {self.item_name}>'


class Cart(db.Model):
    """
    Shopping cart model for user purchases.
    
    Attributes:
        id: Primary key
        user_id: Foreign key to User
        item_id: Foreign key to InventoryItem
        quantity: Number of items in cart
        added_at: Timestamp when added to cart
    """
    __tablename__ = 'cart'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    item_id = db.Column(db.Integer, db.ForeignKey('inventory.id'), nullable=False)
    quantity = db.Column(db.Integer, default=1)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', backref='cart_items')
    item = db.relationship('InventoryItem', backref='in_carts')
    
    @property
    def subtotal(self) -> float:
        """Calculate subtotal for this cart item."""
        return round(self.item.price * self.quantity, 2)
    
    @property
    def is_available(self) -> bool:
        """Check if item has sufficient stock."""
        return self.item.quantity >= self.quantity
    
    def validate_quantity(self) -> bool:
        """Validate cart quantity against available stock."""
        return 0 < self.quantity <= self.item.quantity
    
    def __repr__(self) -> str:
        return f'<Cart User:{self.user_id} Item:{self.item_id} Qty:{self.quantity}>'


# Note: Order and OrderItem models moved to multi_tenant.py for better multi-tenant support
# Keeping this for backward compatibility - import from multi_tenant instead


class CompetitorPrice(db.Model):
    """
    Competitor pricing data model for price comparison.
    
    Attributes:
        id: Primary key
        item_id: Foreign key to InventoryItem
        competitor_name: Store/website name
        competitor_url: Product URL
        price: Competitor's price
        in_stock: Stock availability
        last_checked: Last scrape timestamp
    """
    __tablename__ = 'competitor_prices'
    
    id = db.Column(db.Integer, primary_key=True)
    item_id = db.Column(db.Integer, db.ForeignKey('inventory.id'), nullable=False, index=True)
    competitor_name = db.Column(db.String(200), nullable=False)
    competitor_url = db.Column(db.String(500))
    price = db.Column(db.Float, nullable=False)
    in_stock = db.Column(db.Boolean, default=True)
    last_checked = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    item = db.relationship('InventoryItem', backref='competitor_prices')
    
    @property
    def price_difference(self) -> Optional[float]:
        """Calculate price difference with our price."""
        if self.item:
            return round(self.item.price - self.price, 2)
        return None
    
    @property
    def is_cheaper(self) -> bool:
        """Check if competitor is cheaper."""
        return self.price_difference > 0 if self.price_difference else False
    
    def __repr__(self) -> str:
        return f'<CompetitorPrice {self.competitor_name} - ${self.price}>'
