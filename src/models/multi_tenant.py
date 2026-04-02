"""
Multi-tenant models for GROCERA marketplace

Defines Store, Customer, Order, and Khata models for multi-tenant architecture.
"""
from datetime import datetime
from decimal import Decimal
from typing import Optional, List
from src.models.models import db


class Store(db.Model):
    """
    Store model representing a Kirana shop
    
    Attributes:
        id: Primary key
        name: Store name
        owner_id: Foreign key to User (store owner)
        phone: Contact number
        address: Full address
        latitude: GPS coordinates
        longitude: GPS coordinates
        is_active: Store operational status
        credit_enabled: Whether store offers credit
        delivery_radius_km: Delivery coverage area
        created_at: Registration timestamp
    """
    __tablename__ = 'stores'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False, index=True)
    owner_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, unique=True)
    phone = db.Column(db.String(15), nullable=False)
    address = db.Column(db.Text, nullable=False)
    city = db.Column(db.String(100), index=True)
    state = db.Column(db.String(100))
    pincode = db.Column(db.String(10), index=True)
    latitude = db.Column(db.Numeric(10, 8))
    longitude = db.Column(db.Numeric(11, 8))
    
    # Business settings
    is_active = db.Column(db.Boolean, default=True, index=True)
    credit_enabled = db.Column(db.Boolean, default=False)
    delivery_radius_km = db.Column(db.Numeric(5, 2), default=5.0)
    min_order_amount = db.Column(db.Numeric(10, 2), default=0)
    delivery_fee = db.Column(db.Numeric(10, 2), default=0)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    customers = db.relationship('StoreCustomer', backref='store', lazy='dynamic', cascade='all, delete-orphan')
    orders = db.relationship('Order', backref='store', lazy='dynamic')
    
    def get_customer_count(self) -> int:
        """Get total number of linked customers"""
        return self.customers.filter_by(is_active=True).count()
    
    def get_total_orders(self) -> int:
        """Get total number of orders"""
        return self.orders.count()
    
    def get_revenue(self) -> Decimal:
        """Calculate total revenue"""
        from sqlalchemy import func
        result = db.session.query(func.sum(Order.total_amount)).filter(
            Order.store_id == self.id,
            Order.payment_status.in_(['paid', 'credit'])
        ).scalar()
        return result or Decimal('0.00')
    
    def __repr__(self) -> str:
        return f'<Store {self.name}>'


class StoreCustomer(db.Model):
    """
    Link table between Store and Customer with credit information
    
    Attributes:
        id: Primary key
        store_id: Foreign key to Store
        customer_id: Foreign key to User
        credit_limit: Maximum credit allowed
        outstanding_balance: Current credit balance
        trust_score: Trust rating (0-100)
        is_trusted: Trusted customer flag
        is_active: Relationship status
    """
    __tablename__ = 'store_customers'
    
    id = db.Column(db.Integer, primary_key=True)
    store_id = db.Column(db.Integer, db.ForeignKey('stores.id'), nullable=False)
    customer_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Credit management
    credit_limit = db.Column(db.Numeric(10, 2), default=0)
    outstanding_balance = db.Column(db.Numeric(10, 2), default=0)
    trust_score = db.Column(db.Integer, default=0)  # 0-100
    is_trusted = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)
    
    # Timestamps
    linked_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_purchase = db.Column(db.DateTime)
    
    # Relationships
    customer = db.relationship('User', foreign_keys=[customer_id], backref='store_links')
    
    # Unique constraint
    __table_args__ = (
        db.UniqueConstraint('store_id', 'customer_id', name='unique_store_customer'),
    )
    
    def has_credit_available(self, amount: Decimal) -> bool:
        """Check if customer has enough credit limit"""
        return (self.outstanding_balance + amount) <= self.credit_limit
    
    def get_order_count(self) -> int:
        """Get total orders for this customer at this store"""
        return Order.query.filter_by(
            store_id=self.store_id,
            customer_id=self.customer_id
        ).count()
    
    def __repr__(self) -> str:
        return f'<StoreCustomer store_id={self.store_id} customer_id={self.customer_id}>'


class Order(db.Model):
    """
    Order model representing a customer purchase
    
    Attributes:
        id: Primary key
        order_number: Unique order identifier
        store_id: Foreign key to Store
        customer_id: Foreign key to User
        total_amount: Total order value
        payment_method: Payment type (upi, cash, credit, card)
        payment_status: Payment state (pending, paid, credit)
        status: Order state (placed, confirmed, ready, delivered, cancelled)
        delivery_type: Pickup or delivery
        delivery_address: Delivery location
    """
    __tablename__ = 'orders'
    
    id = db.Column(db.Integer, primary_key=True)
    order_number = db.Column(db.String(20), unique=True, nullable=False, index=True)
    store_id = db.Column(db.Integer, db.ForeignKey('stores.id'), nullable=True)
    customer_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Financial details
    total_amount = db.Column(db.Numeric(10, 2), nullable=False)
    delivery_fee = db.Column(db.Numeric(10, 2), default=0)
    discount_amount = db.Column(db.Numeric(10, 2), default=0)
    final_amount = db.Column(db.Numeric(10, 2), nullable=False)
    
    # Payment
    payment_method = db.Column(db.String(20), nullable=False)  # upi, cash, credit, card
    payment_status = db.Column(db.String(20), default='pending')  # pending, paid, credit, failed
    payment_transaction_id = db.Column(db.String(100))
    
    # Fulfillment
    status = db.Column(db.String(20), default='placed', index=True)  # placed, confirmed, ready, delivered, cancelled
    delivery_type = db.Column(db.String(20), default='pickup')  # pickup, delivery
    delivery_address = db.Column(db.Text)
    delivery_instructions = db.Column(db.Text)
    
    # Timestamps
    placed_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    confirmed_at = db.Column(db.DateTime)
    ready_at = db.Column(db.DateTime)
    delivered_at = db.Column(db.DateTime)
    cancelled_at = db.Column(db.DateTime)
    
    # Additional info
    notes = db.Column(db.Text)
    cancellation_reason = db.Column(db.Text)
    
    # Relationships
    customer = db.relationship('User', foreign_keys=[customer_id], backref='orders')
    items = db.relationship('OrderItem', backref='order', lazy='dynamic', cascade='all, delete-orphan')
    khata_entry = db.relationship('KhataLedger', backref='order', uselist=False)
    
    @staticmethod
    def generate_order_number() -> str:
        """Generate unique order number"""
        import random
        import string
        timestamp = datetime.now().strftime('%Y%m%d')
        random_suffix = ''.join(random.choices(string.digits, k=6))
        return f"ORD{timestamp}{random_suffix}"
    
    def get_item_count(self) -> int:
        """Get total number of items in order"""
        return self.items.count()
    
    def can_cancel(self) -> bool:
        """Check if order can be cancelled"""
        return self.status in ['placed', 'confirmed']
    
    def __repr__(self) -> str:
        return f'<Order {self.order_number}>'


class OrderItem(db.Model):
    """
    Order line item representing individual products in an order
    
    Attributes:
        id: Primary key
        order_id: Foreign key to Order
        item_id: Foreign key to InventoryItem (nullable for deleted items)
        item_name: Product name (snapshot)
        quantity: Quantity ordered
        unit_price: Price at time of order
        subtotal: Line item total
    """
    __tablename__ = 'order_items'
    
    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey('orders.id'), nullable=False)
    item_id = db.Column(db.Integer, db.ForeignKey('inventory.id'))  # Nullable for deleted items
    
    # Snapshot data (in case item is deleted later)
    item_name = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(100))
    quantity = db.Column(db.Integer, nullable=False)
    unit = db.Column(db.String(50))
    unit_price = db.Column(db.Numeric(10, 2), nullable=False)
    subtotal = db.Column(db.Numeric(10, 2), nullable=False)
    
    # Expiry tracking
    expiry_date = db.Column(db.Date)
    
    # Relationships
    inventory_item = db.relationship('InventoryItem', backref='order_items')
    
    def __repr__(self) -> str:
        return f'<OrderItem {self.item_name} x{self.quantity}>'


class KhataLedger(db.Model):
    """
    Khata (credit ledger) for tracking customer credit transactions
    
    Attributes:
        id: Primary key
        store_id: Foreign key to Store
        customer_id: Foreign key to User
        order_id: Foreign key to Order (optional)
        transaction_type: Type (purchase, payment, adjustment)
        amount: Transaction amount
        balance_after: Running balance after transaction
        notes: Additional information
    """
    __tablename__ = 'khata_ledger'
    
    id = db.Column(db.Integer, primary_key=True)
    store_id = db.Column(db.Integer, db.ForeignKey('stores.id'), nullable=False)
    customer_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    order_id = db.Column(db.Integer, db.ForeignKey('orders.id'))
    
    # Transaction details
    transaction_type = db.Column(db.String(20), nullable=False)  # purchase, payment, adjustment
    amount = db.Column(db.Numeric(10, 2), nullable=False)
    balance_after = db.Column(db.Numeric(10, 2), nullable=False)
    payment_method = db.Column(db.String(20))  # For payment transactions
    transaction_reference = db.Column(db.String(100))
    
    # Metadata
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    
    # Relationships
    store = db.relationship('Store', foreign_keys=[store_id])
    customer = db.relationship('User', foreign_keys=[customer_id], backref='khata_entries')
    creator = db.relationship('User', foreign_keys=[created_by])
    
    @staticmethod
    def record_purchase(store_id: int, customer_id: int, order_id: int, amount: Decimal) -> 'KhataLedger':
        """Record a credit purchase"""
        store_customer = StoreCustomer.query.filter_by(
            store_id=store_id,
            customer_id=customer_id
        ).first()
        
        if not store_customer:
            raise ValueError("Customer not linked to store")
        
        new_balance = store_customer.outstanding_balance + amount
        
        if not store_customer.has_credit_available(amount):
            raise ValueError("Credit limit exceeded")
        
        entry = KhataLedger(
            store_id=store_id,
            customer_id=customer_id,
            order_id=order_id,
            transaction_type='purchase',
            amount=amount,
            balance_after=new_balance
        )
        
        store_customer.outstanding_balance = new_balance
        store_customer.last_purchase = datetime.utcnow()
        
        db.session.add(entry)
        db.session.commit()
        
        return entry
    
    @staticmethod
    def record_payment(store_id: int, customer_id: int, amount: Decimal, payment_method: str, notes: str = None) -> 'KhataLedger':
        """Record a credit payment"""
        store_customer = StoreCustomer.query.filter_by(
            store_id=store_id,
            customer_id=customer_id
        ).first()
        
        if not store_customer:
            raise ValueError("Customer not linked to store")
        
        new_balance = store_customer.outstanding_balance - amount
        
        if new_balance < 0:
            raise ValueError("Payment amount exceeds outstanding balance")
        
        entry = KhataLedger(
            store_id=store_id,
            customer_id=customer_id,
            transaction_type='payment',
            amount=amount,
            balance_after=new_balance,
            payment_method=payment_method,
            notes=notes
        )
        
        store_customer.outstanding_balance = new_balance
        
        db.session.add(entry)
        db.session.commit()
        
        return entry
    
    def __repr__(self) -> str:
        return f'<KhataLedger {self.transaction_type} ₹{self.amount}>'
