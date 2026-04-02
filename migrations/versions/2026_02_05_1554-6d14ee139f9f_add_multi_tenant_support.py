"""add_multi_tenant_support

Revision ID: 6d14ee139f9f
Revises: 86da9be7fcab
Create Date: 2026-02-05 15:54:39.450493

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '6d14ee139f9f'
down_revision = '86da9be7fcab'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    def table_exists(table_name: str) -> bool:
        return table_name in inspector.get_table_names()

    def column_exists(table_name: str, column_name: str) -> bool:
        if not table_exists(table_name):
            return False
        return column_name in [c['name'] for c in inspector.get_columns(table_name)]

    def index_exists(table_name: str, index_name: str) -> bool:
        if not table_exists(table_name):
            return False
        return index_name in [i['name'] for i in inspector.get_indexes(table_name)]

    def fk_exists(table_name: str, fk_name: str) -> bool:
        if not table_exists(table_name):
            return False
        return fk_name in [fk['name'] for fk in inspector.get_foreign_keys(table_name) if fk.get('name')]

    # Create stores table
    if not table_exists('stores'):
        op.create_table('stores',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('name', sa.String(length=255), nullable=False),
            sa.Column('owner_id', sa.Integer(), nullable=False),
            sa.Column('phone', sa.String(length=15), nullable=False),
            sa.Column('address', sa.Text(), nullable=False),
            sa.Column('city', sa.String(length=100), nullable=True),
            sa.Column('state', sa.String(length=100), nullable=True),
            sa.Column('pincode', sa.String(length=10), nullable=True),
            sa.Column('latitude', sa.Numeric(precision=10, scale=8), nullable=True),
            sa.Column('longitude', sa.Numeric(precision=11, scale=8), nullable=True),
            sa.Column('is_active', sa.Boolean(), nullable=True),
            sa.Column('credit_enabled', sa.Boolean(), nullable=True),
            sa.Column('delivery_radius_km', sa.Numeric(precision=5, scale=2), nullable=True),
            sa.Column('min_order_amount', sa.Numeric(precision=10, scale=2), nullable=True),
            sa.Column('delivery_fee', sa.Numeric(precision=10, scale=2), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.Column('updated_at', sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('owner_id')
        )
        inspector = sa.inspect(conn)

    if not index_exists('stores', op.f('ix_stores_city')):
        op.create_index(op.f('ix_stores_city'), 'stores', ['city'], unique=False)
    if not index_exists('stores', op.f('ix_stores_is_active')):
        op.create_index(op.f('ix_stores_is_active'), 'stores', ['is_active'], unique=False)
    if not index_exists('stores', op.f('ix_stores_name')):
        op.create_index(op.f('ix_stores_name'), 'stores', ['name'], unique=False)
    if not index_exists('stores', op.f('ix_stores_pincode')):
        op.create_index(op.f('ix_stores_pincode'), 'stores', ['pincode'], unique=False)

    # Create store_customers table
    if not table_exists('store_customers'):
        op.create_table('store_customers',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('store_id', sa.Integer(), nullable=False),
            sa.Column('customer_id', sa.Integer(), nullable=False),
            sa.Column('credit_limit', sa.Numeric(precision=10, scale=2), nullable=True),
            sa.Column('outstanding_balance', sa.Numeric(precision=10, scale=2), nullable=True),
            sa.Column('trust_score', sa.Integer(), nullable=True),
            sa.Column('is_trusted', sa.Boolean(), nullable=True),
            sa.Column('is_active', sa.Boolean(), nullable=True),
            sa.Column('linked_at', sa.DateTime(), nullable=True),
            sa.Column('last_purchase', sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(['customer_id'], ['users.id'], ),
            sa.ForeignKeyConstraint(['store_id'], ['stores.id'], ),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('store_id', 'customer_id', name='unique_store_customer')
        )
        inspector = sa.inspect(conn)

    # Modify existing orders table for multi-tenant support
    if not column_exists('orders', 'store_id'):
        op.add_column('orders', sa.Column('store_id', sa.Integer(), nullable=True))
    if not column_exists('orders', 'customer_id'):
        op.add_column('orders', sa.Column('customer_id', sa.Integer(), nullable=True))
    if not column_exists('orders', 'delivery_fee'):
        op.add_column('orders', sa.Column('delivery_fee', sa.Numeric(precision=10, scale=2), nullable=True))
    if not column_exists('orders', 'discount_amount'):
        op.add_column('orders', sa.Column('discount_amount', sa.Numeric(precision=10, scale=2), nullable=True))
    if not column_exists('orders', 'final_amount'):
        op.add_column('orders', sa.Column('final_amount', sa.Numeric(precision=10, scale=2), nullable=True))
    if not column_exists('orders', 'delivery_type'):
        op.add_column('orders', sa.Column('delivery_type', sa.String(length=20), nullable=True))
    if not column_exists('orders', 'delivery_address'):
        op.add_column('orders', sa.Column('delivery_address', sa.Text(), nullable=True))
    if not column_exists('orders', 'delivery_instructions'):
        op.add_column('orders', sa.Column('delivery_instructions', sa.Text(), nullable=True))
    if not column_exists('orders', 'placed_at'):
        op.add_column('orders', sa.Column('placed_at', sa.DateTime(), nullable=True))
    if not column_exists('orders', 'confirmed_at'):
        op.add_column('orders', sa.Column('confirmed_at', sa.DateTime(), nullable=True))
    if not column_exists('orders', 'ready_at'):
        op.add_column('orders', sa.Column('ready_at', sa.DateTime(), nullable=True))
    if not column_exists('orders', 'delivered_at'):
        op.add_column('orders', sa.Column('delivered_at', sa.DateTime(), nullable=True))
    if not column_exists('orders', 'cancelled_at'):
        op.add_column('orders', sa.Column('cancelled_at', sa.DateTime(), nullable=True))
    if not column_exists('orders', 'cancellation_reason'):
        op.add_column('orders', sa.Column('cancellation_reason', sa.Text(), nullable=True))
    if not column_exists('orders', 'payment_transaction_id'):
        op.add_column('orders', sa.Column('payment_transaction_id', sa.String(length=100), nullable=True))

    inspector = sa.inspect(conn)
    if column_exists('orders', 'customer_id') and column_exists('orders', 'user_id'):
        op.execute("UPDATE orders SET customer_id = user_id WHERE customer_id IS NULL")
    if column_exists('orders', 'placed_at') and column_exists('orders', 'created_at'):
        op.execute("UPDATE orders SET placed_at = created_at WHERE placed_at IS NULL")
    if column_exists('orders', 'final_amount') and column_exists('orders', 'total_amount'):
        op.execute("UPDATE orders SET final_amount = total_amount WHERE final_amount IS NULL")
    if column_exists('orders', 'delivery_address') and column_exists('orders', 'shipping_address'):
        op.execute("UPDATE orders SET delivery_address = shipping_address WHERE delivery_address IS NULL AND shipping_address IS NOT NULL")

    if not index_exists('orders', op.f('ix_orders_placed_at')):
        op.create_index(op.f('ix_orders_placed_at'), 'orders', ['placed_at'], unique=False)

    if not fk_exists('orders', 'fk_orders_store'):
        op.create_foreign_key('fk_orders_store', 'orders', 'stores', ['store_id'], ['id'])
    if not fk_exists('orders', 'fk_orders_customer'):
        op.create_foreign_key('fk_orders_customer', 'orders', 'users', ['customer_id'], ['id'])

    # Modify existing order_items table
    if not column_exists('order_items', 'category'):
        op.add_column('order_items', sa.Column('category', sa.String(length=100), nullable=True))
    if not column_exists('order_items', 'unit'):
        op.add_column('order_items', sa.Column('unit', sa.String(length=50), nullable=True))
    if not column_exists('order_items', 'expiry_date'):
        op.add_column('order_items', sa.Column('expiry_date', sa.Date(), nullable=True))

    # Create khata_ledger table
    if not table_exists('khata_ledger'):
        op.create_table('khata_ledger',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('store_id', sa.Integer(), nullable=False),
            sa.Column('customer_id', sa.Integer(), nullable=False),
            sa.Column('order_id', sa.Integer(), nullable=True),
            sa.Column('transaction_type', sa.String(length=20), nullable=False),
            sa.Column('amount', sa.Numeric(precision=10, scale=2), nullable=False),
            sa.Column('balance_after', sa.Numeric(precision=10, scale=2), nullable=False),
            sa.Column('payment_method', sa.String(length=20), nullable=True),
            sa.Column('transaction_reference', sa.String(length=100), nullable=True),
            sa.Column('notes', sa.Text(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.Column('created_by', sa.Integer(), nullable=True),
            sa.ForeignKeyConstraint(['created_by'], ['users.id'], ),
            sa.ForeignKeyConstraint(['customer_id'], ['users.id'], ),
            sa.ForeignKeyConstraint(['order_id'], ['orders.id'], ),
            sa.ForeignKeyConstraint(['store_id'], ['stores.id'], ),
            sa.PrimaryKeyConstraint('id')
        )
        inspector = sa.inspect(conn)
    if not index_exists('khata_ledger', op.f('ix_khata_ledger_created_at')):
        op.create_index(op.f('ix_khata_ledger_created_at'), 'khata_ledger', ['created_at'], unique=False)

    # Add new columns to users table
    if not column_exists('users', 'user_type'):
        op.add_column('users', sa.Column('user_type', sa.String(length=20), nullable=True))
    if not column_exists('users', 'linked_store_id'):
        op.add_column('users', sa.Column('linked_store_id', sa.Integer(), nullable=True))
    if not fk_exists('users', 'fk_users_linked_store'):
        op.create_foreign_key('fk_users_linked_store', 'users', 'stores', ['linked_store_id'], ['id'])

    # Add store_id to inventory table
    if not column_exists('inventory', 'store_id'):
        op.add_column('inventory', sa.Column('store_id', sa.Integer(), nullable=True))
    if not index_exists('inventory', op.f('ix_inventory_store_id')):
        op.create_index(op.f('ix_inventory_store_id'), 'inventory', ['store_id'], unique=False)
    if not fk_exists('inventory', 'fk_inventory_store'):
        op.create_foreign_key('fk_inventory_store', 'inventory', 'stores', ['store_id'], ['id'])


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    def table_exists(table_name: str) -> bool:
        return table_name in inspector.get_table_names()

    def column_exists(table_name: str, column_name: str) -> bool:
        if not table_exists(table_name):
            return False
        return column_name in [c['name'] for c in inspector.get_columns(table_name)]

    def index_exists(table_name: str, index_name: str) -> bool:
        if not table_exists(table_name):
            return False
        return index_name in [i['name'] for i in inspector.get_indexes(table_name)]

    def fk_exists(table_name: str, fk_name: str) -> bool:
        if not table_exists(table_name):
            return False
        return fk_name in [fk['name'] for fk in inspector.get_foreign_keys(table_name) if fk.get('name')]

    # Drop foreign keys and columns from inventory
    if fk_exists('inventory', 'fk_inventory_store'):
        op.drop_constraint('fk_inventory_store', 'inventory', type_='foreignkey')
    if index_exists('inventory', op.f('ix_inventory_store_id')):
        op.drop_index(op.f('ix_inventory_store_id'), table_name='inventory')
    if column_exists('inventory', 'store_id'):
        op.drop_column('inventory', 'store_id')

    # Drop foreign key and columns from users
    if fk_exists('users', 'fk_users_linked_store'):
        op.drop_constraint('fk_users_linked_store', 'users', type_='foreignkey')
    if column_exists('users', 'linked_store_id'):
        op.drop_column('users', 'linked_store_id')
    if column_exists('users', 'user_type'):
        op.drop_column('users', 'user_type')

    # Drop khata_ledger table
    if index_exists('khata_ledger', op.f('ix_khata_ledger_created_at')):
        op.drop_index(op.f('ix_khata_ledger_created_at'), table_name='khata_ledger')
    if table_exists('khata_ledger'):
        op.drop_table('khata_ledger')

    # Revert order_items modifications
    if column_exists('order_items', 'expiry_date'):
        op.drop_column('order_items', 'expiry_date')
    if column_exists('order_items', 'unit'):
        op.drop_column('order_items', 'unit')
    if column_exists('order_items', 'category'):
        op.drop_column('order_items', 'category')

    # Revert orders modifications
    if index_exists('orders', op.f('ix_orders_placed_at')):
        op.drop_index(op.f('ix_orders_placed_at'), table_name='orders')
    if fk_exists('orders', 'fk_orders_customer'):
        op.drop_constraint('fk_orders_customer', 'orders', type_='foreignkey')
    if fk_exists('orders', 'fk_orders_store'):
        op.drop_constraint('fk_orders_store', 'orders', type_='foreignkey')
    if column_exists('orders', 'payment_transaction_id'):
        op.drop_column('orders', 'payment_transaction_id')
    if column_exists('orders', 'cancellation_reason'):
        op.drop_column('orders', 'cancellation_reason')
    if column_exists('orders', 'cancelled_at'):
        op.drop_column('orders', 'cancelled_at')
    if column_exists('orders', 'delivered_at'):
        op.drop_column('orders', 'delivered_at')
    if column_exists('orders', 'ready_at'):
        op.drop_column('orders', 'ready_at')
    if column_exists('orders', 'confirmed_at'):
        op.drop_column('orders', 'confirmed_at')
    if column_exists('orders', 'placed_at'):
        op.drop_column('orders', 'placed_at')
    if column_exists('orders', 'delivery_instructions'):
        op.drop_column('orders', 'delivery_instructions')
    if column_exists('orders', 'delivery_address'):
        op.drop_column('orders', 'delivery_address')
    if column_exists('orders', 'delivery_type'):
        op.drop_column('orders', 'delivery_type')
    if column_exists('orders', 'final_amount'):
        op.drop_column('orders', 'final_amount')
    if column_exists('orders', 'discount_amount'):
        op.drop_column('orders', 'discount_amount')
    if column_exists('orders', 'delivery_fee'):
        op.drop_column('orders', 'delivery_fee')
    if column_exists('orders', 'customer_id'):
        op.drop_column('orders', 'customer_id')
    if column_exists('orders', 'store_id'):
        op.drop_column('orders', 'store_id')

    # Drop store_customers table
    if table_exists('store_customers'):
        op.drop_table('store_customers')

    # Drop stores table
    if index_exists('stores', op.f('ix_stores_pincode')):
        op.drop_index(op.f('ix_stores_pincode'), table_name='stores')
    if index_exists('stores', op.f('ix_stores_name')):
        op.drop_index(op.f('ix_stores_name'), table_name='stores')
    if index_exists('stores', op.f('ix_stores_is_active')):
        op.drop_index(op.f('ix_stores_is_active'), table_name='stores')
    if index_exists('stores', op.f('ix_stores_city')):
        op.drop_index(op.f('ix_stores_city'), table_name='stores')
    if table_exists('stores'):
        op.drop_table('stores')
