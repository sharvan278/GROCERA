"""
Utility Functions for Grocera Application

This module provides reusable utility functions for:
- File validation and processing
- Data formatting and serialization
- Date/time operations
- Validation helpers
- Error handling decorators
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from functools import wraps
from flask import flash
import logging

# Configure logging
logger = logging.getLogger(__name__)


# ============= File Operations =============

def allowed_file(filename: str, allowed_extensions: set = None) -> bool:
    """
    Check if file has an allowed extension.
    
    Args:
        filename: Name of the file to check
        allowed_extensions: Set of allowed extensions (default: csv, xlsx, xls)
    
    Returns:
        True if file extension is allowed, False otherwise
    """
    if allowed_extensions is None:
        allowed_extensions = {'csv', 'xlsx', 'xls'}
    
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


def get_file_size(filepath: str) -> Optional[int]:
    """
    Get file size in bytes.
    
    Args:
        filepath: Path to the file
    
    Returns:
        File size in bytes or None if file doesn't exist
    """
    try:
        return os.path.getsize(filepath)
    except OSError:
        return None


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def read_csv_safely(filepath: str, **kwargs) -> Optional[pd.DataFrame]:
    """
    Read CSV file with error handling.
    
    Args:
        filepath: Path to CSV file
        **kwargs: Additional arguments for pd.read_csv()
    
    Returns:
        DataFrame or None if error occurs
    """
    try:
        df = pd.read_csv(filepath, **kwargs)
        logger.info(f"Successfully read CSV: {filepath} ({len(df)} rows)")
        return df
    except Exception as e:
        logger.error(f"Error reading CSV {filepath}: {str(e)}")
        return None


# ============= Data Formatting =============

def format_currency(amount: float, currency_symbol: str = '$') -> str:
    """
    Format number as currency.
    
    Args:
        amount: Numeric amount
        currency_symbol: Currency symbol (default: $)
    
    Returns:
        Formatted currency string
    """
    return f"{currency_symbol}{amount:,.2f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format number as percentage.
    
    Args:
        value: Numeric value (0-100)
        decimals: Number of decimal places
    
    Returns:
        Formatted percentage string
    """
    return f"{value:.{decimals}f}%"


def truncate_string(text: str, max_length: int = 50, suffix: str = '...') -> str:
    """
    Truncate string to maximum length.
    
    Args:
        text: String to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
    
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


# ============= Date/Time Operations =============

def format_datetime(dt: datetime, format_string: str = '%Y-%m-%d %H:%M:%S') -> str:
    """
    Format datetime object.
    
    Args:
        dt: Datetime object
        format_string: Format string
    
    Returns:
        Formatted datetime string
    """
    return dt.strftime(format_string)


def get_days_until(target_date: datetime.date) -> int:
    """
    Calculate days from today until target date.
    
    Args:
        target_date: Target date
    
    Returns:
        Number of days (negative if in past)
    """
    today = datetime.now().date()
    return (target_date - today).days


def get_relative_time(dt: datetime) -> str:
    """
    Get relative time description (e.g., "2 hours ago").
    
    Args:
        dt: Datetime object
    
    Returns:
        Relative time string
    """
    now = datetime.now()
    diff = now - dt
    
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif seconds < 604800:
        days = int(seconds / 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"
    else:
        return dt.strftime('%Y-%m-%d')


# ============= Validation Helpers =============

def validate_positive_number(value: Any, field_name: str = "Value") -> Tuple[bool, Optional[str]]:
    """
    Validate that value is a positive number.
    
    Args:
        value: Value to validate
        field_name: Name of field for error message
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        num = float(value)
        if num <= 0:
            return False, f"{field_name} must be positive"
        return True, None
    except (ValueError, TypeError):
        return False, f"{field_name} must be a valid number"


def validate_date_range(start_date: datetime.date, end_date: datetime.date) -> Tuple[bool, Optional[str]]:
    """
    Validate date range.
    
    Args:
        start_date: Start date
        end_date: End date
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if start_date > end_date:
        return False, "Start date must be before end date"
    return True, None


def validate_required_fields(data: dict, required_fields: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate that all required fields are present and not empty.
    
    Args:
        data: Dictionary to validate
        required_fields: List of required field names
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    missing = []
    for field in required_fields:
        if field not in data or not data[field]:
            missing.append(field)
    
    if missing:
        return False, f"Missing required fields: {', '.join(missing)}"
    return True, None


# ============= Model Serialization =============

def serialize_item(item) -> dict:
    """
    Serialize inventory item model to dictionary.
    
    Args:
        item: InventoryItem model instance
    
    Returns:
        Dictionary representation
    """
    return {
        'id': item.id,
        'item_name': item.item_name,
        'category': item.category,
        'price': item.price,
        'quantity': item.quantity,
        'unit': item.unit,
        'store_name': item.store_name,
        'stock_status': item.stock_status,
        'total_value': item.total_value,
        'expiry_date': item.expiry_date.isoformat() if item.expiry_date else None,
        'days_to_expiry': item.days_to_expiry,
        'is_discounted': item.is_discounted,
        'base_price': item.base_price,
        'created_at': item.created_at.isoformat(),
        'updated_at': item.updated_at.isoformat()
    }


def serialize_user(user) -> dict:
    """
    Serialize user model to dictionary (excluding sensitive data).
    
    Args:
        user: User model instance
    
    Returns:
        Dictionary representation
    """
    return {
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'role': user.role,
        'is_active': user.is_active,
        'created_at': user.created_at.isoformat(),
        'inventory_count': user.get_inventory_count(),
        'unread_alerts': user.get_unread_alerts()
    }


# ============= Decorators =============

def handle_errors(flash_message: str = "An error occurred"):
    """
    Decorator to handle exceptions and show flash messages.
    
    Args:
        flash_message: Message to flash on error
    
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                flash(f"{flash_message}: {str(e)}", 'error')
                return None
        return wrapper
    return decorator


def log_execution_time(func):
    """
    Decorator to log function execution time.
    
    Args:
        func: Function to decorate
    
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"{func.__name__} executed in {execution_time:.3f}s")
        return result
    return wrapper


# ============= CSV Processing Helpers =============

def detect_csv_delimiter(filepath: str, sample_size: int = 5) -> str:
    """
    Auto-detect CSV delimiter.
    
    Args:
        filepath: Path to CSV file
        sample_size: Number of lines to sample
    
    Returns:
        Detected delimiter
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        sample = ''.join([f.readline() for _ in range(sample_size)])
    
    # Count occurrences of common delimiters
    delimiters = [',', ';', '\t', '|']
    counts = {d: sample.count(d) for d in delimiters}
    
    return max(counts, key=counts.get)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize DataFrame column names (lowercase, strip whitespace, replace spaces).
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with standardized columns
    """
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    return df


# ============= Business Logic Helpers =============

def calculate_discount_percentage(original_price: float, sale_price: float) -> float:
    """
    Calculate discount percentage.
    
    Args:
        original_price: Original price
        sale_price: Sale price
    
    Returns:
        Discount percentage
    """
    if original_price <= 0:
        return 0.0
    return round(((original_price - sale_price) / original_price) * 100, 2)


def categorize_stock_level(quantity: int, low_threshold: int = 5, critical_threshold: int = 2) -> str:
    """
    Categorize stock level.
    
    Args:
        quantity: Current stock quantity
        low_threshold: Threshold for low stock
        critical_threshold: Threshold for critical stock
    
    Returns:
        Stock level category
    """
    if quantity == 0:
        return 'Out of Stock'
    elif quantity <= critical_threshold:
        return 'Critical'
    elif quantity <= low_threshold:
        return 'Low Stock'
    else:
        return 'In Stock'


def get_expiry_urgency(days_to_expiry: int) -> str:
    """
    Get expiry urgency level.
    
    Args:
        days_to_expiry: Days until expiration
    
    Returns:
        Urgency level (critical, high, medium, low)
    """
    if days_to_expiry < 0:
        return 'expired'
    elif days_to_expiry <= 3:
        return 'critical'
    elif days_to_expiry <= 7:
        return 'high'
    elif days_to_expiry <= 30:
        return 'medium'
    else:
        return 'low'
