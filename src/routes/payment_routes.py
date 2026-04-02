"""
Payment Routes Blueprint

Handles Stripe payment processing, checkout, and order management.
"""
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from src.models.models import db, Cart
from src.models.multi_tenant import Order
from src.services.payment_service import payment_service
from src.services.invoice_service import invoice_service
import os

payment_bp = Blueprint('payment', __name__, url_prefix='/payment')


@payment_bp.route('/checkout', methods=['GET', 'POST'])
@login_required
def checkout():
    """Display checkout page and create Stripe checkout session"""
    cart_items = Cart.query.filter_by(user_id=current_user.id).all()
    
    if not cart_items:
        flash('Your cart is empty', 'warning')
        return redirect(url_for('main.dashboard'))
    
    # Check if all items are available
    unavailable_items = [item for item in cart_items if not item.is_available]
    if unavailable_items:
        flash('Some items in your cart are no longer available', 'error')
        return redirect(url_for('main.dashboard'))
    
    if request.method == 'POST':
        try:
            # Create Stripe checkout session
            success_url = url_for('payment.success', _external=True) + '?session_id={CHECKOUT_SESSION_ID}'
            cancel_url = url_for('payment.cancel', _external=True)
            
            session = payment_service.create_checkout_session(
                user_id=current_user.id,
                cart_items=cart_items,
                success_url=success_url,
                cancel_url=cancel_url
            )
            if not session:
                flash('Unable to create checkout session', 'error')
                return redirect(url_for('payment.checkout'))
            
            return redirect(session['session_url'], code=303)
            
        except Exception as e:
            flash(f'Payment error: {str(e)}', 'error')
            return redirect(url_for('payment.checkout'))
    
    # Calculate total
    total = sum(item.item.price * item.quantity for item in cart_items)
    
    # Return JSON for now (frontend to be implemented)
    return jsonify({
        'cart_items': [{
            'item_name': item.item.item_name,
            'quantity': item.quantity,
            'price': item.item.price,
            'subtotal': item.item.price * item.quantity
        } for item in cart_items],
        'total': total,
        'message': 'Checkout page - frontend to be implemented'
    })


@payment_bp.route('/success')
@login_required
def success():
    """Handle successful payment"""
    session_id = request.args.get('session_id')
    
    if not session_id:
        flash('Invalid payment session', 'error')
        return redirect(url_for('main.dashboard'))
    
    try:
        # Complete the payment
        order = payment_service.complete_payment(session_id)
        
        flash(f'Payment successful! Order #{order.order_number} created.', 'success')
        return redirect(url_for('payment.order_detail', order_number=order.order_number))
        
    except Exception as e:
        flash(f'Payment processing error: {str(e)}', 'error')
        return redirect(url_for('main.dashboard'))


@payment_bp.route('/cancel')
@login_required
def cancel():
    """Handle cancelled payment"""
    session_id = request.args.get('session_id')
    
    if session_id:
        try:
            payment_service.cancel_payment(session_id)
        except:
            pass
    
    flash('Payment cancelled', 'info')
    return redirect(url_for('main.dashboard'))


@payment_bp.route('/orders')
@login_required
def order_history():
    """Display user's order history"""
    page = request.args.get('page', 1, type=int)
    per_page = 10
    
    orders, total = payment_service.get_order_history(
        user_id=current_user.id,
        page=page,
        per_page=per_page
    )
    
    # Return JSON for now (frontend to be implemented)
    return jsonify({
        'orders': [{
            'id': order.id,
            'order_number': order.order_number,
            'total_amount': float(order.total_amount or 0),
            'final_amount': float(order.final_amount or 0),
            'payment_method': order.payment_method,
            'payment_status': order.payment_status,
            'status': order.status,
            'placed_at': order.placed_at.isoformat() if order.placed_at else None
        } for order in orders],
        'total': total,
        'page': page,
        'per_page': per_page,
        'message': 'Order history - frontend to be implemented'
    })


@payment_bp.route('/orders/<order_number>')
@login_required
def order_detail(order_number):
    """Display order details"""
    order = Order.query.filter_by(
        order_number=order_number,
        customer_id=current_user.id
    ).first_or_404()
    
    # Return JSON for now (frontend to be implemented)
    return jsonify({
        'order': {
            'id': order.id,
            'order_number': order.order_number,
            'total_amount': float(order.total_amount or 0),
            'final_amount': float(order.final_amount or 0),
            'payment_method': order.payment_method,
            'payment_status': order.payment_status,
            'status': order.status,
            'placed_at': order.placed_at.isoformat() if order.placed_at else None,
            'items': [{
                'item_name': item.item_name,
                'quantity': item.quantity,
                'unit_price': float(item.unit_price or 0),
                'subtotal': float(item.subtotal or 0)
            } for item in order.items]
        },
        'message': 'Order detail - frontend to be implemented'
    })


@payment_bp.route('/orders/<order_number>/invoice')
@login_required
def download_invoice(order_number):
    """Generate and download invoice PDF"""
    order = Order.query.filter_by(
        order_number=order_number,
        customer_id=current_user.id
    ).first_or_404()
    
    try:
        # Generate invoice
        output_dir = 'static/invoices'
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f'invoice_{order_number}.pdf')
        invoice_service.generate_invoice(order, output_path)
        
        from flask import send_file
        return send_file(
            output_path,
            as_attachment=True,
            download_name=f'invoice_{order_number}.pdf',
            mimetype='application/pdf'
        )
        
    except Exception as e:
        flash(f'Invoice generation error: {str(e)}', 'error')
        return redirect(url_for('payment.order_detail', order_number=order_number))


@payment_bp.route('/webhook', methods=['POST'])
def stripe_webhook():
    """Handle Stripe webhook events"""
    import stripe
    
    payload = request.get_data()
    sig_header = request.headers.get('Stripe-Signature')
    webhook_secret = os.getenv('STRIPE_WEBHOOK_SECRET')
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, webhook_secret
        )
    except ValueError:
        return jsonify({'error': 'Invalid payload'}), 400
    except stripe.error.SignatureVerificationError:
        return jsonify({'error': 'Invalid signature'}), 400
    
    # Handle the event
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        payment_service.complete_payment(session['id'])
    
    elif event['type'] == 'payment_intent.payment_failed':
        payment_intent = event['data']['object']
        # Handle failed payment
        pass
    
    return jsonify({'status': 'success'}), 200


@payment_bp.route('/orders/<int:order_id>/refund', methods=['POST'])
@login_required
def refund_order(order_id):
    """Request order refund"""
    order = Order.query.filter_by(
        id=order_id,
        customer_id=current_user.id
    ).first_or_404()
    
    if order.status not in ['confirmed', 'ready', 'delivered']:
        flash('Only completed orders can be refunded', 'error')
        return redirect(url_for('payment.order_detail', order_number=order.order_number))
    
    try:
        payment_service.refund_order(order_id)
        flash('Refund processed successfully', 'success')
        
    except Exception as e:
        flash(f'Refund error: {str(e)}', 'error')
    
    return redirect(url_for('payment.order_detail', order_number=order.order_number))
