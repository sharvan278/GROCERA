"""
Competitor Pricing Routes Blueprint

Handles competitor price scraping and comparison.
"""
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from flask_login import login_required, current_user
from src.models.models import InventoryItem, CompetitorPrice
from src.services.competitor_pricing_service import competitor_pricing_service

competitor_bp = Blueprint('competitor', __name__, url_prefix='/competitor')


@competitor_bp.route('/prices/<int:item_id>')
@login_required
def item_prices(item_id):
    """Display competitor prices for an item"""
    item = InventoryItem.query.filter_by(
        id=item_id,
        user_id=current_user.id
    ).first_or_404()
    
    comparison = competitor_pricing_service.get_price_comparison(item_id)
    
    # Return JSON for now (frontend to be implemented)
    return jsonify({
        'item': {
            'id': item.id,
            'name': item.item_name,
            'price': item.price
        },
        'comparison': comparison,
        'message': 'Competitor prices - frontend to be implemented'
    })


@competitor_bp.route('/scrape/<int:item_id>', methods=['POST'])
@login_required
def scrape_prices(item_id):
    """Scrape competitor prices for an item"""
    item = InventoryItem.query.filter_by(
        id=item_id,
        user_id=current_user.id
    ).first_or_404()
    
    force_refresh = request.args.get('force', 'false').lower() == 'true'
    
    try:
        competitors = competitor_pricing_service.fetch_competitor_prices(
            item=item,
            force_refresh=force_refresh
        )
        
        return jsonify({
            'success': True,
            'message': f'Found {len(competitors)} competitor prices',
            'competitors': [
                {
                    'name': c.competitor_name,
                    'price': c.price,
                    'difference': c.price_difference,
                    'is_cheaper': c.is_cheaper
                }
                for c in competitors
            ]
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@competitor_bp.route('/add/<int:item_id>', methods=['POST'])
@login_required
def add_manual_price(item_id):
    """Manually add competitor price"""
    item = InventoryItem.query.filter_by(
        id=item_id,
        user_id=current_user.id
    ).first_or_404()
    
    data = request.get_json()
    
    try:
        competitor = competitor_pricing_service.add_manual_competitor_price(
            item_id=item_id,
            competitor_name=data['competitor_name'],
            price=float(data['price']),
            url=data.get('url')
        )
        
        return jsonify({
            'success': True,
            'message': 'Competitor price added',
            'competitor': {
                'name': competitor.competitor_name,
                'price': competitor.price,
                'difference': competitor.price_difference
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@competitor_bp.route('/bulk-scrape', methods=['POST'])
@login_required
def bulk_scrape():
    """Scrape prices for multiple items"""
    limit = request.args.get('limit', 10, type=int)
    
    try:
        results = competitor_pricing_service.bulk_fetch_prices(
            user_id=current_user.id,
            limit=limit
        )
        
        flash(f"Scraped {results['success']} items successfully", 'success')
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@competitor_bp.route('/best-prices')
@login_required
def best_prices():
    """Display items with best competitor prices"""
    items = InventoryItem.query.filter_by(user_id=current_user.id).all()
    
    price_data = []
    for item in items:
        best = competitor_pricing_service.get_best_price(item.id)
        if best:
            price_data.append({
                'item': item,
                'best_price': best
            })
    
    # Sort by savings
    price_data.sort(key=lambda x: x['best_price']['savings'], reverse=True)
    
    # Return JSON for now (frontend to be implemented)
    return jsonify({
        'price_data': [{
            'item': {
                'id': data['item'].id,
                'name': data['item'].item_name,
                'price': data['item'].price
            },
            'best_price': data['best_price']
        } for data in price_data],
        'message': 'Best prices - frontend to be implemented'
    })
