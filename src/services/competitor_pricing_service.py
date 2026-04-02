"""
Competitor Pricing Service

Handles web scraping and API calls to fetch competitor prices.
Includes fallback for manual entry and caching.
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
from src.models.models import db, CompetitorPrice, InventoryItem
from datetime import datetime, timedelta
import time
import re


class CompetitorPricingService:
    """Service for scraping and managing competitor prices"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.cache_duration = timedelta(hours=24)  # Refresh every 24 hours
    
    def scrape_amazon_price(self, search_query: str) -> Optional[Dict]:
        """
        Scrape price from Amazon (demo implementation).
        
        Args:
            search_query: Product search query
        
        Returns:
            Dictionary with price data or None
        """
        try:
            # Note: Actual Amazon scraping requires handling CAPTCHA, rotating IPs, etc.
            # This is a simplified demo. Consider using Amazon Product API in production.
            
            url = f"https://www.amazon.com/s?k={search_query.replace(' ', '+')}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find first product price (simplified selector)
            price_elem = soup.find('span', {'class': 'a-price-whole'})
            if price_elem:
                price_text = price_elem.text.replace(',', '').replace('$', '').strip()
                price = float(price_text)
                
                return {
                    'competitor_name': 'Amazon',
                    'price': price,
                    'url': url,
                    'in_stock': True
                }
            
            return None
            
        except Exception as e:
            print(f"Amazon scraping error: {str(e)}")
            return None
    
    def scrape_walmart_price(self, search_query: str) -> Optional[Dict]:
        """
        Scrape price from Walmart.
        
        Args:
            search_query: Product search query
        
        Returns:
            Dictionary with price data or None
        """
        try:
            url = f"https://www.walmart.com/search?q={search_query.replace(' ', '+')}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Walmart uses dynamic content - this is simplified
            price_elem = soup.find('span', {'class': re.compile('price.*')})
            if price_elem:
                price_text = re.sub(r'[^\d.]', '', price_elem.text)
                if price_text:
                    price = float(price_text)
                    
                    return {
                        'competitor_name': 'Walmart',
                        'price': price,
                        'url': url,
                        'in_stock': True
                    }
            
            return None
            
        except Exception as e:
            print(f"Walmart scraping error: {str(e)}")
            return None
    
    def scrape_target_price(self, search_query: str) -> Optional[Dict]:
        """
        Scrape price from Target.
        
        Args:
            search_query: Product search query
        
        Returns:
            Dictionary with price data or None
        """
        try:
            url = f"https://www.target.com/s?searchTerm={search_query.replace(' ', '+')}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Target also uses dynamic content - simplified implementation
            price_elem = soup.find('span', {'data-test': 'product-price'})
            if price_elem:
                price_text = re.sub(r'[^\d.]', '', price_elem.text)
                if price_text:
                    price = float(price_text)
                    
                    return {
                        'competitor_name': 'Target',
                        'price': price,
                        'url': url,
                        'in_stock': True
                    }
            
            return None
            
        except Exception as e:
            print(f"Target scraping error: {str(e)}")
            return None
    
    def fetch_competitor_prices(self, item: InventoryItem, 
                               force_refresh: bool = False) -> List[CompetitorPrice]:
        """
        Fetch competitor prices for an item.
        
        Args:
            item: InventoryItem to search for
            force_refresh: Force refresh even if cached data exists
        
        Returns:
            List of CompetitorPrice objects
        """
        # Check cache first
        if not force_refresh:
            cached_prices = CompetitorPrice.query.filter_by(item_id=item.id).filter(
                CompetitorPrice.last_checked >= datetime.utcnow() - self.cache_duration
            ).all()
            
            if cached_prices:
                return cached_prices
        
        # Delete old prices
        CompetitorPrice.query.filter_by(item_id=item.id).delete()
        
        # Scrape new prices
        search_query = item.item_name
        competitors = []
        
        # Try each competitor with delay to avoid rate limiting
        scrapers = [
            self.scrape_amazon_price,
            self.scrape_walmart_price,
            self.scrape_target_price
        ]
        
        for scraper in scrapers:
            try:
                result = scraper(search_query)
                if result:
                    competitor = CompetitorPrice(
                        item_id=item.id,
                        competitor_name=result['competitor_name'],
                        competitor_url=result['url'],
                        price=result['price'],
                        in_stock=result['in_stock']
                    )
                    db.session.add(competitor)
                    competitors.append(competitor)
                
                # Delay between requests
                time.sleep(2)
                
            except Exception as e:
                print(f"Scraper error: {str(e)}")
                continue
        
        db.session.commit()
        return competitors
    
    def add_manual_competitor_price(self, item_id: int, competitor_name: str, 
                                   price: float, url: Optional[str] = None) -> CompetitorPrice:
        """
        Manually add competitor price (fallback when scraping fails).
        
        Args:
            item_id: Inventory item ID
            competitor_name: Name of competitor store
            price: Competitor's price
            url: Optional product URL
        
        Returns:
            Created CompetitorPrice object
        """
        competitor = CompetitorPrice(
            item_id=item_id,
            competitor_name=competitor_name,
            competitor_url=url,
            price=price,
            in_stock=True
        )
        db.session.add(competitor)
        db.session.commit()
        return competitor
    
    def get_best_price(self, item_id: int) -> Optional[Dict]:
        """
        Get the best (lowest) price among competitors.
        
        Args:
            item_id: Inventory item ID
        
        Returns:
            Dictionary with best price info or None
        """
        competitors = CompetitorPrice.query.filter_by(item_id=item_id).all()
        
        if not competitors:
            return None
        
        best = min(competitors, key=lambda x: x.price)
        
        return {
            'competitor_name': best.competitor_name,
            'price': best.price,
            'url': best.competitor_url,
            'savings': round(best.price_difference, 2) if best.price_difference else 0
        }
    
    def get_price_comparison(self, item_id: int) -> Dict:
        """
        Get full price comparison for an item.
        
        Args:
            item_id: Inventory item ID
        
        Returns:
            Dictionary with comparison data
        """
        item = InventoryItem.query.get(item_id)
        if not item:
            return {}
        
        competitors = CompetitorPrice.query.filter_by(item_id=item_id).all()
        
        comparison = {
            'our_price': item.price,
            'item_name': item.item_name,
            'competitors': []
        }
        
        for comp in competitors:
            comparison['competitors'].append({
                'name': comp.competitor_name,
                'price': comp.price,
                'difference': comp.price_difference,
                'is_cheaper': comp.is_cheaper,
                'url': comp.competitor_url,
                'last_checked': comp.last_checked.isoformat()
            })
        
        if competitors:
            comparison['lowest_price'] = min(c.price for c in competitors)
            comparison['highest_price'] = max(c.price for c in competitors)
            comparison['average_price'] = round(sum(c.price for c in competitors) / len(competitors), 2)
        
        return comparison
    
    def bulk_fetch_prices(self, user_id: int, limit: int = 10) -> Dict:
        """
        Fetch competitor prices for multiple items in bulk.
        
        Args:
            user_id: User ID
            limit: Maximum number of items to process
        
        Returns:
            Dictionary with results summary
        """
        items = InventoryItem.query.filter_by(user_id=user_id).limit(limit).all()
        
        results = {
            'total_items': len(items),
            'success': 0,
            'failed': 0,
            'items_processed': []
        }
        
        for item in items:
            try:
                competitors = self.fetch_competitor_prices(item)
                results['success'] += 1
                results['items_processed'].append({
                    'item_name': item.item_name,
                    'competitors_found': len(competitors)
                })
            except Exception as e:
                results['failed'] += 1
                print(f"Failed to fetch prices for {item.item_name}: {str(e)}")
        
        return results


# Singleton instance
competitor_pricing_service = CompetitorPricingService()
