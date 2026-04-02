"""
AI Service for Grocera - Grok API Integration
Handles all AI-powered features: chatbot, recommendations, bundling
"""
import os
import requests
from typing import List, Dict, Optional
from src.config import Config

class GrokAIService:
    """Service class for Grok AI integration"""
    
    def __init__(self):
        self.api_key = Config.AI_API_KEY
        self.base_url = Config.AI_API_BASE_URL.rstrip('/')
        self.model = Config.AI_MODEL
        self.provider = Config.AI_PROVIDER
        
    def chat_completion(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Send a chat completion request to Grok API
        
        Args:
            prompt: User's question or prompt
            context: Additional context (e.g., inventory data)
            
        Returns:
            AI response as string
        """
        if not self.api_key:
            return self._fallback_response(prompt, context)
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            messages = []
            
            # Enhanced system prompt with inventory context
            system_prompt = """You are a helpful grocery inventory assistant for Grocera platform. 
Your role is to help users manage their grocery inventory effectively.

When answering questions:
- Be specific and reference actual inventory items when possible
- Provide actionable recommendations
- Use a friendly, conversational tone
- Keep responses concise (2-3 sentences)
"""
            
            if context:
                system_prompt += f"\n\nCurrent user inventory:\n{context}\n\nUse this data to provide specific, relevant answers."
            
            messages.append({
                "role": "system",
                "content": system_prompt
            })
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=Config.AI_TIMEOUT
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                print(f"Grok API error: {response.status_code} - {response.text}")
                return self._fallback_response(prompt, context)
                
        except Exception as e:
            print(f"Grok API exception: {str(e)}")
            return self._fallback_response(prompt, context)
    
    def generate_bundles(self, inventory_items: List[str]) -> List[Dict]:
        """
        Generate product bundles using Grok AI
        
        Args:
            inventory_items: List of available product names
            
        Returns:
            List of bundle dictionaries with name, description, items
        """
        if not inventory_items or len(inventory_items) < 2:
            return self._default_bundles()
        
        prompt = f"""
        As a retail product bundling expert, suggest 5-7 creative product bundles 
        based on these grocery items: {', '.join(inventory_items[:20])}.
        
        For each bundle:
        1. Combine 2-4 complementary products
        2. Create a short name (max 3 words)
        3. One-sentence description (max 15 words)
        
        Format: name|description|item1,item2,item3
        Example: Breakfast Pack|Start your day right|bread,eggs,milk
        """
        
        try:
            response = self.chat_completion(prompt)
            bundles = self._parse_bundle_response(response, inventory_items)
            
            if bundles:
                return bundles
            else:
                return self._default_bundles()
                
        except Exception as e:
            print(f"Bundle generation error: {str(e)}")
            return self._default_bundles()
    
    def _parse_bundle_response(self, response: str, inventory_items: List[str]) -> List[Dict]:
        """Parse AI response into bundle format"""
        bundles = []
        lines = response.strip().split('\n')
        
        for line in lines:
            if '|' in line and line.count('|') >= 2:
                try:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        name = parts[0].strip()
                        desc = parts[1].strip()
                        items = [i.strip().lower() for i in parts[2].split(',')]
                        
                        # Validate items exist in inventory
                        valid_items = [item for item in items 
                                     if any(inv.lower() in item or item in inv.lower() 
                                           for inv in inventory_items)]
                        
                        if len(valid_items) >= 2:
                            bundles.append({
                                "name": name,
                                "desc": desc,
                                "items": set(valid_items)
                            })
                except Exception:
                    continue
        
        return bundles[:7]
    
    def _fallback_response(self, prompt: str, context: Optional[str] = None) -> str:
        """Enhanced fallback response when AI is unavailable"""
        prompt_lower = prompt.lower()
        
        # Parse context for specific answers - extract non-empty item names
        items_list = []
        items_with_details = []
        if context:
            for line in context.split('\n'):
                if ':' in line and line.strip():
                    item_name = line.split(':')[0].strip()
                    if item_name:  # Only add non-empty names
                        items_list.append(item_name)
                        items_with_details.append(line.strip())
        
        # Handle gym/fitness recommendations with smart product detection
        if any(word in prompt_lower for word in ['gym', 'fitness', 'workout', 'protein', 'muscle', 'athlete', 'exercise', 'bodybuilding', 'health']):
            protein_items = []
            healthy_items = []
            
            for line in items_with_details:
                item_lower = line.lower()
                # Look for protein-rich items
                if any(word in item_lower for word in ['protein', 'chicken', 'eggs', 'milk', 'yogurt', 'fish', 'beef', 'nuts', 'peanut', 'cheese', 'tofu', 'beans', 'lentils', 'meat', 'turkey', 'salmon']):
                    protein_items.append(line.split(':')[0].strip())
                # Look for healthy options
                elif any(word in item_lower for word in ['organic', 'fruit', 'vegetable', 'banana', 'apple', 'oats', 'quinoa', 'brown rice', 'whole grain', 'spinach', 'broccoli', 'avocado']):
                    healthy_items.append(line.split(':')[0].strip())
            
            if protein_items or healthy_items:
                response = "Perfect for your gym routine! Here are my recommendations from your inventory:\n\n"
                if protein_items:
                    response += f"💪 Protein Sources: {', '.join(protein_items[:5])}\n"
                if healthy_items:
                    response += f"🥗 Healthy Options: {', '.join(healthy_items[:5])}\n"
                response += "\nThese will help with muscle recovery and energy. Check your inventory for quantities!"
                return response
            else:
                return "For gym nutrition, I recommend stocking up on: protein sources (eggs, chicken, fish, yogurt, protein powder), complex carbs (oats, brown rice, quinoa), and fruits/vegetables. Check the Inventory page to see what you currently have and consider adding these items."
        
        if any(word in prompt_lower for word in ['inventory', 'products', 'items', 'have', 'stock']):
            if items_list:
                item_count = len(items_list)
                sample_items = ', '.join(items_list[:5])
                more = f" and {item_count - 5} more" if item_count > 5 else ""
                return f"You currently have {item_count} items in your inventory including: {sample_items}{more}. Check the Inventory page for full details."
            return "You don't have any items in inventory yet. Upload a CSV file to add products."
        
        if any(word in prompt_lower for word in ['price', 'cost', 'expensive', 'cheap']):
            return "I can help with pricing questions. Visit the Analytics page to see price trends and recommendations for optimizing your inventory costs."
        
        if any(word in prompt_lower for word in ['low', 'restock', 'order', 'buy']):
            return "Check the Analytics page for low stock alerts and restock recommendations based on your current inventory levels."
        
        if any(word in prompt_lower for word in ['expiry', 'expire', 'expiration', 'fresh']):
            return "Visit the Analytics page to see items expiring soon. I recommend checking items expiring within 7 days and planning to use or discount them."
        
        if any(word in prompt_lower for word in ['bundle', 'combo', 'together', 'suggest']):
            return "I can suggest product bundles based on your inventory. Common bundles include breakfast combos, baking kits, and meal prep packages."
        
        return "I'm here to help with inventory management, pricing insights, stock alerts, and product recommendations. What would you like to know?"
    
    def _default_bundles(self) -> List[Dict]:
        """Default bundles when AI is unavailable"""
        return [
            {"name": "Breakfast Essentials", "desc": "Complete breakfast package", 
             "items": {"bread", "eggs", "milk"}},
            {"name": "Protein Combo", "desc": "High protein snacks", 
             "items": {"eggs", "milk", "peanut butter"}},
            {"name": "Baking Kit", "desc": "Everything for baking", 
             "items": {"flour", "sugar", "eggs", "butter"}},
            {"name": "Fruit Basket", "desc": "Fresh fruit selection", 
             "items": {"bananas", "apples", "oranges"}},
            {"name": "Pantry Basics", "desc": "Kitchen essentials", 
             "items": {"rice", "oil", "salt", "sugar"}}
        ]


# Singleton instance
grok_service = GrokAIService()
