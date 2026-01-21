# Outfit Generator - Multi-item outfit creation
# Generates complete outfits, not just single items

from typing import Dict, List, Any, Optional
from app.models.user_vector import FullUserVector, Outfit, OutfitItem
from app.db.mongodb import get_database
from app.services.fashion.scoring_engine import scoring_engine
from groq import Groq
from app.core.config import get_settings
import logging
import random
import json

logger = logging.getLogger(__name__)
settings = get_settings()




class OutfitGenerator:
    """
    Generates complete multi-item outfits from product database.
    
    An outfit consists of:
    - Top (shirt, t-shirt, blouse, etc.)
    - Bottom (pants, jeans, skirt, etc.)
    - Layer (optional - jacket, blazer, cardigan)
    - Shoes
    - Accessories (optional - watch, bag, jewelry)
    """
    
    # Category mappings - MUST MATCH DATABASE VALUES
    # Database uses: Hoodies, Jeans, Dresses, Footwear, Electronics, Clothing, Home, Uncategorized
    # Category groups
    TOP_CATEGORIES = ["Hoodies", "Clothing", "Shirts", "T-Shirts", "Tops", "Kurta", "Sweaters", "shirt", "t-shirt", "tshirt", "blouse", "top", "kurta", "sweater", "hoodie", "polo", "sweatshirt", "tank-top"]
    BOTTOM_CATEGORIES = ["Jeans", "Pants", "Trousers", "Skirts", "Shorts", "pants", "jeans", "trousers", "skirt", "shorts", "leggings", "joggers", "track-pants"]
    LAYER_CATEGORIES = ["Jackets", "Coats", "Blazers", "layer", "jacket", "coat", "blazer", "puffer", "trench-coat", "cardigan", "vest"]
    SHOE_CATEGORIES = ["Footwear", "Shoes", "Sneakers", "Boots", "Sandals", "Heels", "footwear", "sneakers", "boots", "sandals"]
    ACCESSORY_CATEGORIES = ["Accessories", "Bags", "Watches", "Belts", "accessories", "scarf", "shawl", "tie", "belt", "socks", "watch", "bag", "purse", "jewelry", "hat"]

    
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
    
    async def generate_outfits(
        self,
        query: str,
        user: FullUserVector,
        limit: int = 500
    ) -> List[Outfit]:
        """
        Generate outfit candidates based on user query and profile.
        
        Steps:
        1. Parse query for constraints
        2. Fetch candidate items from database
        3. Combine items into outfits
        4. Return as candidate list for scoring
        """
        db = get_database()
        
        # 1. Parse query for constraints
        constraints = await self._parse_query_constraints(query)
        logger.info(f"Parsed constraints: {constraints}")
        
        # 2. Build MongoDB queries for each category
        base_query = self._build_base_query(constraints, user)
        
        # 3. Fetch items by category
        tops = await self._fetch_items(db, "top", base_query, limit=20)
        bottoms = await self._fetch_items(db, "bottom", base_query, limit=15)
        layers = await self._fetch_items(db, "layer", base_query, limit=10)
        shoes = await self._fetch_items(db, "shoes", base_query, limit=10)
        accessories = await self._fetch_items(db, "accessories", base_query, limit=10)
        
        logger.info(f"Fetched items: tops={len(tops)}, bottoms={len(bottoms)}, layers={len(layers)}, shoes={len(shoes)}")
        
        # 4. Generate outfit combinations
        outfits = self._generate_combinations(
            tops, bottoms, layers, shoes, accessories,
            max_outfits=limit,
            user=user,
            constraints=constraints
        )
        
        return outfits
    
    async def _parse_query_constraints(self, query: str) -> Dict[str, Any]:
        """
        Use LLM to parse user query into structured constraints.
        """
        prompt = f"""
        Parse this fashion query into structured constraints.
        Query: "{query}"
        
        Return JSON with these fields (use null for unspecified):
        {{
            "color": "preferred color or null",
            "occasion": "casual/formal/party/date/work/wedding/street or null",
            "style": "streetwear/minimalist/ethnic/western/fusion or null",
            "formality": 0.0-1.0 or null,
            "vibe": "bold/subtle/playful/elegant or null",
            "budget_max": number or null,
            "must_include": ["specific item types to include"],
            "must_exclude": ["items or styles to avoid"],
            "season": "summer/winter/monsoon/all or null"
        }}
        """
        
        try:
            completion = self.client.chat.completions.create(
                model=settings.GROQ_MODEL_SMALL, # Reverting to 8B for sanity
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error parsing query: {e}")
            return {}
    
    def _build_base_query(
        self,
        constraints: Dict[str, Any],
        user: FullUserVector
    ) -> Dict[str, Any]:
        """
        Build MongoDB query from constraints and user profile.
        Excludes rejected colors and filters by budget.
        """
        query = {}
        
        # HARD EXCLUSION: Never show tech, electronics, or unrelated home goods in a fashion stylist app
        query["category"] = {"$nin": ["Electronics", "electronics", "laptop", "gadget", "mobile", "tech", "headphones", "Home", "Home Goods", "HomeDecor", "home"]}
        
        # 1. Color constraint (explicit preference)
        if constraints.get("color") and constraints["color"].lower() not in ["any", "all", ""]:
            query["color"] = {"$regex": constraints["color"], "$options": "i"}
        
        # 2. Exclude Rejected Colors and Must Exclude list
        rejected_colors = list(user.revealed_behavior.rejected_colors)
        
        # Add from query constraints
        must_exclude = constraints.get("must_exclude") or []
        if isinstance(must_exclude, list):
            rejected_colors.extend(must_exclude)
            
        if rejected_colors:
            # Filter out empty strings and normalize
            filtered_rejected = [c.lower() for c in rejected_colors if c and len(c) > 1]
            if filtered_rejected:
                if "color" not in query:
                    query["color"] = {}
                elif isinstance(query["color"], str) or "$regex" in query["color"]:
                    # If it was a simple regex, it already is a dict or we make it one
                    if not isinstance(query["color"], dict):
                        query["color"] = {"$regex": query["color"], "$options": "i"}
                
                query["color"]["$nin"] = filtered_rejected

        
        # 3. Budget constraint
        budget_max = constraints.get("budget_max")
        if budget_max and budget_max > 0:
            query["price"] = {"$lte": budget_max}
        
        return query


    
    async def _fetch_items(
        self,
        db,
        category_type: str,
        base_query: Dict[str, Any],
        limit: int = 20
    ) -> List[OutfitItem]:
        """
        Fetch items from database and convert to OutfitItem.
        """
        # Determine which categories to search
        if category_type == "top":
            categories = self.TOP_CATEGORIES
        elif category_type == "bottom":
            categories = self.BOTTOM_CATEGORIES
        elif category_type == "layer":
            categories = self.LAYER_CATEGORIES
        elif category_type == "shoes":
            categories = self.SHOE_CATEGORIES
        elif category_type == "accessories":
            categories = self.ACCESSORY_CATEGORIES
        else:
            categories = [category_type]
        
        # Build case-insensitive category regex
        category_regex = "|".join(categories)
        query = {**base_query}
        query["category"] = {"$regex": category_regex, "$options": "i"}
        
        try:
            cursor = db.products.find(query).limit(limit)
            items = await cursor.to_list(length=limit)
            
            outfit_items = []
            for item in items:
                # Defensive field extraction to avoid NoneType errors in Pydantic
                images = item.get("images") or []
                image_url = ""
                if images:
                    if isinstance(images[0], str):
                        image_url = images[0]
                    elif isinstance(images[0], dict):
                        image_url = images[0].get("path") or images[0].get("url") or ""

                outfit_items.append(OutfitItem(
                    item_id=str(item.get("_id")),
                    category=category_type, # Use the requested category_type for consistency
                    name=item.get("name") or "Unnamed Item",
                    color_vector=item.get("color_vector", [0, 0, 0]),
                    color_name=item.get("color") or "",
                    pattern=item.get("pattern") or "solid",
                    style_embedding=item.get("style_embedding", [0.0]*64),
                    fit=item.get("fit") or "regular",
                    formality=item.get("formality", 0.5),
                    fabric=item.get("fabric") or "",
                    price=float(item.get("price") or 0.0),
                    brand=item.get("brand") or "",
                    season=item.get("season") or [],
                    occasion=item.get("occasion") or [],
                    image_url=image_url
                ))
            
            return outfit_items
            
        except Exception as e:
            logger.error(f"Error fetching {category_type} items: {e}")
            return []
    
    def _generate_combinations(
        self,
        tops: List[OutfitItem],
        bottoms: List[OutfitItem],
        layers: List[OutfitItem],
        shoes: List[OutfitItem],
        accessories: List[OutfitItem],
        max_outfits: int,
        user: FullUserVector,
        constraints: Dict[str, Any]
    ) -> List[Outfit]:
        """
        Generate outfit combinations from item pools.
        Uses intelligent pairing, not random combinations.
        """
        outfits = []
        outfit_count = 0
        
        # If no tops or bottoms, we can't make outfits
        if not tops or not bottoms:
            logger.warning("COMBINE: Not enough items (tops/bottoms) to generate outfits")
            return outfits

        
        # Limit search space to avoid combinatorial explosion
        tops_pool = tops[:10]
        bottoms_pool = bottoms[:10]
        
        # Generate combinations
        for top in tops_pool:
            for bottom in bottoms_pool:
                if outfit_count >= max_outfits:
                    break
                
                # Basic color compatibility check
                if not self._colors_compatible(top.color_name, bottom.color_name):
                    continue
                
                # Create base outfit
                items = [top, bottom]
                
                # Add layer if needed (based on formality or weather)
                if constraints.get("formality", 0.5) > 0.5 or user.context.climate.temperature_normalized < 0.4:
                    if layers:
                        layer = self._pick_compatible_layer(top, bottom, layers)
                        if layer:
                            items.append(layer)
                
                # Add shoes
                if shoes:
                    shoe = self._pick_compatible_shoes(items, shoes, constraints)
                    if shoe:
                        items.append(shoe)
                
                # Optionally add accessory
                if accessories and random.random() > 0.5:
                    accessory = random.choice(accessories)
                    items.append(accessory)
                
                # Calculate outfit-level properties
                total_price = sum(item.price for item in items)
                avg_formality = sum(item.formality for item in items) / len(items)
                
                outfit = Outfit(
                    outfit_id=f"outfit_{outfit_count}",
                    items=items,
                    dominant_color=top.color_name,  # Top usually sets dominant color
                    total_price=total_price,
                    formality_score=avg_formality
                )
                
                outfits.append(outfit)
                outfit_count += 1
            
            if outfit_count >= max_outfits:
                break
        
        logger.info(f"Generated {len(outfits)} outfit combinations")
        return outfits
    
    def _colors_compatible(self, color1: str, color2: str) -> bool:
        """Check if two colors are compatible."""
        from app.services.fashion.knowledge_graph import ColorTheoryEngine
        if not hasattr(self, '_color_engine'):
            self._color_engine = ColorTheoryEngine()
        
        score = self._color_engine.calculate_harmony_score([color1, color2])
        return score > 0.5
    
    def _pick_compatible_layer(
        self,
        top: OutfitItem,
        bottom: OutfitItem,
        layers: List[OutfitItem]
    ) -> Optional[OutfitItem]:
        """Pick a compatible layer for the outfit."""
        compatible = []
        
        for layer in layers:
            # Layer should complement, not clash
            if self._colors_compatible(layer.color_name, top.color_name):
                # Neutral layers are safe
                if layer.color_name.lower() in ["black", "grey", "navy", "beige", "brown"]:
                    compatible.append((layer, 0.9))
                else:
                    compatible.append((layer, 0.6))
        
        if compatible:
            # Sort by compatibility score
            compatible.sort(key=lambda x: x[1], reverse=True)
            return compatible[0][0]
        
        return None
    
    def _pick_compatible_shoes(
        self,
        current_items: List[OutfitItem],
        shoes: List[OutfitItem],
        constraints: Dict[str, Any]
    ) -> Optional[OutfitItem]:
        """Pick compatible shoes for the outfit."""
        if not shoes:
            return None
        
        formality = constraints.get("formality", 0.5)
        
        compatible = []
        for shoe in shoes:
            # Match formality
            formality_match = 1 - abs(shoe.formality - formality)
            
            # Color compatibility
            colors = [item.color_name for item in current_items]
            color_compatible = any(
                self._colors_compatible(shoe.color_name, c) for c in colors
            )
            
            if color_compatible:
                compatible.append((shoe, formality_match))
        
        if compatible:
            compatible.sort(key=lambda x: x[1], reverse=True)
            return compatible[0][0]
        
        # Fallback to first shoe
        return shoes[0]
    
    async def generate_single_item_suggestions(
        self,
        query: str,
        user: FullUserVector,
        limit: int = 5
    ) -> List[OutfitItem]:
        """
        Fallback for single item queries (not full outfit).
        """
        db = get_database()
        constraints = await self._parse_query_constraints(query)
        base_query = self._build_base_query(constraints, user)
        
        # Search across all categories
        try:
            cursor = db.products.find(base_query).limit(limit * 2)
            items = await cursor.to_list(length=limit * 2)
            
            outfit_items = []
            for item in items:
                # Defensive field extraction to avoid NoneType errors in Pydantic
                images = item.get("images") or []
                image_url = ""
                if images:
                    if isinstance(images[0], str):
                        image_url = images[0]
                    elif isinstance(images[0], dict):
                        image_url = images[0].get("path") or images[0].get("url") or ""

                outfit_items.append(OutfitItem(
                    item_id=str(item.get("_id")),
                    category=item.get("category") or "",
                    name=item.get("name") or "Unnamed Item",
                    color_name=item.get("color") or "",
                    fit=item.get("fit") or "regular",
                    formality=item.get("formality", 0.5),
                    price=float(item.get("price") or 0.0),
                    brand=item.get("brand") or "",
                    image_url=image_url
                ))
            
            return outfit_items[:limit]
            
        except Exception as e:
            logger.error(f"Error generating single items: {e}")
            return []


# Singleton instance
outfit_generator = OutfitGenerator()
