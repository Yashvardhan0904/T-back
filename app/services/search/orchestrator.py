from app.db.mongodb import get_database
from app.services.search.strategy import search_strategy
from app.services.embeddings.service import embedding_service
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class SearchOrchestrator:
    async def hybrid_search(self, query: str, limit: int = 5, exclude_ids: List[str] = None, preferences: Dict[str, Any] = None, mandatory_filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        1. Handle special cases (Trending)
        2. Merge mandatory filters
        3. Extract attributes using Groq
        4. Try MongoDB filtered search with exclusions
        5. Perform relaxation if needed (preserving mandatory filters)
        """
        # 1. SPECIAL CASE: Trending
        if query.lower() in ["trending", "popular", "new arrivals"]:
            return await self._execute_search({}, limit, sort_by_new=True)

        filters = await search_strategy.extract_filters(query)
        
        # 2. Construct MongoDB Filter Query
        mongo_query = {}
        
        # Apply Mandatory Filters First (Gender, Exclusions, etc.)
        if mandatory_filters:
            if "gender" in mandatory_filters:
                gender_val = mandatory_filters["gender"]
                if isinstance(gender_val, list):
                    import re
                    pattern = f"^( {'|'.join([re.escape(g) for g in gender_val])} )$"
                    pattern = pattern.replace(" ", "")
                    mongo_query["gender"] = {"$regex": pattern, "$options": "i"}
                else:
                    mongo_query["gender"] = {"$regex": f"^{gender_val}$", "$options": "i"}
            
            if "exclude_categories" in mandatory_filters:
                # Store for direct name check during fallback if needed
                mongo_query["category"] = {"$nin": mandatory_filters["exclude_categories"]}
        
        # HARD FILTER: Only show fashion-related products
        # Exclude electronics and tech categories that might leak in
        if "category" not in mongo_query:
            mongo_query["category"] = {"$nin": ["electronics", "laptop", "gadget", "mobile", "tech", "headphones"]}
        else:
            # Combine mandatory exclusions with fashion guardrail
            tech_excl = ["electronics", "laptop", "gadget", "mobile", "tech", "headphones"]
            mongo_query["category"]["$nin"].extend([t for t in tech_excl if t not in mongo_query["category"]["$nin"]])

        if filters.get("category"):
            # Ensure it doesn't conflict with mandatory exclusions
            target_cat = filters["category"]
            if "category" in mongo_query and "$nin" in mongo_query["category"]:
                if target_cat not in mongo_query["category"]["$nin"]:
                    mongo_query["category"]["$eq"] = target_cat
            else:
                mongo_query["category"] = {"$eq": target_cat}
                
        if filters.get("color"):
            mongo_query["color"] = {"$regex": filters["color"], "$options": "i"}
        
        # 2.5 Apply Exclusions & Meta Filters
        if exclude_ids:
            from bson import ObjectId
            valid_ids = []
            for eid in exclude_ids:
                try:
                    valid_ids.append(ObjectId(eid))
                except:
                    continue
            if valid_ids:
                mongo_query["_id"] = {"$nin": valid_ids}
        
        # Apply attribute exclusions from preferences
        if preferences:
            import re
            disliked_colors = preferences.get("disliked_colors", [])
            if disliked_colors:
                # Use $regex with $not for case-insensitive exclusion of multiple colors
                regex_pattern = "|".join([re.escape(c) for c in disliked_colors])
                if "color" not in mongo_query:
                    mongo_query["color"] = {"$not": re.compile(regex_pattern, re.I)}
                else:
                    # If already has color filter, we might need to combine with $and
                    pass
            
            disliked_styles = preferences.get("disliked_styles", [])
            if disliked_styles:
                style_pattern = "|".join([re.escape(s) for s in disliked_styles])
                mongo_query["style"] = {"$not": re.compile(style_pattern, re.I)}

        # Fuzzy Price Range
        if filters.get("price_range"):
            price = filters["price_range"]
            if price.get("min") is not None or price.get("max") is not None:
                mongo_query["price"] = {}
                if price.get("min"): mongo_query["price"]["$gte"] = price["min"]
                if price.get("max"): mongo_query["price"]["$lte"] = price["max"]

        # 3. Execute Primary Search
        results = await self._execute_search(mongo_query, limit)
        
        # 4. RELAXATION LOGIC (Must preserve mandatory_filters)
        if len(results) < 2:
            # Relax price but KEEP gender/exclusions
            if "price" in mongo_query:
                del mongo_query["price"]
                results = await self._execute_search(mongo_query, limit)
            
            # Relax color/category but KEEP gender/exclusions
            if len(results) < 1:
                # Create a bare-minimum query with only mandatory filters
                fallback_query = {}
                if mandatory_filters:
                    if "gender" in mongo_query: fallback_query["gender"] = mongo_query["gender"]
                    if "category" in mongo_query: fallback_query["category"] = mongo_query["category"]
                else:
                    fallback_query["category"] = {"$nin": ["electronics", "laptop", "gadget", "mobile", "tech", "headphones"]}
                
                results = await self._execute_search(fallback_query, limit)

        # 5. Fallback: Just show newest items (Must preserve mandatory_filters)
        if not results:
            fallback_query = {}
            if mandatory_filters:
                if "gender" in mongo_query: fallback_query["gender"] = mongo_query["gender"]
                if "category" in mongo_query: fallback_query["category"] = mongo_query["category"]
            else:
                fallback_query["category"] = {"$nin": ["electronics", "laptop", "gadget", "mobile", "tech", "headphones"]}
            
            results = await self._execute_search(fallback_query, limit, sort_by_new=True)

        # 6. SAFETY CHECK: Filter out gender violations as last resort
        if mandatory_filters and "gender" in mandatory_filters:
            target_gender = mandatory_filters["gender"].lower()
            safe_results = []
            
            for product in results:
                product_gender = str(product.get("gender", "")).lower()
                product_name = str(product.get("name", "")).lower()
                
                # Check for obvious gender violations
                is_violation = False
                
                # If target is male, reject ONLY clearly female items
                if target_gender in ["male", "men", "mens"]:
                    # Only reject if explicitly marked as female AND is clearly female clothing
                    if ("female" in product_gender or "women" in product_gender) and \
                       any(word in product_name for word in ["salwar", "saree", "lehenga", "kurti", "dupatta", "blouse"]):
                        is_violation = True
                
                # If target is female, reject ONLY clearly male items  
                elif target_gender in ["female", "women", "womens"]:
                    # Only reject if explicitly marked as male
                    if "male" in product_gender or "men" in product_gender:
                        is_violation = True
                
                if not is_violation:
                    safe_results.append(product)
            
            if len(safe_results) < len(results):
                logger.warning(f"[SearchOrchestrator] Filtered out {len(results) - len(safe_results)} gender violations")
            
            results = safe_results

        return results

    async def _execute_search(self, query: dict, limit: int, sort_by_new: bool = False) -> List[dict]:
        db = get_database()
        cursor = db.products.find(query).limit(limit)
        if sort_by_new:
            cursor = cursor.sort([("_id", -1)])
        
        items = await cursor.to_list(length=limit)
        
        # DIVERSITY SHUFFLE (Phase 2):
        # Shuffling prevents results from being purely based on DB insertion order
        import random
        random.shuffle(items)
        
        for item in items:
            item["id"] = str(item["_id"])
        return items

search_orchestrator = SearchOrchestrator()
