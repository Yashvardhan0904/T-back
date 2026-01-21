from app.db.mongodb import get_database
from app.services.search.strategy import search_strategy
from app.services.embeddings.service import embedding_service
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class SearchOrchestrator:
    async def hybrid_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        1. Handle special cases (Trending)
        2. Extract attributes using Groq
        3. Try MongoDB filtered search
        4. Perform relaxation if needed
        """
        # 1. SPECIAL CASE: Trending
        if query.lower() in ["trending", "popular", "new arrivals"]:
            return await self._execute_search({}, limit, sort_by_new=True)

        filters = await search_strategy.extract_filters(query)
        
        # 2. Construct MongoDB Filter Query
        mongo_query = {}
        
        # HARD FILTER: Only show fashion-related products
        # Exclude electronics and tech categories that might leak in
        mongo_query["category"] = {"$nin": ["electronics", "laptop", "gadget", "mobile", "tech", "headphones"]}

        if filters.get("category"):
            mongo_query["category"]["$eq"] = filters["category"]
        if filters.get("color"):
            mongo_query["color"] = {"$regex": filters["color"], "$options": "i"}
        
        # Fuzzy Price Range
        if filters.get("price_range"):
            price = filters["price_range"]
            if price.get("min") is not None or price.get("max") is not None:
                mongo_query["price"] = {}
                if price.get("min"): mongo_query["price"]["$gte"] = price["min"]
                if price.get("max"): mongo_query["price"]["$lte"] = price["max"]

        # 3. Execute Primary Search
        results = await self._execute_search(mongo_query, limit)
        
        # 4. RELAXATION LOGIC
        if len(results) < 2:
            # Relax price
            if "price" in mongo_query:
                del mongo_query["price"]
                results = await self._execute_search(mongo_query, limit)
            
            # Relax color/category
            if len(results) < 1 and filters.get("category"):
                results = await self._execute_search({"category": filters["category"]}, limit)

        # 5. Fallback: Just show newest items
        if not results:
            results = await self._execute_search({}, limit, sort_by_new=True)

        return results

    async def _execute_search(self, query: dict, limit: int, sort_by_new: bool = False) -> List[dict]:
        db = get_database()
        cursor = db.products.find(query).limit(limit)
        if sort_by_new:
            cursor = cursor.sort([("_id", -1)])
        
        items = await cursor.to_list(length=limit)
        for item in items:
            item["_id"] = str(item["_id"])
        return items

search_orchestrator = SearchOrchestrator()
