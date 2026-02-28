"""
Enhanced Product Search Agent - Database retrieval agent with mandatory gender filtering.

Responsibilities:
- Receive structured queries from orchestrator
- Apply mandatory gender filters that NEVER fall back to opposite gender
- Translate queries into database or vector search queries
- Return ONLY products found in the database that match ALL constraints
- Never rank or recommend
- Log filter application for debugging and verification
"""

from typing import List, Dict, Any, Optional
from app.db.mongodb import get_database
from app.services.search.orchestrator import search_orchestrator
from app.services.search.gender_filter import gender_filter, Gender, GenderInferenceConfidence
from app.services.search.personalized_ranking import personalized_ranking
from app.core.categories import CATEGORY_MAP, CATEGORY_KEYWORDS, normalize_category, is_name_priority_term
import logging

logger = logging.getLogger(__name__)


class SearchAgent:
    """
    Enhanced database retrieval agent with mandatory gender filtering.
    Returns ONLY products found in the database that match ALL constraints.
    Never ranks or recommends.
    NEVER falls back to opposite gender products.
    """
    
    async def search_products(
        self,
        query: str,
        filters: Dict[str, Any] = None,
        category: Optional[str] = None,
        exclude_ids: Optional[List[str]] = None,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Search products in database with mandatory gender filtering.
        
        Args:
            query: Natural language query
            filters: Structured filters (price, brand, color, gender, etc.)
            category: Product category
            exclude_ids: Product IDs to exclude
            limit: Maximum number of products to return
        
        Returns:
        {
            "products": [...],
            "search_authority": {
                "search_status": "FOUND" | "EMPTY" | "ERROR",
                "products_found": int,
                "confidence": float,
                "gender_filter_applied": bool,
                "gender_constraint": str,
                "llm_guardrail": str
            }
        }
        """
        try:
            logger.info(f"[SearchAgent] Processing query: '{query}' with filters: {filters}")
            
            # Extract gender preference from filters
            gender_preference = None
            if filters:
                gender_preference = filters.get("gender")
                # Also check for gender in nested preferences
                if not gender_preference and "preferences" in filters:
                    gender_preference = filters["preferences"].get("gender")
            
            # Check if user is searching for a specific product type (like jersey)
            name_term = is_name_priority_term(query) if query else None
            
            if name_term:
                logger.info(f"[SearchAgent] Name priority search for: {name_term}")
                return await self.search_by_name_priority(
                    name_term, filters, exclude_ids, limit, gender_preference, query
                )
            
            # Normalize category from query
            if not category and query:
                category = normalize_category(query)
            
            # Use direct database query for structured filters or specific categories
            if filters and (filters.get("price") or category or gender_preference or filters.get("exclude_categories")):
                filter_dict = {}
                if category:
                    filter_dict["category"] = category
                if filters.get("price"):
                    filter_dict["price"] = filters["price"]
                if filters.get("brand"):
                    filter_dict["brand"] = filters["brand"]
                if filters.get("color"):
                    filter_dict["color"] = filters["color"]
                if gender_preference:
                    filter_dict["gender"] = gender_preference
                if filters.get("exclude_categories"):
                    filter_dict["exclude_categories"] = filters["exclude_categories"]
                
                return await self.search_by_filters(
                    filter_dict, exclude_ids=exclude_ids, limit=limit, query_text=query
                )
            
            # Use hybrid search with mandatory gender filtering
            return await self.hybrid_search_with_gender_filter(
                query=query,
                filters=filters,
                category=category,
                exclude_ids=exclude_ids,
                limit=limit,
                gender_preference=gender_preference
            )
            
        except Exception as e:
            logger.error(f"[SearchAgent] Search error: {e}")
            return {
                "products": [],
                "search_authority": {
                    "search_status": "ERROR",
                    "products_found": 0,
                    "confidence": 0.0,
                    "gender_filter_applied": False,
                    "gender_constraint": None,
                    "llm_guardrail": "ALLOW_FALLBACK"
                }
            }
    
    async def search_with_personalized_ranking(
        self,
        query: str,
        user_profile,  # UserProfile from context
        filters: Dict[str, Any] = None,
        category: Optional[str] = None,
        exclude_ids: Optional[List[str]] = None,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Search products with personalized ranking based on user profile.
        
        This method combines search results with personalized ranking to provide
        results that match both query intent and user preferences.
        """
        try:
            # First, get search results using existing search logic
            search_results = await self.search_products(
                query=query,
                filters=filters,
                category=category,
                exclude_ids=exclude_ids,
                limit=limit * 2  # Get more results for better ranking
            )
            
            products = search_results.get("products", [])
            if not products:
                return search_results
            
            # Extract query intent for ranking
            query_intent = {
                "product_category": category,
                "filters": filters or {}
            }
            
            # Apply personalized ranking
            ranked_products = personalized_ranking.rank_products(
                products=products,
                user_profile=user_profile,
                query_intent=query_intent,
                search_context={"search_terms": query.split() if query else []}
            )
            
            # Limit to requested number of results
            final_products = ranked_products[:limit]
            
            # Update search authority with ranking information
            search_authority = search_results.get("search_authority", {})
            search_authority.update({
                "personalized_ranking_applied": True,
                "ranking_explanation": personalized_ranking.explain_ranking_decision(final_products, user_profile),
                "total_products_ranked": len(ranked_products)
            })
            
            logger.info(f"[SearchAgent] Applied personalized ranking to {len(products)} products, "
                       f"returning top {len(final_products)}")
            
            return {
                "products": final_products,
                "search_authority": search_authority
            }
            
        except Exception as e:
            logger.error(f"[SearchAgent] Personalized ranking error: {e}")
            # Fall back to regular search if ranking fails
            return await self.search_products(query, filters, category, exclude_ids, limit)
    
    def get_search_result_explanation(
        self,
        search_results: Dict[str, Any],
        user_profile = None
    ) -> str:
        """
        Generate explanation for search results.
        
        Args:
            search_results: Search results from search_products
            user_profile: User profile for personalized explanations
            
        Returns:
            Human-readable explanation of search results
        """
        search_authority = search_results.get("search_authority", {})
        products_found = search_authority.get("products_found", 0)
        
        if products_found == 0:
            # Explain why no results
            gender_constraint = search_authority.get("gender_constraint")
            if gender_constraint:
                return f"No {gender_constraint} products found matching your search. Try a different category or style."
            else:
                return "No products found matching your search criteria. Try different keywords or filters."
        
        # Explain successful results
        explanation_parts = []
        
        # Gender filtering
        if search_authority.get("gender_filter_applied"):
            gender_explanation = search_authority.get("gender_explanation", "")
            if gender_explanation:
                explanation_parts.append(gender_explanation)
        
        # Personalized ranking
        if search_authority.get("personalized_ranking_applied"):
            ranking_explanation = search_authority.get("ranking_explanation", "")
            if ranking_explanation:
                explanation_parts.append(ranking_explanation)
        
        # Default explanation
        if not explanation_parts:
            explanation_parts.append(f"Found {products_found} products matching your search")
        
        return ". ".join(explanation_parts)
    
    async def hybrid_search_with_gender_filter(
        self,
        query: str,
        filters: Dict[str, Any] = None,
        category: Optional[str] = None,
        exclude_ids: Optional[List[str]] = None,
        limit: int = 20,
        gender_preference: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform hybrid search with mandatory gender filtering.
        
        This method ensures gender constraints are ALWAYS applied and never bypassed.
        """
        try:
            # Build search query
            search_query = query
            
            # Enhance query with filters if provided
            if filters:
                if filters.get("brand"):
                    search_query += f" {filters['brand']}"
                if filters.get("color"):
                    search_query += f" {filters['color']}"
                if filters.get("style"):
                    search_query += f" {filters['style']}"
            
            if category:
                search_query = f"{category} {search_query}"
            
            # Prepare mandatory filters for search orchestrator
            mandatory_filters = {}
            
            # CRITICAL: Apply gender filter if specified or inferred
            if gender_preference:
                mandatory_filters["gender"] = gender_preference
                logger.info(f"[SearchAgent] Applying mandatory gender filter: {gender_preference}")
            else:
                # Try to infer gender from query
                inferred_gender, confidence = gender_filter.infer_gender_from_query(query)
                if inferred_gender and confidence in [GenderInferenceConfidence.HIGH, GenderInferenceConfidence.MEDIUM]:
                    mandatory_filters["gender"] = inferred_gender.value
                    logger.info(f"[SearchAgent] Inferred and applying gender filter: {inferred_gender.value} ({confidence.value})")
            
            # Add other mandatory filters
            if filters and filters.get("exclude_categories"):
                mandatory_filters["exclude_categories"] = filters["exclude_categories"]
            
            # Use search orchestrator with mandatory filters
            preferences = filters.get("preferences") or {} if filters else {}
            search_results = await search_orchestrator.hybrid_search(
                query=search_query,
                limit=limit,
                exclude_ids=exclude_ids,
                preferences=preferences,
                mandatory_filters=mandatory_filters
            )
            
            # Validate gender compliance of results
            gender_constraint = mandatory_filters.get("gender")
            if gender_constraint:
                compliant_products, non_compliant = gender_filter.validate_gender_compliance(
                    search_results, gender_constraint
                )
                
                if non_compliant:
                    logger.error(f"[SearchAgent] Found {len(non_compliant)} non-compliant products, filtering them out")
                    search_results = compliant_products
            
            # Format products
            formatted_products = []
            for product in search_results:
                formatted_product = {
                    "id": str(product.get("_id", "")),
                    "name": product.get("name", "Unknown Product"),
                    "category": product.get("category", "unknown"),
                    "price": float(product.get("price", 0)),
                    "attributes": {
                        "color": product.get("color"),
                        "brand": product.get("brand"),
                        "description": product.get("description", ""),
                        "images": product.get("images", []),
                        "fit": product.get("fit"),
                        "formality": product.get("formality"),
                        "style": product.get("style"),
                        "gender": product.get("gender")  # Include gender for debugging
                    },
                    "availability": product.get("isActive", True) and product.get("isApproved", True)
                }
                formatted_products.append(formatted_product)
            
            # Log if no results due to gender constraint
            if not formatted_products and gender_constraint:
                gender_filter.log_no_results_due_to_gender(
                    gender_constraint=gender_constraint,
                    original_query=query,
                    total_products_without_filter=0  # Could query without filter to get count
                )
            
            # Calculate confidence
            confidence = min(0.95, 0.4 + 0.1 * len(formatted_products)) if formatted_products else 0.0
            
            # Build search authority with gender information
            search_authority = {
                "search_status": "FOUND" if formatted_products else "EMPTY",
                "products_found": len(formatted_products),
                "confidence": confidence,
                "gender_filter_applied": bool(gender_constraint),
                "gender_constraint": gender_constraint,
                "gender_explanation": gender_filter.explain_gender_constraint(gender_preference, query),
                "llm_guardrail": "DO_NOT_SAY_NOT_FOUND" if formatted_products else "ALLOW_FALLBACK"
            }
            
            return {
                "products": formatted_products,
                "search_authority": search_authority
            }
            
        except Exception as e:
            logger.error(f"[SearchAgent] Hybrid search with gender filter error: {e}")
            return {
                "products": [],
                "search_authority": {
                    "search_status": "ERROR",
                    "products_found": 0,
                    "confidence": 0.0,
                    "gender_filter_applied": False,
                    "gender_constraint": None,
                    "llm_guardrail": "ALLOW_FALLBACK"
                }
            }
    
    async def search_by_name_priority(
        self,
        name_term: str,
        filters: Dict[str, Any] = None,
        exclude_ids: Optional[List[str]] = None,
        limit: int = 20,
        gender_preference: Optional[str] = None,
        query_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search products by name first with mandatory gender filtering.
        Used for specific product types like 'jersey' that exist within broader categories.
        """
        try:
            db = get_database()
            
            # Build base query: search by name containing the term
            mongo_query = {
                "name": {"$regex": name_term, "$options": "i"},
                "isApproved": True,
                "isActive": True,
                "isVisible": {"$ne": False}
            }
            
            # Apply mandatory gender filter
            gender_constraint = None
            if gender_preference:
                mongo_query, gender_applied = gender_filter.apply_mandatory_gender_filter(
                    mongo_query, gender_preference, query_text
                )
                if gender_applied:
                    gender_constraint = gender_preference
            else:
                # Try to infer gender from name term or query
                inferred_gender, confidence = gender_filter.infer_gender_from_query(name_term)
                if not inferred_gender and query_text:
                    inferred_gender, confidence = gender_filter.infer_gender_from_query(query_text)
                
                if inferred_gender and confidence in [GenderInferenceConfidence.HIGH, GenderInferenceConfidence.MEDIUM]:
                    mongo_query, gender_applied = gender_filter.apply_mandatory_gender_filter(
                        mongo_query, inferred_gender.value, query_text
                    )
                    if gender_applied:
                        gender_constraint = inferred_gender.value
            
            # Handle exclusion IDs
            if exclude_ids:
                from bson import ObjectId
                valid_ids = []
                for id_str in exclude_ids:
                    try:
                        valid_ids.append(ObjectId(id_str))
                    except:
                        continue
                if valid_ids:
                    mongo_query["_id"] = {"$nin": valid_ids}
            
            # Add additional filters if provided
            if filters:
                if filters.get("size"):
                    mongo_query["$or"] = [
                        {"sizes": {"$regex": filters["size"], "$options": "i"}},
                        {"size": {"$regex": filters["size"], "$options": "i"}}
                    ]
                if filters.get("color"):
                    mongo_query["color"] = {"$regex": filters["color"], "$options": "i"}
                
                if filters.get("gender"):
                    gender_val = filters["gender"]
                    if isinstance(gender_val, list):
                        import re
                        pattern = f"^( {'|'.join([re.escape(g) for g in gender_val])} )$"
                        pattern = pattern.replace(" ", "")
                        mongo_query["gender"] = {"$regex": pattern, "$options": "i"}
                    else:
                        mongo_query["gender"] = {"$regex": f"^{gender_val}$", "$options": "i"}
                
                if filters.get("exclude_categories"):
                    excludes = filters["exclude_categories"]
                    if isinstance(excludes, list):
                        all_exclude_cats = []
                        for ex in excludes:
                            ex_lower = ex.lower()
                            if ex_lower in CATEGORY_MAP:
                                all_exclude_cats.extend(CATEGORY_MAP[ex_lower])
                            else:
                                all_exclude_cats.append(ex)
                        
                        if all_exclude_cats:
                            mongo_query["category"] = {"$nin": all_exclude_cats}
                            # Also exclude if category name appears in the product name
                            name_exclude_pattern = f"^(?!.*({'|'.join([re.escape(c) for c in all_exclude_cats])})).*"
                            # Combine with existing name search logic
                            original_name_query = mongo_query["name"]["$regex"]
                            # For simple regex, we can use a lookahead to exclude terms
                            mongo_query["name"] = {
                                "$regex": f"^(?=.*{original_name_query})(?!.*({'|'.join([re.escape(c) for c in all_exclude_cats])})).*",
                                "$options": "i"
                            }
                if filters.get("price"):
                    price = filters["price"]
                    if price.get("min") is not None:
                        mongo_query["price"] = {"$gte": price["min"]}
                    if price.get("max") is not None:
                        mongo_query.setdefault("price", {})["$lte"] = price["max"]
            
            logger.info(f"[SearchAgent] Name priority query: {mongo_query}")
            
            # Execute search
            cursor = db.products.find(mongo_query).limit(limit)
            products = await cursor.to_list(length=limit)
            
            # If no name matches found, fall back to category search
            if not products:
                logger.info(f"[SearchAgent] No name matches for '{name_term}', falling back to category search")
                category = normalize_category(name_term)
                if category:
                    return await self.search_by_filters({"category": category, **(filters or {})}, limit)
            
            # Format products
            formatted_products = []
            for product in products:
                formatted_products.append({
                    "id": str(product.get("_id", "")),
                    "_id": str(product.get("_id", "")),
                    "name": product.get("name", "Unknown"),
                    "brand": product.get("brand", "No Brand"),
                    "category": product.get("category", ""),
                    "price": float(product.get("price", 0)),
                    "images": product.get("images", []),
                    "color": product.get("color", ""),
                    "sizes": product.get("sizes", []),
                    "isApproved": product.get("isApproved", False),
                    "isActive": product.get("isActive", False)
                })
            
            return {
                "products": formatted_products,
                "search_authority": {
                    "search_status": "SUCCESS" if formatted_products else "NO_RESULTS",
                    "products_found": len(formatted_products),
                    "confidence": 0.95 if formatted_products else 0.3,
                    "search_type": "name_priority",
                    "gender_filter_applied": bool(gender_constraint),
                    "gender_constraint": gender_constraint,
                    "gender_explanation": gender_filter.explain_gender_constraint(gender_preference, query_text),
                    "llm_guardrail": "DO_NOT_SAY_NOT_FOUND" if formatted_products else "ALLOW_FALLBACK"
                }
            }
            
        except Exception as e:
            logger.error(f"[SearchAgent] Name priority search error: {e}")
            return {
                "products": [],
                "search_authority": {
                    "search_status": "ERROR",
                    "products_found": 0,
                    "confidence": 0.0,
                    "gender_filter_applied": False,
                    "gender_constraint": None,
                    "llm_guardrail": "ALLOW_FALLBACK"
                }
            }
    
    async def search_by_filters(
        self,
        filters: Dict[str, Any],
        exclude_ids: Optional[List[str]] = None,
        limit: int = 20,
        query_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search products using structured filters with mandatory gender filtering.
        """
        try:
            db = get_database()
            mongo_query = {}
            
            # Apply mandatory gender filter FIRST
            gender_constraint = None
            gender_preference = filters.get("gender")
            
            if gender_preference:
                mongo_query, gender_applied = gender_filter.apply_mandatory_gender_filter(
                    mongo_query, gender_preference, query_text
                )
                if gender_applied:
                    gender_constraint = gender_preference
                    logger.info(f"[SearchAgent] Applied mandatory gender filter: {gender_preference}")
            else:
                # Try to infer gender from query text
                if query_text:
                    inferred_gender, confidence = gender_filter.infer_gender_from_query(query_text)
                    if inferred_gender and confidence in [GenderInferenceConfidence.HIGH, GenderInferenceConfidence.MEDIUM]:
                        mongo_query, gender_applied = gender_filter.apply_mandatory_gender_filter(
                            mongo_query, inferred_gender.value, query_text
                        )
                        if gender_applied:
                            gender_constraint = inferred_gender.value
                            logger.info(f"[SearchAgent] Applied inferred gender filter: {inferred_gender.value}")
            
            # Build other filters using centralized CATEGORY_MAP
            if filters.get("category"):
                category = filters["category"].lower()
                
                # Use centralized category mapping
                if category in CATEGORY_MAP:
                    mongo_query["category"] = {"$in": CATEGORY_MAP[category]}
                else:
                    # Fallback to regex for unmapped categories
                    mongo_query["category"] = {"$regex": category, "$options": "i"}
            
            if filters.get("price"):
                price = filters["price"]
                if price.get("min") is not None or price.get("max") is not None:
                    mongo_query["price"] = {}
                    if price.get("min") is not None:
                        mongo_query["price"]["$gte"] = price["min"]
                    if price.get("max") is not None:
                        mongo_query["price"]["$lte"] = price["max"]
            
            if filters.get("brand"):
                mongo_query["brand"] = {"$regex": filters["brand"], "$options": "i"}
            
            if filters.get("color"):
                mongo_query["color"] = {"$regex": filters["color"], "$options": "i"}
            
            if filters.get("gender"):
                gender_val = filters["gender"]
                if isinstance(gender_val, list):
                    # Combine into a case-insensitive regex OR pattern
                    import re
                    pattern = f"^( {'|'.join([re.escape(g) for g in gender_val])} )$"
                    pattern = pattern.replace(" ", "") # Remove spaces if any
                    mongo_query["gender"] = {"$regex": pattern, "$options": "i"}
                else:
                    mongo_query["gender"] = {"$regex": f"^{gender_val}$", "$options": "i"}
            
            if filters.get("exclude_categories"):
                excludes = filters["exclude_categories"]
                if isinstance(excludes, list):
                    # Combine all excluded categories and their mapped sub-categories
                    all_exclude_cats = []
                    for ex in excludes:
                        ex_lower = ex.lower()
                        if ex_lower in CATEGORY_MAP:
                            all_exclude_cats.extend(CATEGORY_MAP[ex_lower])
                        else:
                            all_exclude_cats.append(ex)
                    
                    if "category" in mongo_query and "$in" in mongo_query["category"]:
                        # If we already have a category filter, we must ensure 
                        # the intersection doesn't include excluded items
                        # (Though usually we search for ONE category or none)
                        if "$nin" not in mongo_query["category"]:
                            mongo_query["category"]["$nin"] = all_exclude_cats
                    else:
                        mongo_query["category"] = {"$nin": all_exclude_cats}

            if filters.get("fit"):
                fit = filters["fit"].lower()
                # Map fit variations
                fit_map = {
                    "regular": ["regular", "normal", "casual", "standard"],
                    "oversized": ["oversized", "oversize", "baggy", "loose"],
                    "slim": ["slim", "tight", "fitted", "skinny"],
                    "loose": ["loose", "baggy", "relaxed"]
                }
                
                if fit in fit_map:
                    mongo_query["fit"] = {"$in": fit_map[fit]}
                else:
                    mongo_query["fit"] = {"$regex": fit, "$options": "i"}
            
            # Only active, approved products
            mongo_query["isActive"] = True
            mongo_query["isApproved"] = True
            mongo_query["isVisible"] = {"$ne": False}  # Handle products where flag is missing
            
            # Handle exclusion IDs
            if exclude_ids:
                from bson import ObjectId
                valid_ids = []
                for id_str in exclude_ids:
                    try:
                        valid_ids.append(ObjectId(id_str))
                    except:
                        continue
                if valid_ids:
                    # If _id filter already exists, combine it
                    if "_id" in mongo_query:
                        mongo_query["_id"]["$nin"] = valid_ids
                    else:
                        mongo_query["_id"] = {"$nin": valid_ids}
            
            # Execute search
            cursor = db.products.find(mongo_query).limit(limit)
            raw_products = await cursor.to_list(length=limit)
            
            # Format products
            formatted_products = []
            for product in raw_products:
                formatted_product = {
                    "id": str(product.get("_id", "")),
                    "name": product.get("name", "Unknown Product"),
                    "category": product.get("category", "unknown"),
                    "price": float(product.get("price", 0)),
                    "attributes": {
                        "color": product.get("color"),
                        "brand": product.get("brand"),
                        "description": product.get("description", ""),
                        "images": product.get("images", []),
                        "fit": product.get("fit"),
                        "formality": product.get("formality"),
                        "style": product.get("style")
                    },
                    "availability": product.get("isActive", True) and product.get("isApproved", True)
                }
                formatted_products.append(formatted_product)
            
            # Calculate dynamic confidence based on result count
            confidence = min(0.95, 0.4 + 0.1 * len(formatted_products)) if formatted_products else 0.0
            
            # Log if no results due to gender constraint
            if not formatted_products and gender_constraint:
                gender_filter.log_no_results_due_to_gender(
                    gender_constraint=gender_constraint,
                    original_query=query_text or "filter-based search",
                    total_products_without_filter=0
                )
            
            return {
                "products": formatted_products,
                "search_authority": {
                    "search_status": "FOUND" if formatted_products else "EMPTY",
                    "products_found": len(formatted_products),
                    "confidence": confidence,
                    "gender_filter_applied": bool(gender_constraint),
                    "gender_constraint": gender_constraint,
                    "gender_explanation": gender_filter.explain_gender_constraint(gender_preference, query_text),
                    "llm_guardrail": "DO_NOT_SAY_NOT_FOUND" if formatted_products else "ALLOW_FALLBACK"
                }
            }
            
        except Exception as e:
            logger.error(f"[SearchAgent] Filter-based search error: {e}")
            return {
                "products": [],
                "search_authority": {
                    "search_status": "ERROR",
                    "products_found": 0,
                    "confidence": 0.0,
                    "gender_filter_applied": False,
                    "gender_constraint": None,
                    "llm_guardrail": "ALLOW_FALLBACK"
                }
            }


# Singleton instance
search_agent = SearchAgent()
