"""
Central Category Mapping - THE SINGLE SOURCE OF TRUTH

This file defines all product category mappings for the AI search system.
When a user searches for something, we map their term to database categories.

HOW TO ADD NEW CATEGORIES:
1. Add entry to CATEGORY_MAP: "user_term": ["db_cat1", "db_cat2", ...]
2. Add entry to CATEGORY_KEYWORDS for natural language detection
3. For specific product types (like jersey, polo), add to NAME_PRIORITY_TERMS
4. Done! All agents automatically pick up changes.
"""

from typing import List, Optional, Set

# ============================================================
# NAME PRIORITY TERMS - Search by product NAME first, not category
# These are specific product types within a broader category
# Example: "jersey" is a type of shirt, so search name for "jersey"
# ============================================================
NAME_PRIORITY_TERMS: Set[str] = {
    "jersey", "jerseys", "jersy", "jersys", "jersyes",
    "polo", "polos",
    "kurta", "kurtas", "kurti", "kurtis",
    "saree", "sari", "sarees", "saris",
    "jhumka", "jhumkas",
    "sherwani",
    "blazer", "blazers",
    "cardigan", "cardigans",
    "tank", "tanktop",
    "joggers", "jogger",
    "cargo", "cargos",
    "tracksuit",
}

# ============================================================
# MASTER CATEGORY MAP
# Maps user search terms → database category values
# ============================================================
CATEGORY_MAP = {
    # ---- TOPS ----
    "shirt": ["shirt", "shirts", "tshirt", "t-shirt", "tee", "tees", "tops", "top", "polo", "tank", "sleeveless"],
    "tshirt": ["shirt", "shirts", "tshirt", "t-shirt", "tee", "tees", "tops"],
    "t-shirt": ["shirt", "shirts", "tshirt", "t-shirt", "tee", "tees", "tops"],
    "tee": ["shirt", "shirts", "tshirt", "t-shirt", "tee", "tees", "tops"],
    "top": ["shirt", "shirts", "tshirt", "t-shirt", "tee", "tees", "tops", "top", "hoodie", "jacket", "kurta", "sweater"],
    "tops": ["shirt", "shirts", "tshirt", "t-shirt", "tee", "tees", "tops", "top", "hoodie", "jacket", "kurta", "sweater"],
    "upper wear": ["shirt", "shirts", "tshirt", "t-shirt", "tee", "tees", "tops", "top", "hoodie", "jacket", "kurta", "sweater", "outerwear"],
    "top wear": ["shirt", "shirts", "tshirt", "t-shirt", "tee", "tees", "tops", "top", "hoodie", "jacket", "kurta", "sweater"],
    "upper": ["shirt", "shirts", "tshirt", "t-shirt", "tee", "tees", "tops", "top", "hoodie", "jacket", "kurta", "sweater"],
    "jersey": ["shirt", "shirts", "tshirt", "t-shirt", "tee", "jersey", "tops"],
    "jerseys": ["shirt", "shirts", "tshirt", "t-shirt", "tee", "jersey", "tops"],
    "jersy": ["shirt", "shirts", "tshirt", "t-shirt", "tee", "jersey", "tops"],
    "jersys": ["shirt", "shirts", "tshirt", "t-shirt", "tee", "jersey", "tops"],
    "polo": ["shirt", "shirts", "polo", "tops"],
    "polos": ["shirt", "shirts", "polo", "tops"],
    "tank": ["shirt", "shirts", "tank", "tanktop", "tops", "sleeveless"],
    "tanktop": ["shirt", "shirts", "tank", "tanktop", "tops"],
    "vest": ["shirt", "shirts", "tank", "tanktop", "tops"],
    
    # ---- OUTERWEAR ----
    "hoodie": ["hoodie", "hoodies", "sweatshirt", "sweater", "jacket"],
    "sweatshirt": ["hoodie", "hoodies", "sweatshirt", "sweater"],
    "sweater": ["sweater", "sweaters", "hoodie", "sweatshirt", "pullover", "cardigan"],
    "pullover": ["sweater", "sweaters", "hoodie", "sweatshirt"],
    "jacket": ["jacket", "jackets", "coat", "blazer", "puffer", "bomber", "windbreaker"],
    "coat": ["jacket", "jackets", "coat", "coats", "overcoat"],
    "blazer": ["jacket", "jackets", "blazer", "blazers", "suit"],
    "cardigan": ["cardigan", "cardigans", "sweater"],
    "shacket": ["shirt", "jacket", "shacket"],
    
    # ---- BOTTOMS ----
    "jeans": ["jeans", "pants", "trousers", "denim", "denims", "bottom", "bottomwear"],
    "denim": ["jeans", "pants", "trousers", "denim", "denims", "bottom"],
    "pants": ["jeans", "pants", "trousers", "bottom", "bottomwear", "chinos", "joggers", "cargos"],
    "trousers": ["jeans", "pants", "trousers", "bottom", "bottomwear", "formal pants"],
    "bottom": ["jeans", "pants", "trousers", "bottom", "bottomwear", "shorts", "joggers", "cargos"],
    "bottomwear": ["jeans", "pants", "trousers", "bottom", "bottomwear", "shorts", "joggers", "cargos"],
    "shorts": ["shorts", "bottom", "bottomwear", "half pants"],
    "joggers": ["joggers", "pants", "trousers", "bottom", "trackpants", "sweatpants"],
    "trackpants": ["joggers", "pants", "trackpants", "bottom", "sweatpants"],
    "sweatpants": ["joggers", "pants", "trackpants", "bottom", "sweatpants"],
    "cargos": ["cargos", "cargo", "pants", "trousers", "bottom"],
    "cargo": ["cargos", "cargo", "pants", "trousers", "bottom"],
    "chinos": ["pants", "trousers", "chinos", "bottom"],
    
    # ---- FOOTWEAR ----
    "sneakers": ["sneakers", "shoes", "footwear", "shoe", "trainers"],
    "shoes": ["sneakers", "shoes", "footwear", "shoe", "boots", "sandals", "formal shoes", "loafers"],
    "footwear": ["sneakers", "shoes", "footwear", "shoe", "boots", "sandals", "slippers", "heels"],
    "boots": ["boots", "shoes", "footwear"],
    "sandals": ["sandals", "shoes", "footwear", "slides", "flip flops", "juttis", "mojaris", "kolhapuris", "wedges", "heels"],
    "slippers": ["slippers", "shoes", "footwear", "slides", "flip flops"],
    "loafers": ["shoes", "loafers", "formal shoes"],
    
    # ---- ACCESSORIES ----
    "socks": ["socks", "sock"],
    "cap": ["cap", "caps", "hat", "hats", "accessories", "beanie"],
    "hat": ["cap", "caps", "hat", "hats", "accessories"],
    "beanie": ["cap", "beanie", "hat", "accessories"],
    "belt": ["belt", "belts", "accessories"],
    "watch": ["watch", "watches", "accessories"],
    "bag": ["bag", "bags", "backpack", "accessories", "handbag", "tote"],
    "backpack": ["bag", "bags", "backpack", "accessories"],
    "wallet": ["wallet", "wallets", "accessories"],
    "sunglasses": ["sunglasses", "shades", "accessories"],
    "earrings": ["earrings", "accessories", "jewelry", "jhumkas", "jewellery", "studs"],
    "jewelry": ["earrings", "necklace", "accessories", "jewelry", "jewellery"],
    
    # ---- ETHNIC/TRADITIONAL ----
    "kurta": ["kurta", "kurtas", "ethnic", "traditional", "kurti", "kurtis"],
    "kurti": ["kurta", "kurtas", "ethnic", "traditional", "kurti", "kurtis"],
    "saree": ["saree", "sari", "sarees", "saris", "ethnic", "traditional"],
    "sherwani": ["sherwani", "ethnic", "traditional"],
    "ethnic": ["kurta", "sherwani", "traditional", "ethnic wear", "saree", "kurti"],
    "bottomwear": ["jeans", "pants", "trousers", "bottom", "bottomwear", "shorts", "joggers", "cargos", "leggings", "palazzo", "salwar", "sharara"],
    
    # ---- SPORTS/ACTIVEWEAR ----
    "tracksuit": ["tracksuit", "activewear", "sports", "gymwear"],
    "activewear": ["activewear", "sports", "tracksuit", "gymwear", "workout"],
    "sportswear": ["activewear", "sports", "tracksuit", "jersey"],
    "gymwear": ["activewear", "gymwear", "sports", "workout"],
}

# ============================================================
# CATEGORY KEYWORDS - For natural language detection
# Maps keywords found in queries to normalized category
# ============================================================
CATEGORY_KEYWORDS = {
    # keyword → normalized category name
    "sock": "socks",
    "t shirt": "shirt",
    "tshirt": "shirt", 
    "tee": "shirt",
    "top": "top",
    "tops": "top",
    "top wear": "top",
    "upper": "top",
    "upper wear": "top",
    "jersey": "jersey",
    "jerseys": "jersey",
    "jer sey": "jersey",
    "jer jerseys": "jersey",
    "jersy": "jersey",
    "jersys": "jersey",
    "jersyes": "jersey",
    "polo": "polo",
    "polos": "polo",
    "hoodie": "hoodie",
    "hoddie": "hoodie",
    "sweatshirt": "hoodie",
    "sweater": "sweater",
    "pullover": "sweater",
    "bottom": "bottom",
    "bottom wear": "bottom",
    "bottomwear": "bottom",
    "botttom": "bottom",
    "pants": "bottom",
    "pant": "bottom",
    "trouser": "bottom",
    "trousers": "bottom",
    "chinos": "chinos",
    "denim": "jeans",
    "denims": "jeans",
    "jean": "jeans",
    "jeans": "jeans",
    "shoe": "sneakers",
    "sneaker": "sneakers",
    "sneakers": "sneakers",
    "footwear": "sneakers",
    "trainers": "sneakers",
    "jacket": "jacket",
    "jackets": "jacket",
    "coat": "jacket",
    "blazer": "jacket",
    "overcoat": "jacket",
    "puffer": "jacket",
    "bomber": "jacket",
    "shorts": "shorts",
    "jogger": "joggers",
    "joggers": "joggers",
    "trackpants": "joggers",
    "track pants": "joggers",
    "cargo": "cargos",
    "cargos": "cargos",
    "cap": "cap",
    "hat": "cap",
    "beanie": "cap",
    "kurta": "kurta",
    "kurti": "kurta",
    "saree": "saree",
    "sari": "saree",
    "bandhani": "saree",
    "silk saree": "saree",
    "anarkali": "kurta",
    "chikankari": "kurta",
    "jhumka": "earrings",
    "earrings": "earrings",
    "sandals": "sandals",
    "juttis": "sandals",
    "mojaris": "sandals",
    "kolhapuris": "sandals",
    "heels": "sandals",
    "leggings": "bottomwear",
    "palazzo": "bottomwear",
    "sharara": "bottomwear",
    "salwar": "bottomwear",
    "ethnic": "ethnic",
    "traditional": "ethnic",
    "tracksuit": "tracksuit",
    "activewear": "activewear",
    "gymwear": "activewear",
    "gym wear": "activewear",
    "workout": "activewear",
}


def get_db_categories(user_term: str) -> List[str]:
    """
    Get database categories for a user search term.
    
    Args:
        user_term: What user searched for (e.g., "jersey", "sneakers")
    
    Returns:
        List of database category values to search
    """
    return CATEGORY_MAP.get(user_term.lower(), [user_term.lower()])


def normalize_category(query: str) -> Optional[str]:
    """
    Extract normalized category from natural language query.
    
    Args:
        query: Full user query (e.g., "I want a jersey")
    
    Returns:
        Normalized category name or None
    """
    query_lower = query.lower()
    
    # Check keywords (longest match first to avoid partial matches)
    for keyword in sorted(CATEGORY_KEYWORDS.keys(), key=len, reverse=True):
        if keyword in query_lower:
            return CATEGORY_KEYWORDS[keyword]
    
    return None


def get_all_categories() -> List[str]:
    """Get list of all valid user-facing category names."""
    return list(CATEGORY_MAP.keys())


def is_name_priority_term(query: str) -> Optional[str]:
    """
    Check if query contains a term that should search by name, not category.
    
    Args:
        query: User's search query
    
    Returns:
        The matched name priority term, or None
    """
    query_lower = query.lower()
    for term in NAME_PRIORITY_TERMS:
        if term in query_lower:
            return term
    return None

