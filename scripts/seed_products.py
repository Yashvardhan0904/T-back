"""
Seed script to add fashion products to MongoDB for testing the multi-agent system.
"""

import asyncio
import sys
import os
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.mongodb import connect_to_mongo, get_database, close_mongo_connection
from app.core.config import get_settings

# Gen-Z Fashion Products for testing
FASHION_PRODUCTS = [
    # SNEAKERS (for testing price filters and category search)
    {
        "name": "Nike Air Force 1 White",
        "description": "Classic white sneakers, perfect for clean-boy aesthetic. Timeless design.",
        "price": 7999,
        "category": "sneakers",
        "gender": "unisex",
        "color": "white",
        "fabric": "leather",
        "fit": "regular",
        "occasion": "casual",
        "brand": "Nike",
        "style": "clean boy",
        "formality": "casual",
        "images": [],
        "tags": ["sneakers", "white", "classic", "clean boy"],
        "stock": 50,
        "isActive": True,
        "isApproved": True,
        "isVisible": True,
        "createdAt": datetime.now(timezone.utc)
    },
    {
        "name": "Adidas Samba OG",
        "description": "Retro-inspired sneakers with suede upper. Low-key fire for streetwear fits.",
        "price": 5999,
        "category": "sneakers",
        "gender": "unisex",
        "color": "black",
        "fabric": "suede",
        "fit": "regular",
        "occasion": "casual",
        "brand": "Adidas",
        "style": "streetwear",
        "formality": "casual",
        "images": [],
        "tags": ["sneakers", "black", "retro", "streetwear"],
        "stock": 40,
        "isActive": True,
        "isApproved": True,
        "isVisible": True,
        "createdAt": datetime.now(timezone.utc)
    },
    {
        "name": "Vans Old Skool",
        "description": "Iconic skate shoes. Perfect for college days and casual outings.",
        "price": 4999,
        "category": "sneakers",
        "gender": "unisex",
        "color": "black",
        "fabric": "canvas",
        "fit": "regular",
        "occasion": "casual",
        "brand": "Vans",
        "style": "streetwear",
        "formality": "casual",
        "images": [],
        "tags": ["sneakers", "skate", "casual", "college"],
        "stock": 60,
        "isActive": True,
        "isApproved": True,
        "isVisible": True,
        "createdAt": datetime.now(timezone.utc)
    },
    {
        "name": "Converse Chuck 70",
        "description": "Premium canvas sneakers. Clean, minimal, versatile.",
        "price": 6499,
        "category": "sneakers",
        "gender": "unisex",
        "color": "white",
        "fabric": "canvas",
        "fit": "regular",
        "occasion": "casual",
        "brand": "Converse",
        "style": "clean boy",
        "formality": "casual",
        "images": [],
        "tags": ["sneakers", "canvas", "minimal", "versatile"],
        "stock": 45,
        "isActive": True,
        "isApproved": True,
        "isVisible": True,
        "createdAt": datetime.now(timezone.utc)
    },
    {
        "name": "New Balance 550",
        "description": "Retro basketball-inspired sneakers. Y2K vibes, clean aesthetic.",
        "price": 8999,
        "category": "sneakers",
        "gender": "unisex",
        "color": "white",
        "fabric": "leather",
        "fit": "regular",
        "occasion": "casual",
        "brand": "New Balance",
        "style": "Y2K",
        "formality": "casual",
        "images": [],
        "tags": ["sneakers", "retro", "Y2K", "basketball"],
        "stock": 35,
        "isActive": True,
        "isApproved": True,
        "isVisible": True,
        "createdAt": datetime.now(timezone.utc)
    },
    
    # TOPS (for outfit building)
    {
        "name": "Oversized Black T-Shirt",
        "description": "Heavyweight cotton tee with relaxed fit. Perfect for clean-boy aesthetic.",
        "price": 1299,
        "category": "shirt",
        "gender": "unisex",
        "color": "black",
        "fabric": "cotton",
        "fit": "oversized",
        "occasion": "casual",
        "brand": "Essentials",
        "style": "clean boy",
        "formality": "casual",
        "images": [],
        "tags": ["tee", "oversized", "black", "minimal"],
        "stock": 100,
        "isActive": True,
        "isApproved": True,
        "isVisible": True,
        "createdAt": datetime.now(timezone.utc)
    },
    {
        "name": "White Basic Tee",
        "description": "Classic white t-shirt. Essential wardrobe piece.",
        "price": 899,
        "category": "shirt",
        "gender": "unisex",
        "color": "white",
        "fabric": "cotton",
        "fit": "regular",
        "occasion": "casual",
        "brand": "Basics",
        "style": "clean boy",
        "formality": "casual",
        "images": [],
        "tags": ["tee", "white", "basic", "essential"],
        "stock": 150,
        "isActive": True,
        "isApproved": True,
        "isVisible": True,
        "createdAt": datetime.now(timezone.utc)
    },
    {
        "name": "Streetwear Hoodie",
        "description": "Oversized hoodie with bold graphics. Streetwear essential.",
        "price": 2999,
        "category": "hoodie",
        "gender": "unisex",
        "color": "black",
        "fabric": "cotton",
        "fit": "oversized",
        "occasion": "casual",
        "brand": "Streetwear Co",
        "style": "streetwear",
        "formality": "casual",
        "images": [],
        "tags": ["hoodie", "oversized", "streetwear", "graphic"],
        "stock": 80,
        "isActive": True,
        "isApproved": True,
        "isVisible": True,
        "createdAt": datetime.now(timezone.utc)
    },
    {
        "name": "Minimalist Crew Neck",
        "description": "Clean, simple crew neck sweater. Old money aesthetic.",
        "price": 2499,
        "category": "sweater",
        "gender": "unisex",
        "color": "cream",
        "fabric": "cotton",
        "fit": "regular",
        "occasion": "casual",
        "brand": "Minimal",
        "style": "old money",
        "formality": "casual",
        "images": [],
        "tags": ["sweater", "minimal", "cream", "preppy"],
        "stock": 60,
        "isActive": True,
        "isApproved": True,
        "isVisible": True,
        "createdAt": datetime.now(timezone.utc)
    },
    
    # BOTTOMS (for outfit building)
    {
        "name": "Relaxed Fit Cargo Pants",
        "description": "Baggy cargos with multiple pockets. Streetwear staple.",
        "price": 3499,
        "category": "pants",
        "gender": "unisex",
        "color": "olive",
        "fabric": "cotton",
        "fit": "loose",
        "occasion": "casual",
        "brand": "Streetwear Co",
        "style": "streetwear",
        "formality": "casual",
        "images": [],
        "tags": ["cargos", "baggy", "streetwear", "olive"],
        "stock": 70,
        "isActive": True,
        "isApproved": True,
        "isVisible": True,
        "createdAt": datetime.now(timezone.utc)
    },
    {
        "name": "Straight Fit Jeans",
        "description": "Classic straight leg jeans. Versatile and timeless.",
        "price": 2799,
        "category": "jeans",
        "gender": "unisex",
        "color": "blue",
        "fabric": "denim",
        "fit": "regular",
        "occasion": "casual",
        "brand": "Levi's",
        "style": "clean boy",
        "formality": "casual",
        "images": [],
        "tags": ["jeans", "straight", "blue", "classic"],
        "stock": 90,
        "isActive": True,
        "isApproved": True,
        "isVisible": True,
        "createdAt": datetime.now(timezone.utc)
    },
    {
        "name": "Black Denim Jeans",
        "description": "Slim black jeans. Clean and versatile.",
        "price": 2499,
        "category": "jeans",
        "gender": "unisex",
        "color": "black",
        "fabric": "denim",
        "fit": "slim",
        "occasion": "casual",
        "brand": "Zara",
        "style": "clean boy",
        "formality": "casual",
        "images": [],
        "tags": ["jeans", "black", "slim", "versatile"],
        "stock": 75,
        "isActive": True,
        "isApproved": True,
        "isVisible": True,
        "createdAt": datetime.now(timezone.utc)
    },
    
    # LAYERS (for complete outfits)
    {
        "name": "Denim Jacket",
        "description": "Classic denim jacket. Perfect layering piece.",
        "price": 3999,
        "category": "jacket",
        "gender": "unisex",
        "color": "blue",
        "fabric": "denim",
        "fit": "regular",
        "occasion": "casual",
        "brand": "Levi's",
        "style": "clean boy",
        "formality": "casual",
        "images": [],
        "tags": ["jacket", "denim", "layering", "classic"],
        "stock": 55,
        "isActive": True,
        "isApproved": True,
        "isVisible": True,
        "createdAt": datetime.now(timezone.utc)
    },
    {
        "name": "Puffer Jacket",
        "description": "Lightweight puffer jacket. Gorpcore aesthetic.",
        "price": 4999,
        "category": "jacket",
        "gender": "unisex",
        "color": "black",
        "fabric": "nylon",
        "fit": "regular",
        "occasion": "casual",
        "brand": "Outdoor Co",
        "style": "gorpcore",
        "formality": "casual",
        "images": [],
        "tags": ["puffer", "outdoor", "gorpcore", "functional"],
        "stock": 40,
        "isActive": True,
        "isApproved": True,
        "isVisible": True,
        "createdAt": datetime.now(timezone.utc)
    },
    
    # BUDGET OPTIONS (for price filter testing)
    {
        "name": "Budget White Sneakers",
        "description": "Affordable white sneakers. Clean look without breaking the bank.",
        "price": 1499,
        "category": "sneakers",
        "gender": "unisex",
        "color": "white",
        "fabric": "synthetic",
        "fit": "regular",
        "occasion": "casual",
        "brand": "Budget Brand",
        "style": "clean boy",
        "formality": "casual",
        "images": [],
        "tags": ["sneakers", "white", "budget", "affordable"],
        "stock": 200,
        "isActive": True,
        "isApproved": True,
        "isVisible": True,
        "createdAt": datetime.now(timezone.utc)
    },
    {
        "name": "Basic Black Sneakers",
        "description": "Simple black sneakers under 2000. Perfect for daily wear.",
        "price": 1799,
        "category": "sneakers",
        "gender": "unisex",
        "color": "black",
        "fabric": "synthetic",
        "fit": "regular",
        "occasion": "casual",
        "brand": "Budget Brand",
        "style": "clean boy",
        "formality": "casual",
        "images": [],
        "tags": ["sneakers", "black", "budget", "daily"],
        "stock": 180,
        "isActive": True,
        "isApproved": True,
        "isVisible": True,
        "createdAt": datetime.now(timezone.utc)
    },
    
    # SOCKS (for testing)
    {
        "name": "Basic White Ankle Socks",
        "description": "Comfortable white ankle socks. Essential for sneakers.",
        "price": 299,
        "category": "socks",
        "gender": "unisex",
        "color": "white",
        "fabric": "cotton",
        "fit": "regular",
        "occasion": "casual",
        "brand": "Basics",
        "style": "clean boy",
        "formality": "casual",
        "images": [],
        "tags": ["socks", "white", "ankle", "basic"],
        "stock": 200,
        "isActive": True,
        "isApproved": True,
        "isVisible": True,
        "createdAt": datetime.now(timezone.utc)
    },
    {
        "name": "Black Crew Socks",
        "description": "Classic black crew socks. Perfect for everyday wear.",
        "price": 349,
        "category": "socks",
        "gender": "unisex",
        "color": "black",
        "fabric": "cotton",
        "fit": "regular",
        "occasion": "casual",
        "brand": "Basics",
        "style": "clean boy",
        "formality": "casual",
        "images": [],
        "tags": ["socks", "black", "crew", "classic"],
        "stock": 180,
        "isActive": True,
        "isApproved": True,
        "isVisible": True,
        "createdAt": datetime.now(timezone.utc)
    },
    
    # MORE T-SHIRTS (to have more variety)
    {
        "name": "Graphic Tee Streetwear",
        "description": "Bold graphic t-shirt with streetwear vibes. Statement piece.",
        "price": 1599,
        "category": "shirt",
        "gender": "unisex",
        "color": "black",
        "fabric": "cotton",
        "fit": "oversized",
        "occasion": "casual",
        "brand": "Streetwear Co",
        "style": "streetwear",
        "formality": "casual",
        "images": [],
        "tags": ["tee", "graphic", "streetwear", "bold"],
        "stock": 70,
        "isActive": True,
        "isApproved": True,
        "isVisible": True,
        "createdAt": datetime.now(timezone.utc)
    },
    {
        "name": "Minimalist White Tee",
        "description": "Clean white t-shirt. Wardrobe essential.",
        "price": 999,
        "category": "shirt",
        "gender": "unisex",
        "color": "white",
        "fabric": "cotton",
        "fit": "regular",
        "occasion": "casual",
        "brand": "Minimal",
        "style": "clean boy",
        "formality": "casual",
        "images": [],
        "tags": ["tee", "white", "minimal", "essential"],
        "stock": 120,
        "isActive": True,
        "isApproved": True,
        "isVisible": True,
        "createdAt": datetime.now(timezone.utc)
    }
]


async def seed_products():
    """Seed fashion products into MongoDB."""
    try:
        # Connect to MongoDB
        settings = get_settings()
        print(f"Connecting to MongoDB...")
        print(f"DB_NAME from settings: {settings.DB_NAME}")
        
        # Force use Yashvardhan database
        from motor.motor_asyncio import AsyncIOMotorClient
        client = AsyncIOMotorClient(settings.MONGODB_URI)
        db = client["Yashvardhan"]  # Use Yashvardhan database directly
        
        # Verify connection
        await client.admin.command('ping')
        print(f"Connected! Using database: {db.name}")
        
        print("Seeding fashion products...")
        
        # Clear existing products (optional - comment out if you want to keep existing)
        # await db.products.delete_many({})
        # print("Cleared existing products")
        
        # Insert products one by one to handle duplicates
        inserted_count = 0
        for product in FASHION_PRODUCTS:
            try:
                # Check if product already exists (by name)
                existing = await db.products.find_one({"name": product["name"]})
                if not existing:
                    await db.products.insert_one(product)
                    inserted_count += 1
                else:
                    print(f"  Skipping duplicate: {product['name']}")
            except Exception as e:
                print(f"  Error inserting {product['name']}: {e}")
        
        result_count = inserted_count
        print(f"Successfully inserted {result_count} products!")
        
        # Print summary
        print("\nProduct Summary:")
        categories = {}
        for product in FASHION_PRODUCTS:
            cat = product["category"]
            categories[cat] = categories.get(cat, 0) + 1
        
        for cat, count in categories.items():
            print(f"  - {cat}: {count} products")
        
        # Price range summary
        prices = [p["price"] for p in FASHION_PRODUCTS]
        print(f"\nPrice Range: Rs {min(prices)} - Rs {max(prices)}")
        print(f"   Products under Rs 2000: {sum(1 for p in prices if p < 2000)}")
        print(f"   Products under Rs 5000: {sum(1 for p in prices if p < 5000)}")
        
        print("\nSeeding complete! Products are ready for testing.")
        
    except Exception as e:
        print(f"Error seeding products: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            client.close()
        except:
            pass


if __name__ == "__main__":
    asyncio.run(seed_products())
