from app.db.mongodb import get_database
from bson import ObjectId
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class CartService:
    async def add_to_cart(self, user_id: str, product_id: str, quantity: int = 1) -> bool:
        """
        Adds a product to the user's cart in MongoDB.
        """
        db = get_database()
        try:
            # Check if item already in cart
            user = await db.users.find_one({"_id": ObjectId(user_id), "cart.product": ObjectId(product_id)})
            
            if user:
                # Update quantity
                await db.users.update_one(
                    {"_id": ObjectId(user_id), "cart.product": ObjectId(product_id)},
                    {"$inc": {"cart.$.quantity": quantity}}
                )
            else:
                # Add new item
                cart_item = {
                    "product": ObjectId(product_id),
                    "quantity": quantity,
                    "addedAt": datetime.utcnow()
                }
                await db.users.update_one(
                    {"_id": ObjectId(user_id)},
                    {"$push": {"cart": cart_item}}
                )
            
            logger.info(f"Successfully added product {product_id} to cart for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add to cart: {e}")
            return False

    async def get_cart(self, user_id: str) -> list:
        """
        Retrieves the user's cart items with product details.
        """
        db = get_database()
        try:
            user = await db.users.find_one({"_id": ObjectId(user_id)})
            if not user or "cart" not in user:
                return []
            
            cart_items = user["cart"]
            results = []
            for item in cart_items:
                product = await db.products.find_one({"_id": item["product"]})
                if product:
                    product["_id"] = str(product["_id"])
                    product["quantity"] = item["quantity"]
                    results.append(product)
            return results
        except Exception as e:
            logger.error(f"Failed to get cart: {e}")
            return []

    async def clear_cart(self, user_id: str) -> bool:
        """
        Empties the user's cart.
        """
        db = get_database()
        try:
            await db.users.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": {"cart": []}}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to clear cart: {e}")
            return False

cart_service = CartService()
