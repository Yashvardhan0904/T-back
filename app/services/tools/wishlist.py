from app.db.mongodb import get_database
from bson import ObjectId
import logging

logger = logging.getLogger(__name__)

class WishlistService:
    async def add_to_wishlist(self, user_id: str, product_id: str) -> bool:
        """
        Adds a product to the user's wishlist in MongoDB.
        Uses $addToSet to ensure no duplicates.
        """
        db = get_database()
        try:
            await db.users.update_one(
                {"_id": ObjectId(user_id)},
                {"$addToSet": {"wishlist": ObjectId(product_id)}}
            )
            logger.info(f"Successfully added product {product_id} to wishlist for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add to wishlist: {e}")
            return False

wishlist_service = WishlistService()
