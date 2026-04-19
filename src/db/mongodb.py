# src/db/mongodb.py
# MongoDB client with auto-reconnect on every operation
# No module-level connection attempt — connects lazily and retries on failure

import os
import uuid
import base64
import hashlib
import time
from datetime import datetime, timedelta
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger("mongodb")

try:
    from pymongo import MongoClient, DESCENDING
    from pymongo.errors import DuplicateKeyError, ConnectionFailure, ServerSelectionTimeoutError
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    logger.warning("pymongo not installed — MongoDB disabled")

# read from environment directly — Docker passes these via env_file in compose
# never rely on load_dotenv inside Docker containers
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB  = os.environ.get("MONGO_DB",  "skin_disease_detection")  # safe default


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


class MongoDB:
    """
    MongoDB client with lazy connection and automatic reconnect.
    Every public method calls _ensure_connected() before operating.
    This means the app keeps working even if MongoDB starts after the backend.
    """

    def __init__(self):
        self.client    = None
        self.db        = None
        self.connected = False
        self._last_attempt = 0
        self._retry_interval = 5   # seconds between reconnect attempts

    def _ensure_connected(self) -> bool:
        """
        Try to connect if not connected. Retries every 5 seconds.
        Returns True if connected, False otherwise.
        """
        if self.connected and self.client:
            try:
                # fast ping to verify connection still alive
                self.client.admin.command("ping")
                return True
            except Exception:
                # connection dropped — reset and try below
                self.connected = False
                self.client    = None
                self.db        = None

        if not MONGO_AVAILABLE:
            return False

        # rate-limit reconnect attempts
        now = time.time()
        if now - self._last_attempt < self._retry_interval:
            return False

        self._last_attempt = now
        try:
            self.client = MongoClient(
                MONGO_URI,
                serverSelectionTimeoutMS=5000,   # 5s timeout on connect
                connectTimeoutMS=5000,
                socketTimeoutMS=10000,
            )
            self.client.admin.command("ping")
            self.db        = self.client[MONGO_DB]
            self.connected = True
            logger.info(f"MongoDB connected | uri={MONGO_URI} | db={MONGO_DB}")
            self._ensure_indexes()
            return True
        except Exception as e:
            self.connected = False
            self.client    = None
            self.db        = None
            logger.warning(f"MongoDB connection failed (will retry): {e}")
            return False

    def _ensure_indexes(self):
        """Create indexes — called once after each successful connection."""
        try:
            self.db.users.create_index("uid",      unique=True)
            self.db.users.create_index("email",    unique=True)
            self.db.users.create_index("username", unique=True)

            self.db.otp_store.create_index("expires_at", expireAfterSeconds=0)
            self.db.otp_store.create_index("email")

            self.db.predictions.create_index("prediction_id", unique=True)
            self.db.predictions.create_index("uid")
            self.db.predictions.create_index("username")
            self.db.predictions.create_index("timestamp")
            self.db.predictions.create_index("predicted_class")

            self.db.feedback.create_index("prediction_id", unique=True)
            self.db.feedback.create_index("uid")
            self.db.feedback.create_index("username")

            self.db.images.create_index("image_id",      unique=True)
            self.db.images.create_index("prediction_id")
            self.db.images.create_index("uid")

            self.db.request_log.create_index("uid")
            self.db.request_log.create_index(
                "timestamp",
                expireAfterSeconds=3600
            )
            logger.info("MongoDB indexes ready")
        except Exception as e:
            logger.warning(f"Index creation warning (may already exist): {e}")

    def is_up(self) -> bool:
        """Check connectivity — used by Prometheus gauge and /health endpoint."""
        return self._ensure_connected()

    # ── users ──────────────────────────────────────────────────────────────────

    def create_user(self, username: str, email: str, password: str, gender: str) -> dict:
        if not self._ensure_connected():
            return {"success": False, "error": "Database not available"}
        try:
            uid = str(uuid.uuid4())
            doc = {
                "uid"             : uid,
                "username"        : username.strip().lower(),
                "email"           : email.strip().lower(),
                "password_hash"   : hash_password(password),
                "gender"          : gender,
                "verified"        : False,
                "created_at"      : datetime.utcnow().isoformat(),
                "prediction_count": 0,
                "feedback_count"  : 0,
            }
            self.db.users.insert_one(doc)
            logger.info(f"User created: {username} | {email}")
            return {"success": True, "uid": uid}
        except DuplicateKeyError as e:
            err = str(e)
            if "email"    in err: return {"success": False, "error": "Email already registered"}
            if "username" in err: return {"success": False, "error": "Username already taken"}
            return {"success": False, "error": "User already exists"}
        except Exception as e:
            logger.error(f"create_user failed: {e}")
            return {"success": False, "error": str(e)}

    def verify_user(self, email: str) -> bool:
        if not self._ensure_connected(): return False
        try:
            self.db.users.update_one(
                {"email": email.lower()},
                {"$set": {"verified": True}}
            )
            return True
        except Exception as e:
            logger.error(f"verify_user failed: {e}")
            return False

    def get_user_by_email(self, email: str) -> Optional[dict]:
        if not self._ensure_connected(): return None
        try:
            return self.db.users.find_one({"email": email.lower()}, {"_id": 0})
        except Exception:
            return None

    def get_user_by_uid(self, uid: str) -> Optional[dict]:
        if not self._ensure_connected(): return None
        try:
            return self.db.users.find_one({"uid": uid}, {"_id": 0, "password_hash": 0})
        except Exception:
            return None

    def authenticate_user(self, email: str, password: str) -> Optional[dict]:
        user = self.get_user_by_email(email)
        if not user:
            return None
        if user.get("password_hash") != hash_password(password):
            return None
        if not user.get("verified"):
            return None
        return user

    def get_user_stats(self, uid: str) -> dict:
        if not self._ensure_connected(): return {}
        try:
            user = self.db.users.find_one({"uid": uid}, {"_id": 0, "password_hash": 0})
            if not user: return {}
            return {
                **user,
                "prediction_count": self.db.predictions.count_documents({"uid": uid}),
                "feedback_count"  : self.db.feedback.count_documents({"uid": uid}),
                "image_count"     : self.db.images.count_documents({"uid": uid}),
            }
        except Exception as e:
            logger.error(f"get_user_stats failed: {e}")
            return {}

    def get_all_users_count(self) -> int:
        if not self._ensure_connected(): return 0
        try:   return self.db.users.count_documents({})
        except: return 0

    def get_verified_users_count(self) -> int:
        if not self._ensure_connected(): return 0
        try:   return self.db.users.count_documents({"verified": True})
        except: return 0

    # ── OTP ────────────────────────────────────────────────────────────────────

    def save_otp(self, email: str, otp: str) -> bool:
        if not self._ensure_connected(): return False
        try:
            expires_at = datetime.utcnow() + timedelta(minutes=10)
            self.db.otp_store.update_one(
                {"email": email.lower()},
                {"$set": {"email": email.lower(), "otp": otp, "expires_at": expires_at}},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"save_otp failed: {e}")
            return False

    def verify_otp(self, email: str, otp: str) -> bool:
        if not self._ensure_connected(): return False
        try:
            doc = self.db.otp_store.find_one({
                "email"     : email.lower(),
                "otp"       : otp,
                "expires_at": {"$gt": datetime.utcnow()}
            })
            if doc:
                self.db.otp_store.delete_one({"email": email.lower()})
                return True
            return False
        except Exception as e:
            logger.error(f"verify_otp failed: {e}")
            return False

    # ── rate limiting ──────────────────────────────────────────────────────────

    def log_request(self, uid: str, endpoint: str) -> int:
        if not self._ensure_connected(): return 0
        try:
            self.db.request_log.insert_one({
                "uid"      : uid,
                "endpoint" : endpoint,
                "timestamp": datetime.utcnow(),
            })
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)
            return self.db.request_log.count_documents({
                "uid"      : uid,
                "timestamp": {"$gte": one_hour_ago}
            })
        except Exception as e:
            logger.error(f"log_request failed: {e}")
            return 0

    # ── predictions ────────────────────────────────────────────────────────────

    def save_prediction(
        self, prediction_id: str, uid: str, username: str,
        predicted_class: str, confidence: float, risk_level: str,
        all_scores: dict, inference_ms: float,
        image_id: Optional[str] = None, image_filename: Optional[str] = None,
    ) -> bool:
        if not self._ensure_connected(): return False
        try:
            self.db.predictions.insert_one({
                "prediction_id"  : prediction_id,
                "uid"            : uid,
                "username"       : username,
                "predicted_class": predicted_class,
                "confidence"     : confidence,
                "risk_level"     : risk_level,
                "all_scores"     : all_scores,
                "inference_ms"   : inference_ms,
                "image_id"       : image_id,
                "image_filename" : image_filename,
                "timestamp"      : datetime.utcnow().isoformat(),
            })
            self.db.users.update_one(
                {"uid": uid},
                {"$inc": {"prediction_count": 1}}
            )
            return True
        except Exception as e:
            logger.error(f"save_prediction failed: {e}")
            return False

    def get_user_predictions(self, uid: str, limit: int = 20) -> list:
        if not self._ensure_connected(): return []
        try:
            return list(
                self.db.predictions
                .find({"uid": uid}, {"_id": 0})
                .sort("timestamp", DESCENDING)
                .limit(limit)
            )
        except: return []

    def get_prediction_stats(self) -> dict:
        if not self._ensure_connected():
            return {"total_predictions": 0, "by_class": {}, "high_risk_total": 0}
        try:
            total    = self.db.predictions.count_documents({})
            pipeline = [{"$group": {"_id": "$predicted_class", "count": {"$sum": 1}}}]
            by_class = {r["_id"]: r["count"] for r in self.db.predictions.aggregate(pipeline)}
            high     = self.db.predictions.count_documents({"risk_level": "HIGH"})
            return {"total_predictions": total, "by_class": by_class, "high_risk_total": high}
        except: return {}

    # ── images ─────────────────────────────────────────────────────────────────

    def save_image(
        self, uid: str, username: str, prediction_id: str,
        image_bytes: bytes, filename: str
    ) -> Optional[str]:
        if not self._ensure_connected(): return None
        try:
            image_id = str(uuid.uuid4())
            size_kb  = len(image_bytes) / 1024
            self.db.images.insert_one({
                "image_id"     : image_id,
                "prediction_id": prediction_id,
                "uid"          : uid,
                "username"     : username,
                "filename"     : filename or "unknown.jpg",
                "image_b64"    : base64.b64encode(image_bytes).decode("utf-8"),
                "image_size_kb": round(size_kb, 2),
                "timestamp"    : datetime.utcnow().isoformat(),
            })
            logger.info(f"Image saved | id={image_id} | user={username} | {size_kb:.1f}KB")
            return image_id
        except Exception as e:
            logger.error(f"save_image failed: {e}")
            return None

    # ── feedback ───────────────────────────────────────────────────────────────

    def save_feedback(
        self, prediction_id: str, uid: str, username: str,
        vote: str, comment: str = ""
    ) -> bool:
        if not self._ensure_connected(): return False
        try:
            self.db.feedback.update_one(
                {"prediction_id": prediction_id},
                {"$set": {
                    "prediction_id": prediction_id,
                    "uid"          : uid,
                    "username"     : username,
                    "vote"         : vote,
                    "comment"      : comment,
                    "timestamp"    : datetime.utcnow().isoformat(),
                }},
                upsert=True
            )
            self.db.users.update_one(
                {"uid": uid},
                {"$inc": {"feedback_count": 1}}
            )
            return True
        except Exception as e:
            logger.error(f"save_feedback failed: {e}")
            return False

    def get_feedback_stats(self) -> dict:
        if not self._ensure_connected():
            return {"total": 0, "thumbs_up": 0, "thumbs_down": 0, "positive_rate": 0.0}
        try:
            total = self.db.feedback.count_documents({})
            up    = self.db.feedback.count_documents({"vote": "thumbs_up"})
            return {
                "total"        : total,
                "thumbs_up"    : up,
                "thumbs_down"  : total - up,
                "positive_rate": round(up / total, 4) if total > 0 else 0.0,
            }
        except: return {}

    def close(self):
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


# singleton — created once at import, connects lazily on first use
mongo = MongoDB()