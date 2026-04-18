# src/db/mongodb.py
# MongoDB client — users, predictions, feedback, images, otp_store
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(
    dotenv_path=Path(__file__).resolve().parents[2] / ".env",
    override=True
)

import os
import uuid
import base64
import hashlib
from datetime import datetime, timedelta
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger("mongodb")

try:
    from pymongo import MongoClient, DESCENDING, ASCENDING
    from pymongo.errors import DuplicateKeyError, ConnectionFailure
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    logger.warning("pymongo not installed — MongoDB disabled")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB",  "CUSTOM_DB_NAME")


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


class MongoDB:
    def __init__(self):
        self.client    = None
        self.db        = None
        self.connected = False
        self._connect()

    def _connect(self):
        if not MONGO_AVAILABLE:
            return
        try:
            self.client    = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
            self.client.admin.command("ping")
            self.db        = self.client[MONGO_DB]
            self.connected = True
            logger.info(f"MongoDB connected | db={MONGO_DB}")
            self._ensure_indexes()
        except Exception as e:
            self.connected = False
            logger.warning(f"MongoDB not reachable: {e}")

    def _ensure_indexes(self):
        try:
            # users — unique username and email
            self.db.users.create_index("uid",      unique=True)
            self.db.users.create_index("email",    unique=True)
            self.db.users.create_index("username", unique=True)

            # otp_store — TTL index expires docs after 600 seconds (10 min)
            self.db.otp_store.create_index("expires_at", expireAfterSeconds=0)
            self.db.otp_store.create_index("email")

            # predictions
            self.db.predictions.create_index("prediction_id", unique=True)
            self.db.predictions.create_index("uid")
            self.db.predictions.create_index("username")
            self.db.predictions.create_index("timestamp")
            self.db.predictions.create_index("predicted_class")

            # feedback
            self.db.feedback.create_index("prediction_id", unique=True)
            self.db.feedback.create_index("uid")
            self.db.feedback.create_index("username")

            # images
            self.db.images.create_index("image_id",      unique=True)
            self.db.images.create_index("prediction_id")
            self.db.images.create_index("uid")

            # rate limiting index — for 50 req/hour alert
            self.db.request_log.create_index("uid")
            self.db.request_log.create_index(
                "timestamp",
                expireAfterSeconds=3600   # auto-delete after 1 hour
            )

            logger.info("MongoDB indexes created")
        except Exception as e:
            logger.warning(f"Index creation failed: {e}")

    def is_up(self) -> bool:
        if not self.connected or not self.client:
            return False
        try:
            self.client.admin.command("ping")
            return True
        except Exception:
            return False

    # ── users ──────────────────────────────────────────────────────────────────

    def create_user(self, username: str, email: str, password: str, gender: str) -> dict:
        """Returns dict with success/error."""
        if not self.connected:
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
            if "email" in err:
                return {"success": False, "error": "Email already registered"}
            if "username" in err:
                return {"success": False, "error": "Username already taken"}
            return {"success": False, "error": "User already exists"}
        except Exception as e:
            logger.error(f"create_user failed: {e}")
            return {"success": False, "error": str(e)}

    def verify_user(self, email: str) -> bool:
        """Mark user as email-verified."""
        if not self.connected:
            return False
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
        if not self.connected:
            return None
        try:
            return self.db.users.find_one({"email": email.lower()}, {"_id": 0})
        except Exception:
            return None

    def get_user_by_uid(self, uid: str) -> Optional[dict]:
        if not self.connected:
            return None
        try:
            return self.db.users.find_one({"uid": uid}, {"_id": 0, "password_hash": 0})
        except Exception:
            return None

    def authenticate_user(self, email: str, password: str) -> Optional[dict]:
        """Returns user doc if credentials valid and verified, else None."""
        user = self.get_user_by_email(email)
        if not user:
            return None
        if user.get("password_hash") != hash_password(password):
            return None
        if not user.get("verified"):
            return None
        return user

    def get_user_stats(self, uid: str) -> dict:
        if not self.connected:
            return {}
        try:
            user = self.db.users.find_one({"uid": uid}, {"_id": 0, "password_hash": 0})
            if not user:
                return {}
            pred_count = self.db.predictions.count_documents({"uid": uid})
            fb_count   = self.db.feedback.count_documents({"uid": uid})
            img_count  = self.db.images.count_documents({"uid": uid})
            return {
                **user,
                "prediction_count": pred_count,
                "feedback_count"  : fb_count,
                "image_count"     : img_count,
            }
        except Exception as e:
            logger.error(f"get_user_stats failed: {e}")
            return {}

    def get_all_users_count(self) -> int:
        if not self.connected: return 0
        try:   return self.db.users.count_documents({})
        except: return 0

    def get_verified_users_count(self) -> int:
        if not self.connected: return 0
        try:   return self.db.users.count_documents({"verified": True})
        except: return 0

    # ── OTP ────────────────────────────────────────────────────────────────────

    def save_otp(self, email: str, otp: str) -> bool:
        if not self.connected: return False
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
        if not self.connected: return False
        try:
            doc = self.db.otp_store.find_one({
                "email": email.lower(),
                "otp"  : otp,
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
        """Log request and return count in last hour."""
        if not self.connected: return 0
        try:
            self.db.request_log.insert_one({
                "uid"      : uid,
                "endpoint" : endpoint,
                "timestamp": datetime.utcnow(),
            })
            # count in last hour
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)
            return self.db.request_log.count_documents({
                "uid"      : uid,
                "timestamp": {"$gte": one_hour_ago}
            })
        except Exception as e:
            logger.error(f"log_request failed: {e}")
            return 0

    def get_requests_last_hour(self, uid: str) -> int:
        if not self.connected: return 0
        try:
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)
            return self.db.request_log.count_documents({
                "uid"      : uid,
                "timestamp": {"$gte": one_hour_ago}
            })
        except: return 0

    # ── predictions ────────────────────────────────────────────────────────────

    def save_prediction(
        self, prediction_id: str, uid: str, username: str,
        predicted_class: str, confidence: float, risk_level: str,
        all_scores: dict, inference_ms: float,
        image_id: Optional[str] = None, image_filename: Optional[str] = None,
    ) -> bool:
        if not self.connected: return False
        try:
            doc = {
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
            }
            self.db.predictions.insert_one(doc)
            self.db.users.update_one(
                {"uid": uid},
                {"$inc": {"prediction_count": 1}}
            )
            return True
        except Exception as e:
            logger.error(f"save_prediction failed: {e}")
            return False

    def get_user_predictions(self, uid: str, limit: int = 20) -> list:
        if not self.connected: return []
        try:
            return list(
                self.db.predictions
                .find({"uid": uid}, {"_id": 0})
                .sort("timestamp", DESCENDING)
                .limit(limit)
            )
        except: return []

    def get_prediction_stats(self) -> dict:
        if not self.connected:
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
        """Save image as base64 for future retraining. Returns image_id."""
        if not self.connected: return None
        try:
            image_id  = str(uuid.uuid4())
            b64_data  = base64.b64encode(image_bytes).decode("utf-8")
            size_kb   = len(image_bytes) / 1024
            self.db.images.insert_one({
                "image_id"     : image_id,
                "prediction_id": prediction_id,
                "uid"          : uid,
                "username"     : username,
                "filename"     : filename or "unknown.jpg",
                "image_b64"    : b64_data,
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
        if not self.connected: return False
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
        if not self.connected:
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
            logger.info("MongoDB closed")


mongo = MongoDB()