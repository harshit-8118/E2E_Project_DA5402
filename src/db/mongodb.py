# src/db/mongodb.py
# MongoDB client — replaces in-memory feedback_store
# Collections: feedback, predictions, images

import os
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger("mongodb")

# ── optional import — graceful fallback if pymongo not installed ───────────────
try:
    from pymongo import MongoClient, DESCENDING
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    logger.warning("pymongo not installed — MongoDB disabled, using in-memory store")

MONGO_URI  = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB   = os.getenv("MONGO_DB",  "skin_disease_detection")


class MongoDB:
    """
    MongoDB wrapper with graceful fallback to in-memory dict.
    All methods work regardless of whether MongoDB is reachable.
    """

    def __init__(self):
        self.client     = None
        self.db         = None
        self.connected  = False
        self._fallback  = {"feedback": {}, "predictions": []}
        self._connect()

    def _connect(self):
        if not MONGO_AVAILABLE:
            logger.warning("Using in-memory fallback — pymongo not available")
            return
        try:
            self.client    = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
            self.client.admin.command("ping")   # verify connection
            self.db        = self.client[MONGO_DB]
            self.connected = True
            logger.info(f"MongoDB connected | uri={MONGO_URI} | db={MONGO_DB}")
            self._ensure_indexes()
        except Exception as e:
            self.connected = False
            logger.warning(f"MongoDB not reachable ({e}) — using in-memory fallback")

    def _ensure_indexes(self):
        """Create indexes for frequent query patterns."""
        try:
            self.db.feedback.create_index("prediction_id", unique=True)
            self.db.feedback.create_index("timestamp")
            self.db.feedback.create_index("vote")
            self.db.predictions.create_index("prediction_id", unique=True)
            self.db.predictions.create_index("predicted_class")
            self.db.predictions.create_index("timestamp")
        except Exception as e:
            logger.warning(f"Index creation failed: {e}")

    def is_up(self) -> bool:
        """Check if MongoDB is reachable — used by Prometheus gauge."""
        if not self.connected or self.client is None:
            return False
        try:
            self.client.admin.command("ping")
            return True
        except Exception:
            return False

    # ── feedback ───────────────────────────────────────────────────────────────

    def save_feedback(self, prediction_id: str, vote: str, comment: str = "") -> bool:
        doc = {
            "prediction_id": prediction_id,
            "vote"         : vote,
            "comment"      : comment,
            "timestamp"    : datetime.utcnow().isoformat(),
        }
        if self.connected:
            try:
                self.db.feedback.update_one(
                    {"prediction_id": prediction_id},
                    {"$set": doc},
                    upsert=True
                )
                return True
            except Exception as e:
                logger.error(f"MongoDB save_feedback failed: {e}")
        # fallback
        self._fallback["feedback"][prediction_id] = doc
        return False

    def get_feedback_stats(self) -> dict:
        if self.connected:
            try:
                total       = self.db.feedback.count_documents({})
                thumbs_up   = self.db.feedback.count_documents({"vote": "thumbs_up"})
                thumbs_down = self.db.feedback.count_documents({"vote": "thumbs_down"})
                return {
                    "total"        : total,
                    "thumbs_up"    : thumbs_up,
                    "thumbs_down"  : thumbs_down,
                    "positive_rate": round(thumbs_up / total, 4) if total > 0 else 0.0,
                    "source"       : "mongodb",
                }
            except Exception as e:
                logger.error(f"MongoDB get_feedback_stats failed: {e}")
        # fallback
        fb    = self._fallback["feedback"]
        total = len(fb)
        up    = sum(1 for f in fb.values() if f["vote"] == "thumbs_up")
        return {
            "total"        : total,
            "thumbs_up"    : up,
            "thumbs_down"  : total - up,
            "positive_rate": round(up / total, 4) if total > 0 else 0.0,
            "source"       : "memory",
        }

    def get_all_feedback(self, limit: int = 100) -> list:
        if self.connected:
            try:
                return list(
                    self.db.feedback
                    .find({}, {"_id": 0})
                    .sort("timestamp", DESCENDING)
                    .limit(limit)
                )
            except Exception as e:
                logger.error(f"MongoDB get_all_feedback failed: {e}")
        return list(self._fallback["feedback"].values())[-limit:]

    # ── predictions ────────────────────────────────────────────────────────────

    def save_prediction(
        self,
        prediction_id : str,
        predicted_class: str,
        confidence    : float,
        risk_level    : str,
        all_scores    : dict,
        inference_ms  : float,
        image_filename: Optional[str] = None,
    ) -> bool:
        doc = {
            "prediction_id" : prediction_id,
            "predicted_class": predicted_class,
            "confidence"    : confidence,
            "risk_level"    : risk_level,
            "all_scores"    : all_scores,
            "inference_ms"  : inference_ms,
            "image_filename": image_filename,
            "timestamp"     : datetime.utcnow().isoformat(),
        }
        if self.connected:
            try:
                self.db.predictions.insert_one(doc)
                return True
            except Exception as e:
                logger.error(f"MongoDB save_prediction failed: {e}")
        self._fallback["predictions"].append(doc)
        return False

    def get_prediction_stats(self) -> dict:
        if self.connected:
            try:
                total    = self.db.predictions.count_documents({})
                pipeline = [{"$group": {"_id": "$predicted_class", "count": {"$sum": 1}}}]
                by_class = {r["_id"]: r["count"] for r in self.db.predictions.aggregate(pipeline)}
                high_risk = self.db.predictions.count_documents({"risk_level": "HIGH"})
                return {
                    "total_predictions": total,
                    "by_class"         : by_class,
                    "high_risk_total"  : high_risk,
                    "source"           : "mongodb",
                }
            except Exception as e:
                logger.error(f"MongoDB get_prediction_stats failed: {e}")
        preds    = self._fallback["predictions"]
        by_class = {}
        for p in preds:
            by_class[p["predicted_class"]] = by_class.get(p["predicted_class"], 0) + 1
        return {
            "total_predictions": len(preds),
            "by_class"         : by_class,
            "high_risk_total"  : sum(1 for p in preds if p.get("risk_level") == "HIGH"),
            "source"           : "memory",
        }

    def close(self):
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


# ── singleton instance ─────────────────────────────────────────────────────────
mongo = MongoDB()