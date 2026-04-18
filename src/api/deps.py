# src/api/deps.py
# Shared FastAPI dependencies — JWT auth, current user extraction
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(
    dotenv_path=Path(__file__).resolve().parents[2] / ".env",
    override=True
)

import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

SECRET_KEY      = os.getenv("JWT_SECRET_KEY", "JWT_SECERET_KEY_FROM_ENV")
ALGORITHM       = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", 60))

security = HTTPBearer(auto_error=False)


def create_access_token(uid: str, username: str, email: str) -> str:
    if not JWT_AVAILABLE:
        # simple base64 fallback for environments without PyJWT
        import base64, json
        payload = {"uid": uid, "username": username, "email": email}
        return base64.b64encode(json.dumps(payload).encode()).decode()

    expire  = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub"     : uid,
        "username": username,
        "email"   : email,
        "exp"     : expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    if not JWT_AVAILABLE:
        import base64, json
        try:
            payload = json.loads(base64.b64decode(token).decode())
            return {"uid": payload["uid"], "username": payload["username"], "email": payload["email"]}
        except Exception:
            return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"uid": payload["sub"], "username": payload["username"], "email": payload["email"]}
    except Exception:
        return None


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> dict:
    """Dependency — raises 401 if token missing or invalid."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated — please login",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = decode_token(credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[dict]:
    """Dependency — returns None if no token (for optional auth endpoints)."""
    if not credentials:
        return None
    return decode_token(credentials.credentials)