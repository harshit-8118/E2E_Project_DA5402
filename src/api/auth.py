# src/api/auth.py
# Auth endpoints — signup, OTP verify, login, profile

from fastapi import APIRouter, HTTPException, status
from src.db.mongodb import mongo
from src.db.models import UserCreate, UserLogin, OTPRequest, OTPVerify, Token, UserPublic
from src.utils.email_otp import generate_otp, send_otp_email
from src.api.deps import create_access_token, get_current_user
from fastapi import Depends
from src.utils.logger import get_logger

logger = get_logger("auth")

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/signup", summary="Register new user — sends OTP to email")
def signup(body: UserCreate):
    # validate username length
    if len(body.username.strip()) < 3:
        raise HTTPException(status_code=422, detail="Username must be at least 3 characters")

    # validate gender
    valid_genders = {"male", "female", "other", "prefer_not_to_say"}
    if body.gender not in valid_genders:
        raise HTTPException(status_code=422, detail=f"gender must be one of {valid_genders}")

    # validate password strength
    if len(body.password) < 8:
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters")

    # create user (unverified)
    result = mongo.create_user(
        username=body.username,
        email=body.email,
        password=body.password,
        gender=body.gender,
    )
    if not result["success"]:
        raise HTTPException(status_code=409, detail=result["error"])

    # generate and send OTP
    otp     = generate_otp()
    saved   = mongo.save_otp(body.email, otp)
    emailed = send_otp_email(body.email, otp, body.username)

    if not saved:
        raise HTTPException(status_code=500, detail="Failed to generate OTP — try again")

    logger.info(f"Signup: {body.username} | {body.email} | email_sent={emailed}")
    return {
        "message"     : "Account created. Check your email for the 6-digit OTP to verify your account.",
        "uid"         : result["uid"],
        "email_sent"  : emailed,
        "dev_note"    : "If GMAIL_USER not set, check server logs for OTP (dev mode)",
    }


@router.post("/verify-otp", summary="Verify OTP to activate account")
def verify_otp(body: OTPVerify):
    valid = mongo.verify_otp(body.email, body.otp)
    if not valid:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP. Request a new one.")

    mongo.verify_user(body.email)
    logger.info(f"Email verified: {body.email}")
    return {"message": "Email verified successfully. You can now login."}


@router.post("/resend-otp", summary="Resend OTP to email")
def resend_otp(body: OTPRequest):
    user = mongo.get_user_by_email(body.email)
    if not user:
        raise HTTPException(status_code=404, detail="Email not registered")
    if user.get("verified"):
        raise HTTPException(status_code=409, detail="Email already verified")

    otp   = generate_otp()
    saved = mongo.save_otp(body.email, otp)
    send_otp_email(body.email, otp, user.get("username", ""))

    return {"message": "OTP resent. Check your email."}


@router.post("/login", response_model=Token, summary="Login with email + password")
def login(body: UserLogin):
    user = mongo.authenticate_user(body.email, body.password)

    if not user:
        # give generic message — don't reveal whether email exists
        raw = mongo.get_user_by_email(body.email)
        if raw and not raw.get("verified"):
            raise HTTPException(
                status_code=403,
                detail="Email not verified. Please verify your email with the OTP sent during signup."
            )
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(user["uid"], user["username"], user["email"])
    logger.info(f"Login: {user['username']} | {user['email']}")

    return Token(
        access_token=token,
        uid=user["uid"],
        username=user["username"],
        email=user["email"],
    )


@router.get("/me", response_model=UserPublic, summary="Get current user profile")
def me(current_user: dict = Depends(get_current_user)):
    stats = mongo.get_user_stats(current_user["uid"])
    if not stats:
        raise HTTPException(status_code=404, detail="User not found")
    return UserPublic(**{
        "uid"             : stats["uid"],
        "username"        : stats["username"],
        "email"           : stats["email"],
        "gender"          : stats["gender"],
        "verified"        : stats["verified"],
        "created_at"      : stats["created_at"],
        "prediction_count": stats.get("prediction_count", 0),
        "feedback_count"  : stats.get("feedback_count", 0),
    })


@router.get("/my-predictions", summary="Get current user prediction history")
def my_predictions(
    limit: int = 20,
    current_user: dict = Depends(get_current_user)
):
    preds = mongo.get_user_predictions(current_user["uid"], limit=limit)
    return {"uid": current_user["uid"], "predictions": preds, "count": len(preds)}