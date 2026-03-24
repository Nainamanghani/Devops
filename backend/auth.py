import os
import bcrypt
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any


# =============================
# 🔐 Security Configuration
# =============================

class SecurityConfig:
    SECRET = os.getenv("SECRET_KEY", "energy-secret-key")
    ALGO = "HS256"
    TOKEN_DURATION_MIN = 60 * 24  # 24 hours


# =============================
# 🔑 Password Utilities
# =============================

class PasswordManager:

    @staticmethod
    def hash_password(raw_password: str) -> str:
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(raw_password.encode("utf-8"), salt)
        return hashed.decode("utf-8")

    @staticmethod
    def validate_password(raw_password: str, stored_hash: str) -> bool:
        if isinstance(stored_hash, str):
            stored_hash = stored_hash.encode("utf-8")

        return bcrypt.checkpw(raw_password.encode("utf-8"), stored_hash)


# =============================
# 🎟 Token Management
# =============================

class TokenManager:

    @staticmethod
    def generate_token(payload: Dict[str, Any],
                       expires_in: Optional[int] = None) -> str:

        expiry_minutes = expires_in or SecurityConfig.TOKEN_DURATION_MIN
        expiry_time = datetime.utcnow() + timedelta(minutes=expiry_minutes)

        token_payload = {
            **payload,
            "exp": expiry_time,
            "iat": datetime.utcnow()
        }

        return jwt.encode(token_payload, SecurityConfig.SECRET, algorithm=SecurityConfig.ALGO)

    @staticmethod
    def verify_token(token: str) -> Optional[Dict[str, Any]]:
        try:
            decoded = jwt.decode(
                token,
                SecurityConfig.SECRET,
                algorithms=[SecurityConfig.ALGO]
            )
            return decoded
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None


# =============================
# 🚀 Backward Compatibility Aliases
# (So main.py does not break)
# =============================

verify_password = PasswordManager.validate_password
get_password_hash = PasswordManager.hash_password
create_access_token = TokenManager.generate_token
decode_access_token = TokenManager.verify_token
ACCESS_TOKEN_EXPIRE_MINUTES = SecurityConfig.TOKEN_DURATION_MIN