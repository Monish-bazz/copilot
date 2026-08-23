from pydantic import BaseModel
from typing import Optional

class UserContext(BaseModel):
    user_key: str
    role: str
    account_id: Optional[str] = None
    
USERS = {
    "priya.northstar": UserContext(user_key="priya.northstar", role="customer", account_id="ACCT-001"),
    "arjun.lumenworks": UserContext(user_key="arjun.lumenworks", role="customer", account_id="ACCT-002"),
    "rohit.ops": UserContext(user_key="rohit.ops", role="internal_agent"),
    "admin.ops": UserContext(user_key="admin.ops", role="internal_admin"),
}

def get_user_by_token(token: str) -> Optional[UserContext]:
    return USERS.get(token)
