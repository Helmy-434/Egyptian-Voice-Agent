from pydantic import BaseModel , ConfigDict , Field
from datetime import datetime , timezone
from enum import StrEnum
from uuid import uuid4
from typing import Optional


class ActionStatus(StrEnum):
    
    PENDING_CONFIRMATION = "pending_confirmation"
    CONFIRMED = "confirmed"
    EXECUTING = "executing"
    SENT = "sent"
    FAILED = "failed"
    CANCELLED = "cancelled"
    
    

class PendingAction(BaseModel):
    
    model_config = ConfigDict(extra="forbid")
    
    action_id: str = Field(default_factory=lambda: str(uuid4()))

    tool: str
    recipient: str
    subject: str
    content: str
    summary: str

    confirmed: bool = False
    status: ActionStatus = ActionStatus.PENDING_CONFIRMATION

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    


class Message(BaseModel):

    model_config = ConfigDict(extra="forbid")

    role: str
    content: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class SessionState(BaseModel):

    model_config = ConfigDict(extra="forbid")

    session_id: str

    # Conversation history
    messages: list[Message] = Field(default_factory=list)

    user_email: Optional[str] = None

    pending_action: Optional[PendingAction] = None

    # Session flags
    summary_offered: bool = False # if user want summary
    escalated: bool = False # if the agent can't solve issue , we will redirect user to support team

    # Sending limits and duplicate prevention
    send_count: int = 0
    processed_action_ids: set[str] = Field(default_factory=set)

    
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )