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
    """
    Attributes:
        action_id (str).
        tool (str): which tool model call ( email , telegram ).
        recipient (str): reciever of message.
        subject (str): title of message.
        content (str): message content.
        summary (str).
        confirmed (bool): flag for message status.
        status (ActionStatus): 6 status.
        created_at (datetime): timestamp.
    """
    
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
    """
    Attributes:
        role (str): who sent message.
        content (str): message content.
        timestamp (datetime): current time of sending message.
    """
    
    model_config = ConfigDict(extra="forbid")

    role: str
    content: str
    
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )



class SessionState(BaseModel):
    """
    Attributes:
        session_id (str).
        messages (list[Message]): list of messages in this conversation.
        user_email (Optional[str]): email of user.
        pending_action (Optional[PendingAction]): details of action .
        summary_offered (bool): if user want summary.
        escalated (bool): if the agent can't solve issue , we will redirect user to support team.
        send_count (int): limit of success sending emails.
        processed_action_ids (set[str]): Prevent duplicate action execution.
        created_at (datetime): timestamp.
        updated_at (datetime): timestamp.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str

    messages: list[Message] = Field(default_factory=list)

    user_email: Optional[str] = None

    pending_action: Optional[PendingAction] = None

    summary_offered: bool = False 
    escalated: bool = False

    send_count: int = 0
    processed_action_ids: set[str] = Field(default_factory=set)

    
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )