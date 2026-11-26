from pydantic import BaseModel , ConfigDict , Field
from datetime import datetime , timezone
from enum import StrEnum
from uuid import uuid4
from typing import Optional
from threading import RLock



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
    
    
    
_sessions : [str , SessionState] = {}
_lock = RLock


def create_sessoin(session_id : str , user_email : str | None) -> SessionState :
    """
    Args:
        session_id (str): identify the session.
        user_email (str | None): user email in initilization if given.

    Returns:
        SessionState: session model.

    Raises:
        ValueError: Issue in value of session id ( empty or exist , unvalid).
    """
    
    if not session_id.strip():
        raise ValueError("Session ID can't be empty")
    
    with _lock : 
        
        if session_id in _sessions :
            raise ValueError(f"Session already Exist {session_id}")
        
        new_session = SessionState(
            session_id=session_id,
            user_email=user_email
        )
        
        _sessions[session_id] = new_session
        
        return new_session.model_copy(deep=True)
    
    
def get_session(session_id : str)-> SessionState | None : 
    """
    Args:
        session_id (str): identify the session.

    Returns:
        SessionState | None: session model.
    """
    
    with _lock : 
        
        session = _sessions.get(session_id)
        
        if session is None : 
            return None
        
        return session.model_copy(deep=True)
    

def add_message(session_id : str , content : str , role : str)-> Message : 
    """
    Args:
        session_id (str).
        content (str).
        role (str).

    Returns:
        Message.

    Raises:
        ValueError.
        KeyError.
    """
    
    if not content.strip() :
        raise ValueError("Message Content can't be empty")
    
    new_message = Message(
        role=role,
        content=content
    )
    
    with _lock : 
        
        session = _sessions.get(session_id)
        
        if session is None : 
            raise KeyError("Session Not Found")
        
        session.messages.append(new_message)
        session.updated_at = datatime.now(timezone.utc)
        

    return new_message.model_copy(deep=True)


def get_messages(session_id : str)-> list[Message] :
    """
    Args:
        session_id (str).

    Returns:
        list[Message].

    Raises:
        KeyError.
    """
    
    if session_id is None : 
        raise KeyError("Sessions ID not exist")
    
    
    with _lock :
        
        session = _sessions.get(session_id)
        
        messages = sessions.messages
        
        return messages.model_copy(deep=True)