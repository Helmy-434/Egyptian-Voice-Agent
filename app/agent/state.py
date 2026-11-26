from pydantic import BaseModel , ConfigDict , Field
from datetime import datetime , timezone
from enum import StrEnum
from uuid import uuid4
from typing import Optional
from threading import RLock
from typing import Literal


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
_lock = RLock()


def create_sessoin(session_id : str , user_email : str | None) -> SessionState :

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

    if not session_id.strip() : 
        raise ValueError("Session ID can't be Empty")
    
    with _lock : 
        
        session = _sessions.get(session_id)
        
        if session is None : 
            return None
        
        return session.model_copy(deep=True)
    

def add_message(session_id : str , content : str , role: Literal["system", "user", "assistant", "tool"] )-> Message : 

    if not session_id.strip() : 
        raise ValueError("Session ID can't be Empty")
    
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

    if not session_id.strip() : 
        raise ValueError("Session ID can't be Empty")
    
    
    with _lock :
        
        session = _sessions.get(session_id)
        
        if session is None : 
            raise KeyError("Session Not Found!")
        
        messages = sessions.messages
        
        return messages.model_copy(deep=True)
    
    
    
def set_pending_action(session_id : str , tool : str , recipient : str , subject : str , content : str , summary : str)-> PendingAction :
    
    if not session_id.strip() : 
        raise ValueError("Session ID can't be Empty")
    
    
    action = PendingAction(
        tool=tool,
        recipient=recipient,
        subject=subject,
        content=content,
        summary=summary,
    )

    with _lock:
        session = _sessions.get(session_id)

        if session is None:
            raise KeyError(f"Session not found: {session_id}")

        if session.pending_action is not None:
            
            active_statuses = {
                ActionStatus.PENDING_CONFIRMATION,
                ActionStatus.CONFIRMED,
                ActionStatus.EXECUTING,
            }

            if session.pending_action.status in active_statuses:
                raise ValueError(
                    "An active pending action already exists"
                )

        session.pending_action = action
        session.updated_at = datetime.now(timezone.utc)

    return action.model_copy(deep=True)



def get_pending_action(session_id : str)-> PendingAction | None :
    
    if not session_id.strip() : 
        raise ValueError("Session ID can't be Empty")
    
    with _lock : 
        
        session = _sessions.get(session_id)
        
        if session is None : 
            raise KeyError("Session Not Found!")
        
        if session.pending_action is None : 
            return None
        
        return session.pending_action.model_copy(deep=True)
    
    
    
def confirm_pending_action(session_id : str , action_id : str)->PendingAction :
    
    if session_id is None : 
        raise ValueError("Session ID can't be Empty!")
    
    if action_id is None :
        raise ValueError("Actino ID can't be Empty!")
    
    
    with _lock : 
        
        session = _sessions[session_id]
        
        if session is None : 
            raise KeyError("Session Not Found!")
        
        action = session.pending_action
        
        if action.action_id != action_id:
            raise ValueError("No matching pending action")

        if action.status != ActionStatus.PENDING_CONFIRMATION:
            raise ValueError("Action is not awaiting confirmation")
        
        action.status = ActionStatus.CONFIRMED
        action.confirmed = True
        session.updated_at = datetime.now(timezone.utc)
        
        return action.model_copy(deep=True)
    
    
def cancel_pending_action(session_id: str, action_id: str) -> PendingAction:

    with _lock:
        session = _sessions.get(session_id)

        if session is None:
            raise KeyError(f"Session not found: {session_id}")

        action = session.pending_action

        if action is None or action.action_id != action_id:
            raise ValueError("No matching pending action")

        if action.status in {
            ActionStatus.EXECUTING,
            ActionStatus.SENT,
        }:
            raise ValueError("Cannot cancel an action that has started")

        action.confirmed = False
        action.status = ActionStatus.CANCELLED
        session.updated_at = datetime.now(timezone.utc)

        return action.model_copy(deep=True)
    
    
def claim_confirmed_action(session_id: str,action_id: str,max_sends: int) -> PendingAction:

    with _lock:
        session = _sessions.get(session_id)

        if session is None:
            raise KeyError(f"Session not found: {session_id}")

        action = session.pending_action

        if action is None or action.action_id != action_id:
            raise ValueError("No matching pending action")

        if not action.confirmed:
            raise PermissionError("Action has not been confirmed")

        if action.status != ActionStatus.CONFIRMED:
            raise ValueError("Action is not available for execution")

        if action_id in session.processed_action_ids:
            raise ValueError("Action has already been processed")

        if session.send_count >= max_sends:
            raise PermissionError("Session send limit reached")

        action.status = ActionStatus.EXECUTING

        session.processed_action_ids.add(action_id)

        session.updated_at = datetime.now(timezone.utc)

        return action.model_copy(deep=True)
    
    
    
def mark_action_sent(session_id: str, action_id: str) -> PendingAction:

    with _lock:
        session = _sessions.get(session_id)

        if session is None:
            raise KeyError(f"Session not found: {session_id}")

        action = session.pending_action

        if action is None or action.action_id != action_id:
            raise ValueError("No matching pending action")

        if action.status != ActionStatus.EXECUTING:
            raise ValueError("Action is not executing")

        action.status = ActionStatus.SENT
        session.send_count += 1
        session.updated_at = datetime.now(timezone.utc)

        return action.model_copy(deep=True)
    
    
def mark_action_failed(session_id: str,action_id: str) -> PendingAction:

    with _lock:
        session = _sessions.get(session_id)

        if session is None:
            raise KeyError(f"Session not found: {session_id}")

        action = session.pending_action

        if action is None or action.action_id != action_id:
            raise ValueError("No matching pending action")

        if action.status != ActionStatus.EXECUTING:
            raise ValueError("Action is not executing")

        action.status = ActionStatus.FAILED
        action.confirmed = False
        session.updated_at = datetime.now(timezone.utc)

        return action.model_copy(deep=True)
    
    

def mark_summary_offered(session_id: str) -> bool:

    with _lock:
        session = _sessions.get(session_id)

        if session is None:
            raise KeyError(f"Session not found: {session_id}")

        if session.summary_offered:
            return False

        session.summary_offered = True
        session.updated_at = datetime.now(timezone.utc)

        return True
    
    
def mark_escalated(session_id: str) -> bool:

    with _lock:
        session = _sessions.get(session_id)

        if session is None:
            raise KeyError(f"Session not found: {session_id}")

        if session.escalated:
            return False

        session.escalated = True
        session.updated_at = datetime.now(timezone.utc)

        return True
    
def get_send_count(session_id: str) -> int:

    with _lock:
        session = _sessions.get(session_id)

        if session is None:
            raise KeyError(f"Session not found: {session_id}")

        return session.send_count

def close_session(session_id: str) -> bool:

    with _lock:
        session = _sessions.pop(session_id, None)

        return session is not None0