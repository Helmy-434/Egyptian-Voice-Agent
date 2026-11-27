from app.agent.tools.email import send_email
from app.agent.state import create_session, get_send_count , get_session

SESSION_ID = "email-test-session"

# Create session if it doesn't exist.
create_session(SESSION_ID, None)

result = send_email(
    recipient="abdelrahmangaber002@gmail.com",
    subject="Arabic AI Voice Agent - SMTP Test",
    content="Hello! This is a test email from our Arabic AI Voice Agent.",
    confirmed=True,
    send_count=get_send_count(SESSION_ID),
)

print(result)