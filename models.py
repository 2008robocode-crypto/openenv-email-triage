from pydantic import BaseModel

class Ticket(BaseModel):
    id: int
    subject: str
    message: str
    issue_type: str
    urgency: int
    customer_type: str
    resolved: bool = False