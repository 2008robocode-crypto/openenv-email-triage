from pydantic import BaseModel


class Ticket(BaseModel):
    id: int
    subject: str
    message: str

    issue_type: str        # refund / complaint / query / spam
    urgency: int           # 1 to 5
    customer_type: str     # normal / vip

    resolved: bool = False