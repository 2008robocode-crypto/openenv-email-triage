from models import Ticket

class CustomerSupportEnv:

    def __init__(self, task_mode="hard"):
        self.task_mode = task_mode
        self.reset()

    def reset(self):
        self.inbox = [
            Ticket(id=1, subject="Refund not received", message="I want money back",
                   issue_type="refund", urgency=4, customer_type="normal"),

            Ticket(id=2, subject="App crash", message="Very angry",
                   issue_type="complaint", urgency=5, customer_type="vip"),

            Ticket(id=3, subject="Discount offer", message="Buy now",
                   issue_type="spam", urgency=1, customer_type="normal"),

            Ticket(id=4, subject="Feature question", message="How to export?",
                   issue_type="query", urgency=2, customer_type="normal"),
        ]

        self.escalated_vip = set()
        self.step_count = 0
        return self.state()

    def state(self):
        return {
            "inbox": [t.dict() for t in self.inbox],
            "step_count": self.step_count,
            "task_mode": self.task_mode
        }

    def find_ticket(self, ticket_id):
        return next((t for t in self.inbox if t.id == ticket_id), None)

    def highest_urgency_unresolved(self):
        unresolved = [t for t in self.inbox if not t.resolved]
        return max(unresolved, key=lambda x: x.urgency) if unresolved else None

    def step(self, action_dict):
        reward = -1
        done = False
        info = {}

        ticket = self.find_ticket(action_dict["ticket_id"])
        if ticket is None:
            return self.state(), -10, False, {"error": "invalid_ticket"}

        action = action_dict["action"]

        # HARD constraint
        if self.task_mode == "hard":
            if ticket.customer_type == "vip" and action == "close":
                if ticket.id not in self.escalated_vip:
                    return self.state(), -15, False, {"error": "vip_not_escalated"}

        if action == "mark_spam" and ticket.issue_type == "spam":
            reward += 8
            ticket.resolved = True

        elif action == "escalate" and ticket.urgency >= 4:
            reward += 10
            if ticket.customer_type == "vip":
                self.escalated_vip.add(ticket.id)

        elif action == "reply":
            reward += 3
            if ticket.issue_type != "query":
                reward -= 2

        elif action == "close":
            reward += 5
            ticket.resolved = True

        else:
            reward -= 7

        # bonuses
        if self.task_mode == "easy":
            if ticket.issue_type == "spam" and ticket.resolved:
                reward += 5

        if self.task_mode == "medium":
            highest = self.highest_urgency_unresolved()
            if highest and ticket.id == highest.id and ticket.resolved:
                reward += 6

        if self.task_mode == "hard":
            if ticket.customer_type == "vip" and ticket.resolved:
                if ticket.id in self.escalated_vip:
                    reward += 8

        self.step_count += 1

        done = all(t.resolved for t in self.inbox) or self.step_count > 20

        return self.state(), reward, done, info