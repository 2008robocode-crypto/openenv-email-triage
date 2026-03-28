from models import Ticket


class CustomerSupportEnv:

    def __init__(self, task_mode="easy"):
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
            "inbox": self.inbox,
            "step_count": self.step_count,
            "task_mode": self.task_mode
        }

    def find_ticket(self, ticket_id):
        for t in self.inbox:
            if t.id == ticket_id:
                return t
        return None

    def highest_urgency_unresolved(self):
        unresolved = [t for t in self.inbox if not t.resolved]
        if not unresolved:
            return None
        return max(unresolved, key=lambda x: x.urgency)
    
    

    def step(self, action_dict):

        reward = -1
        done = False
        info = {}

        ticket = self.find_ticket(action_dict["ticket_id"])



        if ticket is None:
            return self.state(), -10, False, {"error": "invalid_ticket"}

        action = action_dict["action"]

        # ---------- HARD WORKFLOW CHECK ----------
        # discourage idle / inefficient behavior
        if action == "reply" and ticket.issue_type != "query":
            reward -= 2


        if self.task_mode == "hard":
            if ticket.customer_type == "vip" and action == "close":
                if ticket.id not in self.escalated_vip:
                    reward -= 15  # BIG penalty
                    return self.state(), reward, False, {"error": "vip_not_escalated"}

        # ---------- BASE LOGIC ----------

        if action == "mark_spam" and ticket.issue_type == "spam":
            reward += 8
            ticket.resolved = True

        elif action == "escalate" and ticket.urgency >= 4:
            reward += 10
            if ticket.customer_type == "vip":
                self.escalated_vip.add(ticket.id)
            ticket.resolved = True

        elif action == "reply":
            reward += 3

        elif action == "close":
            reward += 5
            ticket.resolved = True

        else:
            reward -= 7

        # ---------- EASY BONUS ----------
        if self.task_mode == "easy":
            if ticket.issue_type == "spam" and ticket.resolved:
                reward += 5

        # ---------- MEDIUM BONUS ----------
        if self.task_mode == "medium":
            highest = self.highest_urgency_unresolved()
            if highest is not None:
                if ticket.id == highest.id and ticket.resolved:
                    reward += 6
                else:
                    reward -= 3

        # ---------- HARD BONUS ----------
        if self.task_mode == "hard":
            if ticket.customer_type == "vip" and ticket.resolved:
                if ticket.id in self.escalated_vip:
                    reward += 8

        self.step_count += 1

        if all(t.resolved for t in self.inbox):
            done = True

        return self.state(), reward, done, info