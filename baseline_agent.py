def baseline_policy(state):

    inbox = state["inbox"]

    # VIP workflow first
    for t in inbox:
        if not t.resolved and t.customer_type == "vip":
            return {"ticket_id": t.id, "action": "escalate"}

    # high urgency next
    for t in inbox:
        if not t.resolved and t.urgency >= 4:
            return {"ticket_id": t.id, "action": "close"}

    # spam cleanup
    for t in inbox:
        if not t.resolved and t.issue_type == "spam":
            return {"ticket_id": t.id, "action": "mark_spam"}

    # low urgency close
    for t in inbox:
        if not t.resolved:
            return {"ticket_id": t.id, "action": "close"}