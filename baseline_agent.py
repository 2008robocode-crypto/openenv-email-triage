def baseline_policy(state):
    inbox = state["inbox"]

    for t in inbox:
        if not t["resolved"] and t["issue_type"] == "spam":
            return {"ticket_id": t["id"], "action": "mark_spam"}

    for t in inbox:
        if not t["resolved"] and t["customer_type"] == "vip":
            return {"ticket_id": t["id"], "action": "escalate"}

    for t in inbox:
        if not t["resolved"]:
            return {"ticket_id": t["id"], "action": "close"}

    return {"ticket_id": 1, "action": "reply"}