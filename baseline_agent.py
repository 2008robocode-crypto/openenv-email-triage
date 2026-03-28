def baseline_policy(state):

    inbox = state["inbox"]

    # STEP 1: Handle VIP properly (escalate → then close)
    for t in inbox:
        if not t.resolved and t.customer_type == "vip":
            return {"ticket_id": t.id, "action": "escalate"}

    for t in inbox:
        if not t.resolved and t.customer_type == "vip":
            return {"ticket_id": t.id, "action": "close"}

    # STEP 2: Remove spam early
    for t in inbox:
        if not t.resolved and t.issue_type == "spam":
            return {"ticket_id": t.id, "action": "mark_spam"}

    # STEP 3: High urgency tickets
    for t in inbox:
        if not t.resolved and t.urgency >= 4:
            return {"ticket_id": t.id, "action": "close"}

    # STEP 4: Queries → reply first
    for t in inbox:
        if not t.resolved and t.issue_type == "query":
            return {"ticket_id": t.id, "action": "reply"}

    # STEP 5: Close everything else
    for t in inbox:
        if not t.resolved:
            return {"ticket_id": t.id, "action": "close"}