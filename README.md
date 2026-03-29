---
title: openenv-email-triage-v1
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# openenv-email-triage

# Customer Support Email Triage Environment

This project implements an OpenEnv-compatible reinforcement learning environment simulating real-world customer support ticket triage.

## Motivation

Modern customer support systems require agents to triage large volumes of incoming tickets under constraints such as urgency, customer priority, and workflow policies.

This environment simulates a realistic SaaS customer support pipeline where:

- VIP users require escalation workflows
- High urgency issues must be prioritized
- Spam must be filtered efficiently
- Queries require conversational handling

The goal is to provide a benchmark environment for evaluating decision-making agents in structured operational workflows.

## Why This Matters

This environment goes beyond simple classification tasks and introduces:

- Sequential decision constraints
- Workflow correctness requirements
- Multi-objective reward balancing

It is designed to test whether agents can follow real-world operational policies rather than just optimize isolated actions.

## Tasks

### Easy Task
Spam filtering — correctly mark spam tickets.

### Medium Task
Urgency prioritization — resolve high urgency tickets before low urgency ones.

### Hard Task
VIP workflow — escalate VIP complaints before resolving.

## Action Space

Dictionary:
- ticket_id (int)
- action (string)

## Observation Space

Dictionary:
- inbox (list of Ticket objects)
- step_count (int)

## Reward Function

Reward Design:

The environment uses a dense reward structure to guide agent behavior:

Base Rewards:
+8  Correct spam classification (mark_spam)
+10 Correct escalation of high-urgency tickets
+5  Correct ticket closure
+3  Reply action (used for query handling)

Penalties:
-7   Incorrect or invalid actions
-10  Invalid ticket selection
-2   Replying to non-query tickets
-15  Violating VIP workflow (closing without escalation)

Task-Specific Bonuses:
Easy:
+5 additional reward for correctly handling spam

Medium:
+6 for resolving highest urgency ticket first
-3 for incorrect prioritization

Hard:
+8 for correctly completing VIP escalation → closure workflow

The reward function provides both immediate and delayed feedback, encouraging agents to follow structured workflows rather than greedy actions.

## Setup

```bash
pip install -r requirements.txt
python run_eval.py