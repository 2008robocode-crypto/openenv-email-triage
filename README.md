---
title: openenv-email-triage
emoji: 📧
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
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

Dense reward structure:
- +7 spam correctly filtered
- +9 VIP escalation
- +4 correct closure
- negative reward for incorrect action

## Setup

```bash
pip install -r requirements.txt
python run_eval.py