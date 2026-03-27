# openenv-email-triage

# Customer Support Email Triage Environment

This project implements an OpenEnv-compatible reinforcement learning environment simulating real-world customer support ticket triage.

## Environment Description

Agents must process incoming customer support tickets and choose actions such as:

- reply
- close
- escalate
- mark_spam

The goal is to maximize cumulative reward by correctly prioritizing urgent issues, escalating VIP customers, and filtering spam.

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