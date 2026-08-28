"""Pydantic models for structured orchestrator output."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SubTask(BaseModel):
    subtask_id: int = Field(description="Sequential ID starting from 1")
    description: str = Field(description="Clear, actionable description of the subtask")
    reasoning: str = Field(description="Why this subtask was created from the original task")


class SelectedAgent(BaseModel):
    agent_name: str = Field(description="Exact name from the agent registry")
    agent_file: str = Field(description="Filename in agents/ directory, e.g. 'ml-research-scientist.json'")
    match_reason: str = Field(description="Why this agent is the best fit for the subtask")


class TaskAssignment(BaseModel):
    subtask: SubTask
    selected_agent: SelectedAgent


class OrchestratorOutput(BaseModel):
    original_task: str = Field(description="The original user task, verbatim")
    plan_summary: str = Field(description="Brief summary of how the task was decomposed")
    assignments: list[TaskAssignment] = Field(description="Ordered list of subtask-to-agent assignments")
