# High-Level Usage Flow (User • CLI • Agent)

This slide-friendly overview shows how a person uses the BDI framework from the CLI or a Python script. No code knowledge required.

---
## 1. User Starts the Agent
The user launches the CLI or runs a script with a goal like: "Summarize repo changes and draft a release note."

Action: Runs a command or script entry point (for example, `start`).

---
## 2. Runner Initializes the BDI Core
The CLI or script:
1. Validates the input goal.
2. Creates an agent session and loads configuration.
3. Converts the goal into a new desire for the BDI core.

At this point, the BDI core begins internal planning (details abstracted for users).

---
## 3. BDI Core Generates and Runs a Plan
Inside the core (not visible to the user):
- The goal is turned into a plan with steps.
- Steps start executing one by one (research, analyze, draft, refine, etc.).
- Progress events are emitted (status changes, partial outputs, errors, completion notes).

---
## 4. Progress Updates in the CLI
The CLI displays lightweight updates:
- Current status (Working / Waiting / Complete).
- Latest message and intermediate results.
- Optional logs for deeper inspection.

---
## 5. Completion and Output
When the plan finishes:
- The BDI core emits a completion event with final output (summary text, structured data, etc.).
- The CLI prints the final report and can optionally save it to disk.

---
## 6. (Optional) Session History
Run logs can be stored for later review, making it easy to inspect past goals and outputs.

---
## 7. Key Value for Users
- Natural language in -> actionable structured work out.
- Real-time progress visibility builds trust.
- Final deliverable is clean and shareable.

---
## Mermaid Diagram (Slide Friendly)
```mermaid
graph TD
   U[User<br/>Types Goal] --> CLI[CLI or Script<br/>Start Session]
   CLI --> BDI[BDI Core<br/>Plan and Execute]
   BDI --> P[(Progress Events)]
   P --> CLI2[CLI Updates Output]
   BDI --> C[Completion Event]
   C --> CLI3[CLI Prints Final Report]
   CLI3 --> U2[User Reviews / Saves]

   style U fill:#e1f5fe
   style CLI fill:#fff3e0
   style BDI fill:#f3e5f5
   style CLI2 fill:#fff3e0
   style CLI3 fill:#fff3e0
   style U2 fill:#e1f5fe
```

---
## Simplified Sequence (Narrative)
1. User enters a goal and runs the CLI or script.
2. The runner registers the session and passes the goal to the BDI core.
3. BDI core plans and executes internally.
4. CLI streams progress updates.
5. BDI finishes -> final result returned.
6. User reviews, saves, or starts another.
