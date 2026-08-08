# voluntas — AI Context Map

> **Stack:** raw-http | none | unknown | python

> 0 routes | 0 models | 0 components | 47 lib files | 5 env vars | 1 middleware | 0% test coverage
> **Token savings:** this file is ~4,000 tokens. Without it, AI exploration would cost ~23,700 tokens. **Saves ~19,700 tokens per conversation.**
> **Last scanned:** 2026-06-04 02:00 — re-run after significant changes

---

# Libraries

- `voluntas/agent.py` — class BDI
- `voluntas/belief_updates.py`
  - function update_beliefs_from_desire_extraction: (agent, beliefs) -> BeliefStats
  - function update_beliefs_from_step_extraction: (agent, beliefs, Any]], source) -> BeliefStats
- `voluntas/cycle.py` — function is_final_cycle_status: (status) -> bool, function bdi_cycle: (agent) -> str
- `voluntas/errors.py` — function is_validation_output_error: (error) -> bool
- `voluntas/execution.py`
  - function extract_relevant_beliefs_from_result: (agent, step, result, step_success) -> List[Dict[str, Any]]
  - function analyze_step_outcome_and_update_beliefs: (agent, step, result, extracted_beliefs_out, Any]]]) -> bool
  - function execute_intentions: (agent) -> Dict
  - class StepRetryContext
- `voluntas/logging.py`
  - function configure_terminal_output_mirror: (log_file_path, *, strip_ansi) -> None
  - function disable_terminal_output_mirror: () -> None
  - function build_structured_run_log_entry: (user_prompt, result) -> dict[str, Any]
  - function format_beliefs_for_context: (agent) -> str
  - function log_states: (agent, types, "desires", "intentions"]], message) -> None
- `voluntas/monitoring.py` — function generate_history_context: (intention, max_history, include_details) -> str, function reconsider_current_intention: (agent) -> None
- `voluntas/planning.py` — function generate_intentions_from_desires: (agent) -> None
- `voluntas/prompts.py`
  - function build_initial_belief_extraction_prompt: (desires_text) -> str
  - function build_planning_stage1_prompt: (desires_text, beliefs_text, intention_guidance_text) -> str
  - function build_step_belief_extraction_prompt: (step_description, step_result, step_success, current_beliefs) -> str
  - function build_step_assessment_prompt: (step_description, result_output, step_type, history_context) -> str
  - function build_desire_satisfaction_prompt: (desire_id, desire_description, completed_intention_description, completed_intention_history, current_beliefs, remaining_intentions_text) -> str
  - function build_tool_execution_prompt: (beliefs_context, retry_context, tool_name, tool_params, Any], is_retry) -> str
  - _...5 more_
- `voluntas/schemas/belief_schemas.py`
  - class Belief
  - class BeliefSet
  - class ExtractedBelief
  - class BeliefExtractionResult
  - class BeliefUpdateDecision
  - class BeliefNameResolutionDecision
  - _...2 more_
- `voluntas/schemas/desire_schemas.py`
  - function generate_desire_id: (description, timestamp) -> str
  - class DesireStatus
  - class Desire
- `voluntas/schemas/intention_schemas.py`
  - class Intention
  - class HighLevelIntention
  - class HighLevelIntentionList
- `voluntas/schemas/plan_schemas.py`
  - class PlanStatus
  - class PlanStep
  - class PlanStepHistory
  - class Plan
- `voluntas/schemas/reconsider_schemas.py`
  - class ReconsiderResult
  - class StepAssessmentResult
  - class DesireSatisfactionResult
- `voluntas/state_transitions.py`
  - function update_desire_status: (agent, desire_id, status, *, force) -> bool
  - function remove_intention: (agent, intention) -> RemovalResult
  - function all_desires_terminal: (agent) -> bool
  - function replan_desire_for_intention: (agent, intention, *, reason) -> None
  - function fail_desire_for_intention: (agent, intention, *, reason) -> None
  - function finalize_current_intention: (agent, intention, *, desire_status, force_status_update) -> None
  - _...2 more_
- `benchmarks/agents/base_agent.py`
  - function read_file_tool: (file_path) -> str
  - function write_file_tool: (file_path, content) -> str
  - function list_directory_tool: (path) -> str
  - function run_shell_command_tool: (command) -> str
  - function get_tools_by_names: (tool_names) -> List[Tool]
  - class Tool
  - _...2 more_
- `benchmarks/agents/bdi_agent.py` — class BDIBenchmarkAgent
- `benchmarks/agents/crewai_agent.py` — class CrewAIAgent
- `benchmarks/agents/langgraph_agent.py` — class LangGraphAgent
- `benchmarks/evaluation/runner.py` — function main: (), class BenchmarkRunner
- `benchmarks/experiments/base_experiment.py`
  - function collect_metrics: (framework, experiment_id)
  - class ExperimentMetrics
  - class MetricCollector
  - class BaseExperiment
- `benchmarks/experiments/voluntas/TEMPLATE.py`
  - function build_agent: (model, mcp_servers) -> BDI
  - function get_mcp_servers: (repo_path)
  - function run_agent: (agent, metric_collector)
- `benchmarks/experiments/voluntas/runner.py` — function main: (participant_path) -> None, function run_experiment: (participant_path, experiment_id, task_id, participant_id) -> Dict[str, Any]
- `benchmarks/experiments/voluntas/simple_file_read/experiment-1.py`
  - function build_agent: (model, mcp_servers) -> BDI
  - function get_mcp_servers: (repo_path)
  - function run_agent: (agent, metric_collector)
- `benchmarks/experiments/crewai/TEMPLATE.py` — function build_agent: (model), function run_agent: (crew, metric_collector)
- `benchmarks/experiments/crewai/pydantic_ai_llm.py` — class PydanticAILLM
- `benchmarks/experiments/crewai/runner.py` — function main: (participant_path) -> None, function run_experiment: (participant_path, experiment_id, task_id, participant_id) -> Dict[str, Any]
- `benchmarks/experiments/crewai/simple_file_read/experiment-1.py` — function build_agent: (model), function run_agent: (crew, metric_collector)
- `benchmarks/experiments/langgraph/TEMPLATE.py`
  - function build_agent: (model)
  - function run_agent: (agent, metric_collector) -> Dict
  - class AgentState
- `benchmarks/experiments/langgraph/runner.py` — function main: (participant_path) -> None, function run_experiment: (participant_path, experiment_id, task_id, participant_id) -> Dict[str, Any]
- `benchmarks/experiments/langgraph/simple_file_read/experiment-1.py`
  - function build_agent: (model)
  - function run_agent: (agent, metric_collector) -> Dict
  - class AgentState
- `benchmarks/experiments/run_experiments.py` — function main: (), class ExperimentRunner
- `benchmarks/metrics/collector.py` — class MetricsCollector, class BenchmarkMetricsAggregator
- `benchmarks/metrics/usage_tracker.py` — class UsageTracker
- `benchmarks/tasks/setup_teardown.py`
  - function ensure_readme_exists: ()
  - function cleanup_uppercase_file: ()
  - function cleanup_test_directory: ()
  - function cleanup_report_file: ()
  - function cleanup_dependencies_report: ()
  - function ensure_log_file_exists: ()
  - _...16 more_
- `benchmarks/tasks/task_schema.py`
  - class TaskCategory
  - class TaskDomain
  - class SuccessCriteria
  - class TaskDefinition
  - class TaskResult
  - class BenchmarkRun
- `benchmarks/tasks/validators.py`
  - function file_exists: (file_path, **kwargs) -> ValidationResult
  - function directory_exists: (path, **kwargs) -> ValidationResult
  - function file_was_read: (file_path, execution_log, **kwargs) -> ValidationResult
  - function correct_line_count: (file_path, tolerance, **kwargs) -> ValidationResult
  - function all_python_files_found: (directory, final_state, Any], **kwargs) -> ValidationResult
  - function no_false_positives: (extension, final_state, Any], **kwargs) -> ValidationResult
  - _...32 more_
- `benchmarks/usability/ease_of_use.py`
  - function create_standard_assessments: () -> EaseOfUseEvaluator
  - class UsabilityDimension
  - class UsabilityMetric
  - class UsabilityAssessment
  - class EaseOfUseEvaluator
  - function generate_pkce: () -> PKCEPair
  - function get_token_storage_path: () -> Path
  - function load_stored_tokens: () -> TokenData | None
  - function save_tokens: (tokens) -> None
  - function clear_stored_tokens: () -> None
  - function is_token_expired: (tokens, buffer_seconds) -> bool
  - _...10 more_
  - function normalize_model_name: (model_name) -> str
  - function create_model: (model_name, provider, usage_tracker, settings) -> CodexModel
  - class CodexModel
- `toy.py`
  - function discover_tasks: () -> list[Path]
  - function build_filesystem_server: (task_path) -> MCPServerStdio
  - function run_command: (task_path, command, timeout_seconds) -> str
  - function create_agent: (model, task_path, log_path) -> BDI
  - function run_task: (model, task_path, output_dir) -> None
  - function main: () -> None

---

# Config

## Environment Variables

- `MODEL` (has default) — .env
- `OLLAMA_BASE_URL` (has default) — .env
- `PROVIDER` (has default) — .env
- `PROVIDER_API_KEY` (has default) — .env

## Config Files

- `pyproject.toml`

---

# Middleware

## auth

---

# Dependency Graph

## Most Imported Files (change these carefully)

- `/constants.py` — imported by **4** files
- `/auth.py` — imported by **2** files
- `/provider.py` — imported by **2** files
- `/task_schema.py` — imported by **4** files
- `/model.py` — imported by **1** files
- `/base_agent.py` — imported by **1** files
- `/bdi_agent.py` — imported by **1** files
- `/langgraph_agent.py` — imported by **1** files
- `/crewai_agent.py` — imported by **1** files
- `/runner.py` — imported by **1** files
- `/base_experiment.py` — imported by **1** files
- `/collector.py` — imported by **1** files
- `/usage_tracker.py` — imported by **1** files
- `/simple_tasks.py` — imported by **1** files
- `/medium_tasks.py` — imported by **1** files
- `/complex_tasks.py` — imported by **1** files
- `/ease_of_use.py` — imported by **1** files

## Import Map (who imports what)

- `/task_schema.py` ← `benchmarks/tasks/__init__.py`, `benchmarks/tasks/complex_tasks.py`, `benchmarks/tasks/medium_tasks.py`, `benchmarks/tasks/simple_tasks.py`
- `/base_agent.py` ← `benchmarks/agents/__init__.py`
- `/bdi_agent.py` ← `benchmarks/agents/__init__.py`
- `/langgraph_agent.py` ← `benchmarks/agents/__init__.py`
- `/crewai_agent.py` ← `benchmarks/agents/__init__.py`

---

# Test Coverage

> **0%** of routes and models are covered by tests
> 13 test files found

---

_Generated by [codesight](https://github.com/Houseofmvps/codesight) — see your codebase clearly_
