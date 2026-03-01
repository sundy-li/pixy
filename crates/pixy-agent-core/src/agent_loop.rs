use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use pixy_ai::{
    AssistantContentBlock, AssistantMessage, AssistantMessageEvent, Context, EventStream, Message,
    PiAiError, PiAiErrorCode, StopReason, ToolResultContentBlock,
};
use serde_json::{json, Value};
use tracing::{debug, warn};

use crate::types::{
    AgentAbortSignal, AgentContext, AgentEvent, AgentLoopConfig, AgentMessage, AgentRunMetrics,
    AgentTool, AgentToolResult, MessageQueueFn,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentLoopError {
    EmptyContext,
    CannotContinueFromAssistant,
}

impl AgentLoopError {
    pub fn message(self) -> &'static str {
        match self {
            AgentLoopError::EmptyContext => "Cannot continue: no messages in context",
            AgentLoopError::CannotContinueFromAssistant => {
                "Cannot continue from message role: assistant"
            }
        }
    }
}

impl std::fmt::Display for AgentLoopError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for AgentLoopError {}

pub fn agent_loop(
    prompts: Vec<AgentMessage>,
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Option<AgentAbortSignal>,
) -> EventStream<AgentEvent, Vec<AgentMessage>> {
    let stream = EventStream::new(|event: &AgentEvent| match event {
        AgentEvent::AgentEnd { messages } => Some(messages.clone()),
        _ => None,
    });

    let task_stream = stream.clone();
    tokio::spawn(async move {
        run_loop(prompts, context, config, signal, task_stream).await;
    });

    stream
}

pub fn try_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Option<AgentAbortSignal>,
) -> Result<EventStream<AgentEvent, Vec<AgentMessage>>, AgentLoopError> {
    validate_continue_context(&context)?;
    Ok(agent_loop(vec![], context, config, signal))
}

pub fn agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Option<AgentAbortSignal>,
) -> EventStream<AgentEvent, Vec<AgentMessage>> {
    match try_agent_loop_continue(context, config, signal) {
        Ok(stream) => stream,
        Err(error) => panic!("{}", error.message()),
    }
}

fn validate_continue_context(context: &AgentContext) -> Result<(), AgentLoopError> {
    if context.messages.is_empty() {
        return Err(AgentLoopError::EmptyContext);
    }
    if matches!(context.messages.last(), Some(Message::Assistant { .. })) {
        return Err(AgentLoopError::CannotContinueFromAssistant);
    }
    Ok(())
}

async fn run_loop(
    prompts: Vec<AgentMessage>,
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Option<AgentAbortSignal>,
    stream: EventStream<AgentEvent, Vec<AgentMessage>>,
) {
    let mut runner = AgentLoopRunner::new(context, config, signal, stream);
    runner.run(prompts).await;
}

struct AgentLoopRunner {
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Option<AgentAbortSignal>,
    stream: EventStream<AgentEvent, Vec<AgentMessage>>,
    new_messages: Vec<AgentMessage>,
    metrics: AgentRunMetrics,
    pending_messages: Vec<AgentMessage>,
    first_assistant_turn: bool,
}

impl AgentLoopRunner {
    fn new(
        context: AgentContext,
        config: AgentLoopConfig,
        signal: Option<AgentAbortSignal>,
        stream: EventStream<AgentEvent, Vec<AgentMessage>>,
    ) -> Self {
        let pending_messages = pull_queue_messages(config.get_steering_messages.as_ref());
        Self {
            context,
            config,
            signal,
            stream,
            new_messages: Vec::new(),
            metrics: AgentRunMetrics::default(),
            pending_messages,
            first_assistant_turn: true,
        }
    }

    async fn run(&mut self, prompts: Vec<AgentMessage>) {
        self.initialize(prompts);

        loop {
            let mut has_more_tool_calls = true;
            let mut steering_after_tools: Option<Vec<AgentMessage>> = None;

            while has_more_tool_calls || !self.pending_messages.is_empty() {
                self.start_turn_if_needed();
                self.flush_pending_messages();

                if self.end_turn_on_abort() {
                    return;
                }

                let assistant_outcome = self.request_assistant_response().await;
                self.record_assistant_metrics(&assistant_outcome);
                let assistant_message = assistant_outcome.message;

                self.new_messages.push(assistant_message.clone());
                if is_error_or_aborted(&assistant_message) {
                    self.stream.push(AgentEvent::TurnEnd {
                        message: assistant_message,
                        tool_results: vec![],
                    });
                    self.finish();
                    return;
                }

                let mut tool_results = Vec::new();
                let mut aborted_during_tools = false;
                has_more_tool_calls = !extract_tool_calls(&assistant_message).is_empty();

                if has_more_tool_calls {
                    let outcome = execute_tool_calls(
                        &self.context.tools,
                        &assistant_message,
                        &self.stream,
                        self.signal.as_ref(),
                        self.config.get_steering_messages.as_ref(),
                    )
                    .await;
                    self.record_tool_metrics(&outcome);
                    tool_results = outcome.tool_results;
                    steering_after_tools = outcome.steering_messages;
                    aborted_during_tools = outcome.aborted;
                }

                self.append_tool_results(&tool_results);
                self.stream.push(AgentEvent::TurnEnd {
                    message: assistant_message,
                    tool_results,
                });

                if aborted_during_tools {
                    self.finish();
                    return;
                }

                self.pending_messages =
                    self.resolve_next_pending_messages(steering_after_tools.take());
            }

            let follow_up_messages =
                pull_queue_messages(self.config.get_follow_up_messages.as_ref());
            if follow_up_messages.is_empty() {
                break;
            }
            self.pending_messages = follow_up_messages;
        }

        self.finish();
    }

    fn initialize(&mut self, prompts: Vec<AgentMessage>) {
        self.new_messages = prompts.clone();
        self.stream.push(AgentEvent::AgentStart);
        self.stream.push(AgentEvent::TurnStart);

        for prompt in prompts {
            self.stream.push(AgentEvent::MessageStart {
                message: prompt.clone(),
            });
            self.stream.push(AgentEvent::MessageEnd {
                message: prompt.clone(),
            });
            self.context.messages.push(prompt);
        }
    }

    fn start_turn_if_needed(&mut self) {
        if self.first_assistant_turn {
            self.first_assistant_turn = false;
        } else {
            self.stream.push(AgentEvent::TurnStart);
        }
    }

    fn flush_pending_messages(&mut self) {
        for message in std::mem::take(&mut self.pending_messages) {
            self.stream.push(AgentEvent::MessageStart {
                message: message.clone(),
            });
            self.stream.push(AgentEvent::MessageEnd {
                message: message.clone(),
            });
            self.context.messages.push(message.clone());
            self.new_messages.push(message);
        }
    }

    fn end_turn_on_abort(&mut self) -> bool {
        if !is_aborted(self.signal.as_ref()) {
            return false;
        }

        let (api, provider, model) = self.primary_model_identity();
        let message = aborted_assistant_message(api, provider, model);
        self.push_terminal_message(message.clone());
        self.stream.push(AgentEvent::TurnEnd {
            message,
            tool_results: vec![],
        });
        self.finish();
        true
    }

    async fn request_assistant_response(&mut self) -> AssistantResponseOutcome {
        match stream_assistant_response(
            &mut self.context,
            &self.config,
            &self.stream,
            self.signal.as_ref(),
        )
        .await
        {
            Ok(outcome) => outcome,
            Err(error) => {
                let (api, provider, model) = self.primary_model_identity();
                let message =
                    error_assistant_message(api, provider, model, error.as_compact_json());
                self.push_terminal_message(message.clone());
                AssistantResponseOutcome {
                    message,
                    duration_ms: 0,
                    retries: 0,
                }
            }
        }
    }

    fn record_assistant_metrics(&mut self, outcome: &AssistantResponseOutcome) {
        self.metrics.assistant_request_count =
            self.metrics.assistant_request_count.saturating_add(1);
        self.metrics.assistant_request_total_ms = self
            .metrics
            .assistant_request_total_ms
            .saturating_add(outcome.duration_ms);
        self.metrics.retry_count = self.metrics.retry_count.saturating_add(outcome.retries);
    }

    fn record_tool_metrics(&mut self, outcome: &ToolExecutionOutcome) {
        self.metrics.tool_execution_count = self
            .metrics
            .tool_execution_count
            .saturating_add(outcome.executed_count);
        self.metrics.tool_execution_total_ms = self
            .metrics
            .tool_execution_total_ms
            .saturating_add(outcome.executed_total_duration_ms);
    }

    fn append_tool_results(&mut self, tool_results: &[AgentMessage]) {
        for message in tool_results {
            self.context.messages.push(message.clone());
            self.new_messages.push(message.clone());
        }
    }

    fn resolve_next_pending_messages(
        &self,
        steering_after_tools: Option<Vec<AgentMessage>>,
    ) -> Vec<AgentMessage> {
        match steering_after_tools {
            Some(messages) if !messages.is_empty() => messages,
            _ => pull_queue_messages(self.config.get_steering_messages.as_ref()),
        }
    }

    fn push_terminal_message(&mut self, message: AgentMessage) {
        self.stream.push(AgentEvent::MessageStart {
            message: message.clone(),
        });
        self.stream.push(AgentEvent::MessageEnd {
            message: message.clone(),
        });
        self.context.messages.push(message.clone());
        self.new_messages.push(message);
    }

    fn finish(&mut self) {
        emit_metrics_event(&self.stream, &self.metrics);
        let messages = std::mem::take(&mut self.new_messages);
        self.stream.push(AgentEvent::AgentEnd {
            messages: messages.clone(),
        });
        self.stream.end(Some(messages));
    }

    fn primary_model_identity(&self) -> (String, String, String) {
        (
            self.config.model.api.clone(),
            self.config.model.provider.clone(),
            self.config.model.id.clone(),
        )
    }
}

struct AssistantResponseOutcome {
    message: AgentMessage,
    duration_ms: u64,
    retries: usize,
}

async fn stream_assistant_response(
    context: &mut AgentContext,
    config: &AgentLoopConfig,
    stream: &EventStream<AgentEvent, Vec<AgentMessage>>,
    signal: Option<&AgentAbortSignal>,
) -> Result<AssistantResponseOutcome, PiAiError> {
    AssistantRequestRunner::new(context, config, stream, signal)
        .run()
        .await
}

struct AssistantRequestRunner<'a> {
    context: &'a mut AgentContext,
    config: &'a AgentLoopConfig,
    stream: &'a EventStream<AgentEvent, Vec<AgentMessage>>,
    signal: Option<&'a AgentAbortSignal>,
    models: Vec<pixy_ai::Model>,
    max_attempts: usize,
    started_at: Instant,
}

impl<'a> AssistantRequestRunner<'a> {
    fn new(
        context: &'a mut AgentContext,
        config: &'a AgentLoopConfig,
        stream: &'a EventStream<AgentEvent, Vec<AgentMessage>>,
        signal: Option<&'a AgentAbortSignal>,
    ) -> Self {
        Self {
            context,
            config,
            stream,
            signal,
            models: attempt_models(config),
            max_attempts: config.retry.max_attempts.max(1),
            started_at: Instant::now(),
        }
    }

    async fn run(self) -> Result<AssistantResponseOutcome, PiAiError> {
        let mut attempt = 1usize;
        loop {
            let active_model = self.model_for_attempt(attempt).clone();
            match stream_assistant_response_once(
                self.context,
                self.config,
                self.stream,
                self.signal,
                &active_model,
            )
            .await
            {
                Ok(message) => return Ok(self.success_outcome(message, &active_model, attempt)),
                Err(error) => {
                    if let Some(outcome) = self
                        .handle_attempt_failure(&active_model, attempt, &error)
                        .await
                    {
                        return outcome;
                    }
                }
            }
            attempt = attempt.saturating_add(1);
        }
    }

    fn model_for_attempt(&self, attempt: usize) -> &pixy_ai::Model {
        let model_index = (attempt - 1).min(self.models.len().saturating_sub(1));
        &self.models[model_index]
    }

    fn success_outcome(
        &self,
        message: AgentMessage,
        active_model: &pixy_ai::Model,
        attempt: usize,
    ) -> AssistantResponseOutcome {
        let retries = attempt.saturating_sub(1);
        let duration_ms = self.elapsed_ms();
        debug!(
            provider = active_model.provider.as_str(),
            model = active_model.id.as_str(),
            attempts = attempt,
            retries,
            duration_ms,
            "assistant response completed"
        );
        AssistantResponseOutcome {
            message,
            duration_ms,
            retries,
        }
    }

    async fn handle_attempt_failure(
        &self,
        active_model: &pixy_ai::Model,
        attempt: usize,
        error: &PiAiError,
    ) -> Option<Result<AssistantResponseOutcome, PiAiError>> {
        warn!(
            provider = active_model.provider.as_str(),
            model = active_model.id.as_str(),
            attempt,
            max_attempts = self.max_attempts,
            error_code = ?error.code,
            error = error.message.as_str(),
            "assistant response attempt failed"
        );

        if attempt >= self.max_attempts {
            return Some(Err(error.clone()));
        }

        self.emit_model_fallback_event(active_model, attempt);

        let delay_ms = retry_delay_ms(&self.config.retry, attempt);
        warn!(
            provider = active_model.provider.as_str(),
            model = active_model.id.as_str(),
            attempt,
            max_attempts = self.max_attempts,
            delay_ms,
            "scheduling retry after assistant response failure"
        );
        self.stream.push(AgentEvent::RetryScheduled {
            attempt,
            max_attempts: self.max_attempts,
            delay_ms,
            error: error.as_compact_json(),
        });

        if let Some(outcome) = self
            .wait_retry_backoff_or_abort(active_model, attempt, delay_ms)
            .await
        {
            return Some(Ok(outcome));
        }

        None
    }

    fn emit_model_fallback_event(&self, active_model: &pixy_ai::Model, attempt: usize) {
        let next_model = self.model_for_attempt(attempt.saturating_add(1));
        if next_model.provider == active_model.provider && next_model.id == active_model.id {
            return;
        }

        warn!(
            from_provider = active_model.provider.as_str(),
            from_model = active_model.id.as_str(),
            to_provider = next_model.provider.as_str(),
            to_model = next_model.id.as_str(),
            attempt,
            "switching to fallback model for next retry"
        );
        self.stream.push(AgentEvent::ModelFallback {
            from_provider: active_model.provider.clone(),
            from_model: active_model.id.clone(),
            to_provider: next_model.provider.clone(),
            to_model: next_model.id.clone(),
        });
    }

    async fn wait_retry_backoff_or_abort(
        &self,
        active_model: &pixy_ai::Model,
        attempt: usize,
        delay_ms: u64,
    ) -> Option<AssistantResponseOutcome> {
        if delay_ms == 0 {
            return None;
        }

        if let Some(signal_ref) = self.signal {
            tokio::select! {
                _ = signal_ref.cancelled() => {
                    return Some(AssistantResponseOutcome {
                        message: aborted_assistant_message(
                            active_model.api.clone(),
                            active_model.provider.clone(),
                            active_model.id.clone(),
                        ),
                        duration_ms: self.elapsed_ms(),
                        retries: attempt.saturating_sub(1),
                    });
                }
                _ = tokio::time::sleep(Duration::from_millis(delay_ms)) => {}
            }
        } else {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        }

        None
    }

    fn elapsed_ms(&self) -> u64 {
        self.started_at.elapsed().as_millis() as u64
    }
}

async fn stream_assistant_response_once(
    context: &mut AgentContext,
    config: &AgentLoopConfig,
    stream: &EventStream<AgentEvent, Vec<AgentMessage>>,
    signal: Option<&AgentAbortSignal>,
    model: &pixy_ai::Model,
) -> Result<AgentMessage, PiAiError> {
    let llm_context = build_llm_context(context, config);

    let stream_fn = config.stream_fn.clone();
    let stream_model = model.clone();
    let response = stream_fn.stream(stream_model, llm_context, None)?;

    let mut state = AssistantStreamState::default();

    loop {
        let next_event = if let Some(signal_ref) = signal {
            tokio::select! {
                _ = signal_ref.cancelled() => None,
                event = response.next() => event,
            }
        } else {
            response.next().await
        };

        if next_event.is_none() && is_aborted(signal) {
            return Ok(state.finalize_aborted(context, stream, model));
        }

        let Some(event) = next_event else {
            break;
        };

        match &event {
            AssistantMessageEvent::Start { partial } => {
                state.handle_start(context, stream, partial.clone());
            }
            AssistantMessageEvent::TextStart { partial, .. }
            | AssistantMessageEvent::TextDelta { partial, .. }
            | AssistantMessageEvent::TextEnd { partial, .. }
            | AssistantMessageEvent::ThinkingStart { partial, .. }
            | AssistantMessageEvent::ThinkingDelta { partial, .. }
            | AssistantMessageEvent::ThinkingEnd { partial, .. }
            | AssistantMessageEvent::ToolcallStart { partial, .. }
            | AssistantMessageEvent::ToolcallDelta { partial, .. }
            | AssistantMessageEvent::ToolcallEnd { partial, .. } => {
                state.handle_update(context, stream, event.clone(), partial.clone());
            }
            AssistantMessageEvent::Done { message, .. } => {
                let final_message = to_agent_assistant_message(message.clone());
                return Ok(state.finalize_message(context, stream, final_message));
            }
            AssistantMessageEvent::Error { error, .. } => {
                let final_message = to_agent_assistant_message(error.clone());
                return Ok(state.finalize_message(context, stream, final_message));
            }
        }
    }

    if is_aborted(signal) {
        return Ok(state.finalize_aborted(context, stream, model));
    }

    state.last_message_or_error()
}

fn build_llm_context(context: &AgentContext, config: &AgentLoopConfig) -> Context {
    let llm_messages = config.convert_to_llm.convert(context.messages.clone());
    let llm_tools = if context.tools.is_empty() {
        None
    } else {
        Some(
            context
                .tools
                .iter()
                .map(AgentTool::to_llm_tool)
                .collect::<Vec<_>>(),
        )
    };

    Context {
        system_prompt: Some(context.system_prompt.clone()),
        messages: llm_messages,
        tools: llm_tools,
    }
}

#[derive(Default)]
struct AssistantStreamState {
    has_partial: bool,
    last_message: Option<AgentMessage>,
}

impl AssistantStreamState {
    fn handle_start(
        &mut self,
        context: &mut AgentContext,
        stream: &EventStream<AgentEvent, Vec<AgentMessage>>,
        partial: AssistantMessage,
    ) {
        let message = to_agent_assistant_message(partial);
        context.messages.push(message.clone());
        stream.push(AgentEvent::MessageStart {
            message: message.clone(),
        });
        self.last_message = Some(message);
        self.has_partial = true;
    }

    fn handle_update(
        &mut self,
        context: &mut AgentContext,
        stream: &EventStream<AgentEvent, Vec<AgentMessage>>,
        assistant_message_event: AssistantMessageEvent,
        partial: AssistantMessage,
    ) {
        let message = to_agent_assistant_message(partial);
        if self.has_partial {
            if let Some(last) = context.messages.last_mut() {
                *last = message.clone();
            }
        }
        stream.push(AgentEvent::MessageUpdate {
            message: message.clone(),
            assistant_message_event,
        });
        self.last_message = Some(message);
    }

    fn finalize_message(
        &mut self,
        context: &mut AgentContext,
        stream: &EventStream<AgentEvent, Vec<AgentMessage>>,
        message: AgentMessage,
    ) -> AgentMessage {
        if self.has_partial {
            if let Some(last) = context.messages.last_mut() {
                *last = message.clone();
            }
        } else {
            context.messages.push(message.clone());
            stream.push(AgentEvent::MessageStart {
                message: message.clone(),
            });
            self.has_partial = true;
        }

        stream.push(AgentEvent::MessageEnd {
            message: message.clone(),
        });
        self.last_message = Some(message.clone());
        message
    }

    fn finalize_aborted(
        &mut self,
        context: &mut AgentContext,
        stream: &EventStream<AgentEvent, Vec<AgentMessage>>,
        model: &pixy_ai::Model,
    ) -> AgentMessage {
        let aborted = to_aborted_message(
            self.last_message.clone(),
            model.api.clone(),
            model.provider.clone(),
            model.id.clone(),
        );
        self.finalize_message(context, stream, aborted)
    }

    fn last_message_or_error(self) -> Result<AgentMessage, PiAiError> {
        self.last_message.ok_or_else(|| {
            PiAiError::new(
                PiAiErrorCode::ProviderProtocol,
                "Assistant stream ended without terminal event",
            )
        })
    }
}

struct ToolExecutionOutcome {
    tool_results: Vec<AgentMessage>,
    steering_messages: Option<Vec<AgentMessage>>,
    aborted: bool,
    executed_count: usize,
    executed_total_duration_ms: u64,
}

async fn execute_tool_calls(
    tools: &[AgentTool],
    assistant_message: &AgentMessage,
    stream: &EventStream<AgentEvent, Vec<AgentMessage>>,
    signal: Option<&AgentAbortSignal>,
    get_steering_messages: Option<&MessageQueueFn>,
) -> ToolExecutionOutcome {
    ToolExecutionRunner::new(
        tools,
        assistant_message,
        stream,
        signal,
        get_steering_messages,
    )
    .run()
    .await
}

struct ToolExecutionRunner<'a> {
    tools: &'a [AgentTool],
    stream: &'a EventStream<AgentEvent, Vec<AgentMessage>>,
    signal: Option<&'a AgentAbortSignal>,
    get_steering_messages: Option<&'a MessageQueueFn>,
    tool_calls: Vec<(String, String, Value)>,
    results: Vec<AgentMessage>,
    steering_messages: Option<Vec<AgentMessage>>,
    aborted: bool,
    executed_count: usize,
    executed_total_duration_ms: u64,
}

impl<'a> ToolExecutionRunner<'a> {
    fn new(
        tools: &'a [AgentTool],
        assistant_message: &'a AgentMessage,
        stream: &'a EventStream<AgentEvent, Vec<AgentMessage>>,
        signal: Option<&'a AgentAbortSignal>,
        get_steering_messages: Option<&'a MessageQueueFn>,
    ) -> Self {
        Self {
            tools,
            stream,
            signal,
            get_steering_messages,
            tool_calls: extract_tool_calls(assistant_message),
            results: Vec::new(),
            steering_messages: None,
            aborted: false,
            executed_count: 0,
            executed_total_duration_ms: 0,
        }
    }

    async fn run(mut self) -> ToolExecutionOutcome {
        for index in 0..self.tool_calls.len() {
            if self.is_aborted() {
                self.skip_remaining_calls(index, "Skipped due to abort signal.");
                self.aborted = true;
                break;
            }

            let (tool_call_id, tool_name, args) = self.tool_calls[index].clone();
            self.stream.push(AgentEvent::ToolExecutionStart {
                tool_call_id: tool_call_id.clone(),
                tool_name: tool_name.clone(),
                args: args.clone(),
            });

            let tool_execution_started = Instant::now();
            let (result, is_error) = self
                .execute_single_call(&tool_call_id, &tool_name, args.clone())
                .await;
            let duration_ms = tool_execution_started.elapsed().as_millis() as u64;
            self.executed_count = self.executed_count.saturating_add(1);
            self.executed_total_duration_ms =
                self.executed_total_duration_ms.saturating_add(duration_ms);
            debug!(
                tool_call_id = tool_call_id.as_str(),
                tool_name = tool_name.as_str(),
                duration_ms,
                is_error,
                "tool execution finished"
            );

            self.stream.push(AgentEvent::ToolExecutionEnd {
                tool_call_id: tool_call_id.clone(),
                tool_name: tool_name.clone(),
                result: result.clone(),
                is_error,
                duration_ms,
            });

            let message = Message::ToolResult {
                tool_call_id,
                tool_name,
                content: result.content.clone(),
                details: Some(result.details.clone()),
                is_error,
                timestamp: now_millis(),
            };
            self.stream.push(AgentEvent::MessageStart {
                message: message.clone(),
            });
            self.stream.push(AgentEvent::MessageEnd {
                message: message.clone(),
            });
            self.results.push(message);

            if self.is_aborted() {
                self.skip_remaining_calls(index + 1, "Skipped due to abort signal.");
                self.aborted = true;
                break;
            }

            if self.stop_on_steering(index + 1) {
                break;
            }
        }

        ToolExecutionOutcome {
            tool_results: self.results,
            steering_messages: self.steering_messages,
            aborted: self.aborted,
            executed_count: self.executed_count,
            executed_total_duration_ms: self.executed_total_duration_ms,
        }
    }

    fn is_aborted(&self) -> bool {
        is_aborted(self.signal)
    }

    async fn execute_single_call(
        &self,
        tool_call_id: &str,
        tool_name: &str,
        args: Value,
    ) -> (AgentToolResult, bool) {
        let tool = self.tools.iter().find(|tool| tool.name == tool_name);
        if let Some(tool) = tool {
            let execute_future = tool.execute.execute(tool_call_id.to_string(), args);
            let execution = if let Some(signal_ref) = self.signal {
                tokio::select! {
                    _ = signal_ref.cancelled() => Err(tool_execution_aborted_error()),
                    result = execute_future => result,
                }
            } else {
                execute_future.await
            };
            return match execution {
                Ok(result) => (result, false),
                Err(error) => (tool_error_result(error), true),
            };
        }

        (tool_error_result(tool_not_found_error(tool_name)), true)
    }

    fn stop_on_steering(&mut self, next_index: usize) -> bool {
        let Some(queue) = self.get_steering_messages else {
            return false;
        };

        let steering = queue.poll();
        if steering.is_empty() {
            return false;
        }

        self.steering_messages = Some(steering);
        self.skip_remaining_calls(next_index, "Skipped due to queued user message.");
        true
    }

    fn skip_remaining_calls(&mut self, start_index: usize, reason: &str) {
        for (id, name, args) in self.tool_calls.iter().skip(start_index) {
            self.results
                .push(skip_tool_call(id, name, args, self.stream, reason));
        }
    }
}

fn extract_tool_calls(message: &AgentMessage) -> Vec<(String, String, Value)> {
    match message {
        Message::Assistant { content, .. } => content
            .iter()
            .filter_map(|block| {
                if let AssistantContentBlock::ToolCall {
                    id,
                    name,
                    arguments,
                    ..
                } = block
                {
                    Some((id.clone(), name.clone(), arguments.clone()))
                } else {
                    None
                }
            })
            .collect(),
        _ => vec![],
    }
}

fn skip_tool_call(
    tool_call_id: &str,
    tool_name: &str,
    args: &Value,
    stream: &EventStream<AgentEvent, Vec<AgentMessage>>,
    reason: &str,
) -> AgentMessage {
    let result = AgentToolResult {
        content: vec![ToolResultContentBlock::Text {
            text: reason.to_string(),
            text_signature: None,
        }],
        details: json!({}),
    };

    stream.push(AgentEvent::ToolExecutionStart {
        tool_call_id: tool_call_id.to_string(),
        tool_name: tool_name.to_string(),
        args: args.clone(),
    });
    stream.push(AgentEvent::ToolExecutionEnd {
        tool_call_id: tool_call_id.to_string(),
        tool_name: tool_name.to_string(),
        result: result.clone(),
        is_error: true,
        duration_ms: 0,
    });

    let message = Message::ToolResult {
        tool_call_id: tool_call_id.to_string(),
        tool_name: tool_name.to_string(),
        content: result.content,
        details: Some(result.details),
        is_error: true,
        timestamp: now_millis(),
    };
    stream.push(AgentEvent::MessageStart {
        message: message.clone(),
    });
    stream.push(AgentEvent::MessageEnd {
        message: message.clone(),
    });
    message
}

fn is_error_or_aborted(message: &AgentMessage) -> bool {
    match message {
        Message::Assistant { stop_reason, .. } => {
            matches!(stop_reason, StopReason::Error | StopReason::Aborted)
        }
        _ => false,
    }
}

fn is_aborted(signal: Option<&AgentAbortSignal>) -> bool {
    signal.map(|signal| signal.is_aborted()).unwrap_or(false)
}

fn attempt_models(config: &AgentLoopConfig) -> Vec<pixy_ai::Model> {
    let mut models = vec![config.model.clone()];
    for fallback in &config.fallback_models {
        if models
            .iter()
            .any(|existing| existing.provider == fallback.provider && existing.id == fallback.id)
        {
            continue;
        }
        models.push(fallback.clone());
    }
    models
}

fn retry_delay_ms(retry: &crate::types::AgentRetryConfig, attempt: usize) -> u64 {
    if retry.initial_backoff_ms == 0 {
        return 0;
    }
    let shift = attempt.saturating_sub(1).min(62) as u32;
    let factor = 1_u64 << shift;
    let delay = retry.initial_backoff_ms.saturating_mul(factor);
    if retry.max_backoff_ms == 0 {
        delay
    } else {
        delay.min(retry.max_backoff_ms)
    }
}

fn to_agent_assistant_message(message: AssistantMessage) -> AgentMessage {
    Message::Assistant {
        content: message.content,
        api: message.api,
        provider: message.provider,
        model: message.model,
        usage: message.usage,
        stop_reason: message.stop_reason,
        error_message: message.error_message,
        timestamp: message.timestamp,
    }
}

fn to_aborted_message(
    partial: Option<AgentMessage>,
    api: String,
    provider: String,
    model: String,
) -> AgentMessage {
    if let Some(Message::Assistant {
        content,
        api,
        provider,
        model,
        usage,
        timestamp,
        ..
    }) = partial
    {
        return Message::Assistant {
            content,
            api,
            provider,
            model,
            usage,
            stop_reason: StopReason::Aborted,
            error_message: Some("Request was aborted".to_string()),
            timestamp,
        };
    }

    aborted_assistant_message(api, provider, model)
}

fn error_assistant_message(
    api: String,
    provider: String,
    model: String,
    error_message: String,
) -> AgentMessage {
    status_assistant_message(api, provider, model, StopReason::Error, Some(error_message))
}

fn aborted_assistant_message(api: String, provider: String, model: String) -> AgentMessage {
    status_assistant_message(
        api,
        provider,
        model,
        StopReason::Aborted,
        Some("Request was aborted".to_string()),
    )
}

fn status_assistant_message(
    api: String,
    provider: String,
    model: String,
    stop_reason: StopReason,
    error_message: Option<String>,
) -> AgentMessage {
    Message::Assistant {
        content: vec![AssistantContentBlock::Text {
            text: String::new(),
            text_signature: None,
        }],
        api,
        provider,
        model,
        usage: zero_usage(),
        stop_reason,
        error_message,
        timestamp: now_millis(),
    }
}

fn zero_usage() -> pixy_ai::Usage {
    pixy_ai::Usage {
        input: 0,
        output: 0,
        cache_read: 0,
        cache_write: 0,
        total_tokens: 0,
        cost: pixy_ai::Cost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
            total: 0.0,
        },
    }
}

fn tool_error_result(error: PiAiError) -> AgentToolResult {
    AgentToolResult {
        content: vec![ToolResultContentBlock::Text {
            text: error.message.clone(),
            text_signature: None,
        }],
        details: json!({
            "error": error,
        }),
    }
}

fn tool_execution_aborted_error() -> PiAiError {
    PiAiError::new(PiAiErrorCode::ToolExecutionFailed, "Tool execution aborted")
}

fn tool_not_found_error(tool_name: &str) -> PiAiError {
    PiAiError::new(
        PiAiErrorCode::ToolNotFound,
        format!("Tool {tool_name} not found"),
    )
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

fn emit_metrics_event(
    stream: &EventStream<AgentEvent, Vec<AgentMessage>>,
    metrics: &AgentRunMetrics,
) {
    debug!(
        assistant_request_count = metrics.assistant_request_count,
        assistant_request_total_ms = metrics.assistant_request_total_ms,
        tool_execution_count = metrics.tool_execution_count,
        tool_execution_total_ms = metrics.tool_execution_total_ms,
        retry_count = metrics.retry_count,
        "agent loop metrics"
    );
    stream.push(AgentEvent::Metrics {
        metrics: metrics.clone(),
    });
}

fn pull_queue_messages(callback: Option<&MessageQueueFn>) -> Vec<AgentMessage> {
    callback.map(|callback| callback.poll()).unwrap_or_default()
}
