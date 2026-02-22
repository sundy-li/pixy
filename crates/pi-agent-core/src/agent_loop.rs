use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use pi_ai::{
    AssistantContentBlock, AssistantMessage, AssistantMessageEvent, Context, EventStream, Message,
    StopReason, ToolResultContentBlock,
};
use serde_json::{Value, json};
use tracing::{debug, warn};

use crate::types::{
    AgentAbortSignal, AgentContext, AgentEvent, AgentLoopConfig, AgentMessage, AgentRunMetrics,
    AgentTool, AgentToolResult, MessageQueueFn,
};

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

pub fn agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Option<AgentAbortSignal>,
) -> EventStream<AgentEvent, Vec<AgentMessage>> {
    if context.messages.is_empty() {
        panic!("Cannot continue: no messages in context");
    }
    if matches!(context.messages.last(), Some(Message::Assistant { .. })) {
        panic!("Cannot continue from message role: assistant");
    }
    agent_loop(vec![], context, config, signal)
}

async fn run_loop(
    prompts: Vec<AgentMessage>,
    mut context: AgentContext,
    config: AgentLoopConfig,
    signal: Option<AgentAbortSignal>,
    stream: EventStream<AgentEvent, Vec<AgentMessage>>,
) {
    let mut new_messages = prompts.clone();
    let mut metrics = AgentRunMetrics::default();

    stream.push(AgentEvent::AgentStart);
    stream.push(AgentEvent::TurnStart);
    for prompt in &prompts {
        stream.push(AgentEvent::MessageStart {
            message: prompt.clone(),
        });
        stream.push(AgentEvent::MessageEnd {
            message: prompt.clone(),
        });
        context.messages.push(prompt.clone());
    }

    let mut pending_messages = pull_queue_messages(config.get_steering_messages.as_ref());
    let mut first_assistant_turn = true;

    loop {
        let mut has_more_tool_calls = true;
        let mut steering_after_tools: Option<Vec<AgentMessage>> = None;

        while has_more_tool_calls || !pending_messages.is_empty() {
            if !first_assistant_turn {
                stream.push(AgentEvent::TurnStart);
            } else {
                first_assistant_turn = false;
            }

            if !pending_messages.is_empty() {
                let messages = pending_messages.clone();
                pending_messages.clear();
                for message in messages {
                    stream.push(AgentEvent::MessageStart {
                        message: message.clone(),
                    });
                    stream.push(AgentEvent::MessageEnd {
                        message: message.clone(),
                    });
                    context.messages.push(message.clone());
                    new_messages.push(message);
                }
            }

            if is_aborted(signal.as_ref()) {
                let message = aborted_assistant_message(
                    config.model.api.clone(),
                    config.model.provider.clone(),
                    config.model.id.clone(),
                );
                stream.push(AgentEvent::MessageStart {
                    message: message.clone(),
                });
                stream.push(AgentEvent::MessageEnd {
                    message: message.clone(),
                });
                context.messages.push(message.clone());
                new_messages.push(message.clone());
                stream.push(AgentEvent::TurnEnd {
                    message,
                    tool_results: vec![],
                });
                emit_metrics_event(&stream, &metrics);
                stream.push(AgentEvent::AgentEnd {
                    messages: new_messages.clone(),
                });
                stream.end(Some(new_messages));
                return;
            }

            let assistant_outcome =
                match stream_assistant_response(&mut context, &config, &stream, signal.as_ref())
                    .await
                {
                    Ok(outcome) => outcome,
                    Err(error) => {
                        let message = error_assistant_message(
                            config.model.api.clone(),
                            config.model.provider.clone(),
                            config.model.id.clone(),
                            error,
                        );
                        stream.push(AgentEvent::MessageStart {
                            message: message.clone(),
                        });
                        stream.push(AgentEvent::MessageEnd {
                            message: message.clone(),
                        });
                        context.messages.push(message.clone());
                        AssistantResponseOutcome {
                            message,
                            duration_ms: 0,
                            retries: 0,
                        }
                    }
                };
            metrics.assistant_request_count = metrics.assistant_request_count.saturating_add(1);
            metrics.assistant_request_total_ms = metrics
                .assistant_request_total_ms
                .saturating_add(assistant_outcome.duration_ms);
            metrics.retry_count = metrics
                .retry_count
                .saturating_add(assistant_outcome.retries);
            let assistant_message = assistant_outcome.message;

            new_messages.push(assistant_message.clone());

            if is_error_or_aborted(&assistant_message) {
                stream.push(AgentEvent::TurnEnd {
                    message: assistant_message,
                    tool_results: vec![],
                });
                emit_metrics_event(&stream, &metrics);
                stream.push(AgentEvent::AgentEnd {
                    messages: new_messages.clone(),
                });
                stream.end(Some(new_messages));
                return;
            }

            let mut tool_results = vec![];
            has_more_tool_calls = !extract_tool_calls(&assistant_message).is_empty();
            let mut aborted_during_tools = false;

            if has_more_tool_calls {
                let outcome = execute_tool_calls(
                    &context.tools,
                    &assistant_message,
                    &stream,
                    signal.as_ref(),
                    config.get_steering_messages.as_ref(),
                )
                .await;
                tool_results = outcome.tool_results;
                steering_after_tools = outcome.steering_messages;
                aborted_during_tools = outcome.aborted;
                metrics.tool_execution_count = metrics
                    .tool_execution_count
                    .saturating_add(outcome.executed_count);
                metrics.tool_execution_total_ms = metrics
                    .tool_execution_total_ms
                    .saturating_add(outcome.executed_total_duration_ms);
            }

            for tool_result in &tool_results {
                context.messages.push(tool_result.clone());
                new_messages.push(tool_result.clone());
            }

            stream.push(AgentEvent::TurnEnd {
                message: assistant_message,
                tool_results: tool_results.clone(),
            });

            if aborted_during_tools {
                emit_metrics_event(&stream, &metrics);
                stream.push(AgentEvent::AgentEnd {
                    messages: new_messages.clone(),
                });
                stream.end(Some(new_messages));
                return;
            }

            pending_messages = if let Some(messages) = steering_after_tools.take() {
                if messages.is_empty() {
                    pull_queue_messages(config.get_steering_messages.as_ref())
                } else {
                    messages
                }
            } else {
                pull_queue_messages(config.get_steering_messages.as_ref())
            };
        }

        let follow_up_messages = pull_queue_messages(config.get_follow_up_messages.as_ref());
        if !follow_up_messages.is_empty() {
            pending_messages = follow_up_messages;
            continue;
        }

        break;
    }

    emit_metrics_event(&stream, &metrics);
    stream.push(AgentEvent::AgentEnd {
        messages: new_messages.clone(),
    });
    stream.end(Some(new_messages));
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
) -> Result<AssistantResponseOutcome, String> {
    let models = attempt_models(config);
    let max_attempts = config.retry.max_attempts.max(1);
    let mut attempt = 1usize;
    let started_at = Instant::now();

    loop {
        let model_index = (attempt - 1).min(models.len().saturating_sub(1));
        let active_model = models[model_index].clone();

        match stream_assistant_response_once(context, config, stream, signal, &active_model).await {
            Ok(message) => {
                let retries = attempt.saturating_sub(1);
                debug!(
                    provider = active_model.provider.as_str(),
                    model = active_model.id.as_str(),
                    attempts = attempt,
                    retries,
                    duration_ms = started_at.elapsed().as_millis() as u64,
                    "assistant response completed"
                );
                return Ok(AssistantResponseOutcome {
                    message,
                    duration_ms: started_at.elapsed().as_millis() as u64,
                    retries,
                });
            }
            Err(error) => {
                warn!(
                    provider = active_model.provider.as_str(),
                    model = active_model.id.as_str(),
                    attempt,
                    max_attempts,
                    error = error.as_str(),
                    "assistant response attempt failed"
                );
                if attempt >= max_attempts {
                    return Err(error);
                }

                let next_attempt = attempt + 1;
                let next_index = (next_attempt - 1).min(models.len().saturating_sub(1));
                let next_model = &models[next_index];
                if next_model.provider != active_model.provider || next_model.id != active_model.id
                {
                    warn!(
                        from_provider = active_model.provider.as_str(),
                        from_model = active_model.id.as_str(),
                        to_provider = next_model.provider.as_str(),
                        to_model = next_model.id.as_str(),
                        attempt,
                        "switching to fallback model for next retry"
                    );
                    stream.push(AgentEvent::ModelFallback {
                        from_provider: active_model.provider.clone(),
                        from_model: active_model.id.clone(),
                        to_provider: next_model.provider.clone(),
                        to_model: next_model.id.clone(),
                    });
                }

                let delay_ms = retry_delay_ms(&config.retry, attempt);
                warn!(
                    provider = active_model.provider.as_str(),
                    model = active_model.id.as_str(),
                    attempt,
                    max_attempts,
                    delay_ms,
                    "scheduling retry after assistant response failure"
                );
                stream.push(AgentEvent::RetryScheduled {
                    attempt,
                    max_attempts,
                    delay_ms,
                    error: error.clone(),
                });

                if delay_ms > 0 {
                    if let Some(signal_ref) = signal {
                        tokio::select! {
                            _ = signal_ref.cancelled() => {
                                return Ok(AssistantResponseOutcome {
                                    message: aborted_assistant_message(
                                        active_model.api.clone(),
                                        active_model.provider.clone(),
                                        active_model.id.clone(),
                                    ),
                                    duration_ms: started_at.elapsed().as_millis() as u64,
                                    retries: attempt.saturating_sub(1),
                                });
                            }
                            _ = tokio::time::sleep(Duration::from_millis(delay_ms)) => {}
                        }
                    } else {
                        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    }
                }

                attempt = next_attempt;
            }
        }
    }
}

async fn stream_assistant_response_once(
    context: &mut AgentContext,
    config: &AgentLoopConfig,
    stream: &EventStream<AgentEvent, Vec<AgentMessage>>,
    signal: Option<&AgentAbortSignal>,
    model: &pi_ai::Model,
) -> Result<AgentMessage, String> {
    let llm_messages = (config.convert_to_llm)(context.messages.clone());
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
    let llm_context = Context {
        system_prompt: Some(context.system_prompt.clone()),
        messages: llm_messages,
        tools: llm_tools,
    };

    let stream_fn = config.stream_fn.clone();
    let stream_model = model.clone();
    let response =
        tokio::task::spawn_blocking(move || (stream_fn)(stream_model, llm_context, None))
            .await
            .map_err(|error| format!("Assistant stream task failed: {error}"))??;

    let mut added_partial = false;
    let mut last_message: Option<AgentMessage> = None;

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
            let aborted = to_aborted_message(
                last_message,
                model.api.clone(),
                model.provider.clone(),
                model.id.clone(),
            );
            if added_partial {
                if let Some(last) = context.messages.last_mut() {
                    *last = aborted.clone();
                }
            } else {
                context.messages.push(aborted.clone());
                stream.push(AgentEvent::MessageStart {
                    message: aborted.clone(),
                });
            }
            stream.push(AgentEvent::MessageEnd {
                message: aborted.clone(),
            });
            return Ok(aborted);
        }

        let Some(event) = next_event else {
            break;
        };

        match &event {
            AssistantMessageEvent::Start { partial } => {
                let message = to_agent_assistant_message(partial.clone());
                context.messages.push(message.clone());
                stream.push(AgentEvent::MessageStart {
                    message: message.clone(),
                });
                last_message = Some(message);
                added_partial = true;
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
                let message = to_agent_assistant_message(partial.clone());
                if added_partial {
                    if let Some(last) = context.messages.last_mut() {
                        *last = message.clone();
                    }
                }
                stream.push(AgentEvent::MessageUpdate {
                    message: message.clone(),
                    assistant_message_event: event.clone(),
                });
                last_message = Some(message);
            }
            AssistantMessageEvent::Done { message, .. } => {
                let final_message = to_agent_assistant_message(message.clone());
                if added_partial {
                    if let Some(last) = context.messages.last_mut() {
                        *last = final_message.clone();
                    }
                } else {
                    context.messages.push(final_message.clone());
                    stream.push(AgentEvent::MessageStart {
                        message: final_message.clone(),
                    });
                }
                stream.push(AgentEvent::MessageEnd {
                    message: final_message.clone(),
                });
                return Ok(final_message);
            }
            AssistantMessageEvent::Error { error, .. } => {
                let final_message = to_agent_assistant_message(error.clone());
                if added_partial {
                    if let Some(last) = context.messages.last_mut() {
                        *last = final_message.clone();
                    }
                } else {
                    context.messages.push(final_message.clone());
                    stream.push(AgentEvent::MessageStart {
                        message: final_message.clone(),
                    });
                }
                stream.push(AgentEvent::MessageEnd {
                    message: final_message.clone(),
                });
                return Ok(final_message);
            }
        }
    }

    if is_aborted(signal) {
        let aborted = to_aborted_message(
            last_message,
            model.api.clone(),
            model.provider.clone(),
            model.id.clone(),
        );
        if added_partial {
            if let Some(last) = context.messages.last_mut() {
                *last = aborted.clone();
            }
        } else {
            context.messages.push(aborted.clone());
            stream.push(AgentEvent::MessageStart {
                message: aborted.clone(),
            });
        }
        stream.push(AgentEvent::MessageEnd {
            message: aborted.clone(),
        });
        return Ok(aborted);
    }

    if let Some(message) = last_message {
        Ok(message)
    } else {
        Err("Assistant stream ended without terminal event".to_string())
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
    execute_tool_calls_with_controls(
        tools,
        assistant_message,
        stream,
        signal,
        get_steering_messages,
    )
    .await
}

async fn execute_tool_calls_with_controls(
    tools: &[AgentTool],
    assistant_message: &AgentMessage,
    stream: &EventStream<AgentEvent, Vec<AgentMessage>>,
    signal: Option<&AgentAbortSignal>,
    get_steering_messages: Option<&MessageQueueFn>,
) -> ToolExecutionOutcome {
    let mut results = Vec::new();
    let tool_calls = extract_tool_calls(assistant_message);
    let mut steering_messages = None;
    let mut aborted = false;
    let mut executed_count = 0usize;
    let mut executed_total_duration_ms = 0u64;

    for (index, (tool_call_id, tool_name, args)) in tool_calls.iter().enumerate() {
        if is_aborted(signal) {
            for (id, name, args) in tool_calls.iter().skip(index) {
                results.push(skip_tool_call(
                    id,
                    name,
                    args,
                    stream,
                    "Skipped due to abort signal.",
                ));
            }
            aborted = true;
            break;
        }

        stream.push(AgentEvent::ToolExecutionStart {
            tool_call_id: tool_call_id.to_string(),
            tool_name: tool_name.to_string(),
            args: args.clone(),
        });

        let tool_execution_started = Instant::now();
        let tool = tools.iter().find(|tool| tool.name == *tool_name);
        let (result, is_error) = if let Some(tool) = tool {
            let execute_future = (tool.execute)(tool_call_id.to_string(), args.clone());
            let execution = if let Some(signal_ref) = signal {
                tokio::select! {
                    _ = signal_ref.cancelled() => Err("Tool execution aborted".to_string()),
                    result = execute_future => result,
                }
            } else {
                execute_future.await
            };
            match execution {
                Ok(result) => (result, false),
                Err(error) => (tool_error_result(error), true),
            }
        } else {
            (
                tool_error_result(format!("Tool {tool_name} not found")),
                true,
            )
        };
        let duration_ms = tool_execution_started.elapsed().as_millis() as u64;
        executed_count = executed_count.saturating_add(1);
        executed_total_duration_ms = executed_total_duration_ms.saturating_add(duration_ms);
        debug!(
            tool_call_id = tool_call_id.as_str(),
            tool_name = tool_name.as_str(),
            duration_ms,
            is_error,
            "tool execution finished"
        );

        stream.push(AgentEvent::ToolExecutionEnd {
            tool_call_id: tool_call_id.to_string(),
            tool_name: tool_name.to_string(),
            result: result.clone(),
            is_error,
            duration_ms,
        });

        let message = Message::ToolResult {
            tool_call_id: tool_call_id.to_string(),
            tool_name: tool_name.to_string(),
            content: result.content.clone(),
            details: Some(result.details.clone()),
            is_error,
            timestamp: now_millis(),
        };
        stream.push(AgentEvent::MessageStart {
            message: message.clone(),
        });
        stream.push(AgentEvent::MessageEnd {
            message: message.clone(),
        });
        results.push(message);

        if is_aborted(signal) {
            for (id, name, args) in tool_calls.iter().skip(index + 1) {
                results.push(skip_tool_call(
                    id,
                    name,
                    args,
                    stream,
                    "Skipped due to abort signal.",
                ));
            }
            aborted = true;
            break;
        }

        if let Some(get_steering_messages) = get_steering_messages {
            let steering = get_steering_messages();
            if !steering.is_empty() {
                steering_messages = Some(steering);
                for (id, name, args) in tool_calls.iter().skip(index + 1) {
                    results.push(skip_tool_call(
                        id,
                        name,
                        args,
                        stream,
                        "Skipped due to queued user message.",
                    ));
                }
                break;
            }
        }
    }

    ToolExecutionOutcome {
        tool_results: results,
        steering_messages,
        aborted,
        executed_count,
        executed_total_duration_ms,
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

fn attempt_models(config: &AgentLoopConfig) -> Vec<pi_ai::Model> {
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
    Message::Assistant {
        content: vec![AssistantContentBlock::Text {
            text: String::new(),
            text_signature: None,
        }],
        api,
        provider,
        model,
        usage: pi_ai::Usage {
            input: 0,
            output: 0,
            cache_read: 0,
            cache_write: 0,
            total_tokens: 0,
            cost: pi_ai::Cost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.0,
            },
        },
        stop_reason: StopReason::Error,
        error_message: Some(error_message),
        timestamp: now_millis(),
    }
}

fn aborted_assistant_message(api: String, provider: String, model: String) -> AgentMessage {
    Message::Assistant {
        content: vec![AssistantContentBlock::Text {
            text: String::new(),
            text_signature: None,
        }],
        api,
        provider,
        model,
        usage: pi_ai::Usage {
            input: 0,
            output: 0,
            cache_read: 0,
            cache_write: 0,
            total_tokens: 0,
            cost: pi_ai::Cost {
                input: 0.0,
                output: 0.0,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.0,
            },
        },
        stop_reason: StopReason::Aborted,
        error_message: Some("Request was aborted".to_string()),
        timestamp: now_millis(),
    }
}

fn tool_error_result(error: String) -> AgentToolResult {
    AgentToolResult {
        content: vec![ToolResultContentBlock::Text {
            text: error,
            text_signature: None,
        }],
        details: json!({}),
    }
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
    callback.map(|callback| callback()).unwrap_or_default()
}
