use std::collections::HashSet;
use std::sync::{Arc, Mutex, OnceLock};

use crate::api_registry::{ApiProvider, ApiProviderFuture};
use crate::error::{PiAiError, PiAiErrorCode};
use crate::types::{Context, Model, SimpleStreamOptions, StreamOptions};
use crate::{ApiProviderRef, AssistantMessageEventStream};

use super::openai_completions::{run_openai_completions, run_simple_openai_completions};
use super::openai_responses::{run_openai_responses, run_simple_openai_responses};

struct OpenAICompatProvider {
    api: &'static str,
}

impl ApiProvider for OpenAICompatProvider {
    fn api(&self) -> &str {
        self.api
    }

    fn stream(
        &self,
        model: Model,
        context: Context,
        options: Option<StreamOptions>,
        stream: AssistantMessageEventStream,
    ) -> ApiProviderFuture {
        let api = self.api;
        Box::pin(async move {
            if api == "openai-responses" {
                return run_openai_responses_with_fallback(model, context, options, stream).await;
            }
            run_openai_responses(model, context, options, stream).await
        })
    }

    fn stream_simple(
        &self,
        model: Model,
        context: Context,
        options: Option<SimpleStreamOptions>,
        stream: AssistantMessageEventStream,
    ) -> ApiProviderFuture {
        let api = self.api;
        Box::pin(async move {
            if api == "openai-responses" {
                return run_simple_openai_responses_with_fallback(model, context, options, stream)
                    .await;
            }
            run_simple_openai_responses(model, context, options, stream).await
        })
    }
}

fn provider_for_api(api: &'static str) -> ApiProviderRef {
    Arc::new(OpenAICompatProvider { api })
}

pub(super) fn openai_responses_provider() -> ApiProviderRef {
    provider_for_api("openai-responses")
}

pub(super) fn azure_openai_responses_provider() -> ApiProviderRef {
    provider_for_api("azure-openai-responses")
}

pub(super) fn openai_codex_responses_provider() -> ApiProviderRef {
    provider_for_api("openai-codex-responses")
}

async fn run_openai_responses_with_fallback(
    model: Model,
    context: Context,
    options: Option<StreamOptions>,
    stream: AssistantMessageEventStream,
) -> Result<(), PiAiError> {
    if cached_responses_fallback(&model.base_url) {
        return run_openai_completions(
            as_openai_completions_model(model),
            context,
            options,
            stream,
        )
        .await;
    }

    let base_url = model.base_url.clone();
    match run_openai_responses(
        model.clone(),
        context.clone(),
        options.clone(),
        stream.clone(),
    )
    .await
    {
        Ok(()) => Ok(()),
        Err(error) if is_responses_404_error(&error) => {
            cache_responses_fallback(&base_url);
            run_openai_completions(as_openai_completions_model(model), context, options, stream)
                .await
        }
        Err(error) => Err(error),
    }
}

async fn run_simple_openai_responses_with_fallback(
    model: Model,
    context: Context,
    options: Option<SimpleStreamOptions>,
    stream: AssistantMessageEventStream,
) -> Result<(), PiAiError> {
    if cached_responses_fallback(&model.base_url) {
        return run_simple_openai_completions(
            as_openai_completions_model(model),
            context,
            options,
            stream,
        )
        .await;
    }

    let base_url = model.base_url.clone();
    match run_simple_openai_responses(
        model.clone(),
        context.clone(),
        options.clone(),
        stream.clone(),
    )
    .await
    {
        Ok(()) => Ok(()),
        Err(error) if is_responses_404_error(&error) => {
            cache_responses_fallback(&base_url);
            run_simple_openai_completions(
                as_openai_completions_model(model),
                context,
                options,
                stream,
            )
            .await
        }
        Err(error) => Err(error),
    }
}

fn responses_fallback_cache() -> &'static Mutex<HashSet<String>> {
    static CACHE: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashSet::new()))
}

fn cache_responses_fallback(base_url: &str) {
    let mut cache = responses_fallback_cache()
        .lock()
        .expect("responses fallback cache lock poisoned");
    cache.insert(fallback_cache_key(base_url));
}

fn cached_responses_fallback(base_url: &str) -> bool {
    let cache = responses_fallback_cache()
        .lock()
        .expect("responses fallback cache lock poisoned");
    cache.contains(&fallback_cache_key(base_url))
}

fn fallback_cache_key(base_url: &str) -> String {
    base_url.trim_end_matches('/').to_string()
}

fn as_openai_completions_model(mut model: Model) -> Model {
    model.api = "openai-completions".to_string();
    model
}

fn is_responses_404_error(error: &PiAiError) -> bool {
    error.code == PiAiErrorCode::ProviderHttp && error.message.contains("HTTP 404")
}
