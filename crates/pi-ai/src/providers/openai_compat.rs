use std::collections::HashSet;
use std::sync::{Arc, Mutex, OnceLock};

use crate::api_registry::ApiProvider;
use crate::error::{PiAiError, PiAiErrorCode};
use crate::types::{Context, Model, SimpleStreamOptions, StreamOptions};
use crate::{ApiProviderRef, AssistantMessageEventStream};

use super::openai_completions::{stream_openai_completions, stream_simple_openai_completions};
use super::openai_responses::{stream_openai_responses, stream_simple_openai_responses};

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
    ) -> Result<AssistantMessageEventStream, String> {
        if self.api == "openai-responses" {
            return stream_openai_responses_with_fallback(model, context, options);
        }
        stream_openai_responses(model, context, options)
    }

    fn stream_simple(
        &self,
        model: Model,
        context: Context,
        options: Option<SimpleStreamOptions>,
    ) -> Result<AssistantMessageEventStream, String> {
        if self.api == "openai-responses" {
            return stream_simple_openai_responses_with_fallback(model, context, options);
        }
        stream_simple_openai_responses(model, context, options)
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

fn stream_openai_responses_with_fallback(
    model: Model,
    context: Context,
    options: Option<StreamOptions>,
) -> Result<AssistantMessageEventStream, String> {
    if cached_responses_fallback(&model.base_url) {
        return stream_openai_completions(as_openai_completions_model(model), context, options);
    }

    let base_url = model.base_url.clone();
    match stream_openai_responses(model.clone(), context.clone(), options.clone()) {
        Ok(stream) => Ok(stream),
        Err(error) if is_responses_404_error(&error) => {
            cache_responses_fallback(&base_url);
            stream_openai_completions(as_openai_completions_model(model), context, options)
        }
        Err(error) => Err(error),
    }
}

fn stream_simple_openai_responses_with_fallback(
    model: Model,
    context: Context,
    options: Option<SimpleStreamOptions>,
) -> Result<AssistantMessageEventStream, String> {
    if cached_responses_fallback(&model.base_url) {
        return stream_simple_openai_completions(
            as_openai_completions_model(model),
            context,
            options,
        );
    }

    let base_url = model.base_url.clone();
    match stream_simple_openai_responses(model.clone(), context.clone(), options.clone()) {
        Ok(stream) => Ok(stream),
        Err(error) if is_responses_404_error(&error) => {
            cache_responses_fallback(&base_url);
            stream_simple_openai_completions(as_openai_completions_model(model), context, options)
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

fn is_responses_404_error(error: &str) -> bool {
    if let Ok(parsed) = serde_json::from_str::<PiAiError>(error) {
        return parsed.code == PiAiErrorCode::ProviderHttp && parsed.message.contains("HTTP 404");
    }
    error.contains("HTTP 404")
}
