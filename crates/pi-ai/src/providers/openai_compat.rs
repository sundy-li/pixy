use std::sync::Arc;

use crate::api_registry::ApiProvider;
use crate::types::{Context, Model, SimpleStreamOptions, StreamOptions};
use crate::{ApiProviderRef, AssistantMessageEventStream};

use super::openai_completions::{stream_openai_completions, stream_simple_openai_completions};

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
        stream_openai_completions(model, context, options)
    }

    fn stream_simple(
        &self,
        model: Model,
        context: Context,
        options: Option<SimpleStreamOptions>,
    ) -> Result<AssistantMessageEventStream, String> {
        stream_simple_openai_completions(model, context, options)
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
