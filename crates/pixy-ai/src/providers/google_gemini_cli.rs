use std::sync::Arc;

use crate::api_registry::{ApiProvider, ApiProviderFuture};
use crate::types::{Context, Model, SimpleStreamOptions, StreamOptions};
use crate::{ApiProviderRef, AssistantMessageEventStream};

use super::google_generative_ai::{
    run_google_with_mode, run_simple_google_with_mode, GoogleAuthMode,
};

const GOOGLE_GEMINI_CLI_FALLBACK_ENVS: &[&str] = &[
    "GOOGLE_GEMINI_CLI_AUTH_TOKEN",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
];

struct GoogleGeminiCliProvider;

impl ApiProvider for GoogleGeminiCliProvider {
    fn api(&self) -> &str {
        "google-gemini-cli"
    }

    fn stream(
        &self,
        model: Model,
        context: Context,
        options: Option<StreamOptions>,
        stream: AssistantMessageEventStream,
    ) -> ApiProviderFuture {
        Box::pin(async move {
            run_google_with_mode(
                model,
                context,
                options,
                GoogleAuthMode::Auto,
                GOOGLE_GEMINI_CLI_FALLBACK_ENVS,
                stream,
            )
            .await
        })
    }

    fn stream_simple(
        &self,
        model: Model,
        context: Context,
        options: Option<SimpleStreamOptions>,
        stream: AssistantMessageEventStream,
    ) -> ApiProviderFuture {
        Box::pin(async move {
            run_simple_google_with_mode(
                model,
                context,
                options,
                GoogleAuthMode::Auto,
                GOOGLE_GEMINI_CLI_FALLBACK_ENVS,
                stream,
            )
            .await
        })
    }
}

pub(super) fn provider() -> ApiProviderRef {
    Arc::new(GoogleGeminiCliProvider)
}
