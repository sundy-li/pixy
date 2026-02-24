use std::sync::Arc;

use crate::api_registry::{ApiProvider, ApiProviderFuture};
use crate::types::{Context, Model, SimpleStreamOptions, StreamOptions};
use crate::{ApiProviderRef, AssistantMessageEventStream};

use super::google_generative_ai::{
    GoogleAuthMode, run_google_with_mode, run_simple_google_with_mode,
};

const GOOGLE_VERTEX_FALLBACK_ENVS: &[&str] = &[
    "GOOGLE_VERTEX_ACCESS_TOKEN",
    "GOOGLE_OAUTH_ACCESS_TOKEN",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
];

struct GoogleVertexProvider;

impl ApiProvider for GoogleVertexProvider {
    fn api(&self) -> &str {
        "google-vertex"
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
                GoogleAuthMode::Bearer,
                GOOGLE_VERTEX_FALLBACK_ENVS,
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
                GoogleAuthMode::Bearer,
                GOOGLE_VERTEX_FALLBACK_ENVS,
                stream,
            )
            .await
        })
    }
}

pub(super) fn provider() -> ApiProviderRef {
    Arc::new(GoogleVertexProvider)
}
