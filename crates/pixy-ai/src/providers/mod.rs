use std::sync::{Arc, Once};

use crate::ApiProviderRef;
use crate::api_registry::{clear_api_providers, register_api_provider};

mod anthropic;
mod bedrock_converse_stream;
mod common;
mod google_gemini_cli;
mod google_generative_ai;
mod google_vertex;
mod openai_compat;
mod openai_completions;
mod openai_responses;
mod reliable;

pub use reliable::ReliableProvider;

const BUILTIN_SOURCE_ID: &str = "pixy-ai-builtins";

fn register_builtin_provider(provider: ApiProviderRef) {
    let provider = Arc::new(reliable::ReliableProvider::wrap(provider));
    register_api_provider(provider, Some(BUILTIN_SOURCE_ID.to_string()));
}

pub fn register_builtin_api_providers() {
    register_builtin_provider(openai_completions::provider());
    register_builtin_provider(anthropic::provider());
    register_builtin_provider(openai_compat::openai_responses_provider());
    register_builtin_provider(openai_compat::azure_openai_responses_provider());
    register_builtin_provider(openai_compat::openai_codex_responses_provider());
    register_builtin_provider(google_generative_ai::provider());
    register_builtin_provider(google_gemini_cli::provider());
    register_builtin_provider(google_vertex::provider());
    register_builtin_provider(bedrock_converse_stream::provider());
}

pub fn reset_api_providers() {
    clear_api_providers();
    register_builtin_api_providers();
}

pub(crate) fn ensure_builtin_api_providers_registered() {
    static ONCE: Once = Once::new();
    ONCE.call_once(register_builtin_api_providers);
}
