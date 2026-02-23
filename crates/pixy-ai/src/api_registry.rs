use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use crate::AssistantMessageEventStream;
use crate::types::{Api, Context, Model, SimpleStreamOptions, StreamOptions};

pub type StreamResult<T> = Result<T, String>;

pub type ApiStreamFunction = Arc<
    dyn Fn(Model, Context, Option<StreamOptions>) -> StreamResult<AssistantMessageEventStream>
        + Send
        + Sync,
>;

pub type ApiStreamSimpleFunction = Arc<
    dyn Fn(Model, Context, Option<SimpleStreamOptions>) -> StreamResult<AssistantMessageEventStream>
        + Send
        + Sync,
>;

pub trait ApiProvider: Send + Sync {
    fn api(&self) -> &str;
    fn stream(
        &self,
        model: Model,
        context: Context,
        options: Option<StreamOptions>,
    ) -> StreamResult<AssistantMessageEventStream>;
    fn stream_simple(
        &self,
        model: Model,
        context: Context,
        options: Option<SimpleStreamOptions>,
    ) -> StreamResult<AssistantMessageEventStream>;
}

pub type ApiProviderRef = Arc<dyn ApiProvider>;

#[derive(Clone)]
pub struct ClosureApiProvider {
    pub api: Api,
    pub stream: ApiStreamFunction,
    pub stream_simple: ApiStreamSimpleFunction,
}

impl ApiProvider for ClosureApiProvider {
    fn api(&self) -> &str {
        &self.api
    }

    fn stream(
        &self,
        model: Model,
        context: Context,
        options: Option<StreamOptions>,
    ) -> StreamResult<AssistantMessageEventStream> {
        (self.stream)(model, context, options)
    }

    fn stream_simple(
        &self,
        model: Model,
        context: Context,
        options: Option<SimpleStreamOptions>,
    ) -> StreamResult<AssistantMessageEventStream> {
        (self.stream_simple)(model, context, options)
    }
}

#[derive(Clone)]
struct RegisteredApiProvider {
    provider: ApiProviderRef,
    source_id: Option<String>,
}

fn api_registry() -> &'static RwLock<HashMap<String, RegisteredApiProvider>> {
    static REGISTRY: OnceLock<RwLock<HashMap<String, RegisteredApiProvider>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

pub fn register_api_provider(provider: ApiProviderRef, source_id: Option<String>) {
    let mut registry = api_registry().write().expect("api registry lock poisoned");
    registry.insert(
        provider.api().to_string(),
        RegisteredApiProvider {
            provider,
            source_id,
        },
    );
}

pub fn get_api_provider(api: &str) -> Option<ApiProviderRef> {
    let registry = api_registry().read().expect("api registry lock poisoned");
    registry.get(api).map(|entry| entry.provider.clone())
}

pub fn get_api_providers() -> Vec<ApiProviderRef> {
    let registry = api_registry().read().expect("api registry lock poisoned");
    registry
        .values()
        .map(|entry| entry.provider.clone())
        .collect()
}

pub fn unregister_api_providers(source_id: &str) {
    let mut registry = api_registry().write().expect("api registry lock poisoned");
    registry.retain(|_, entry| entry.source_id.as_deref() != Some(source_id));
}

pub fn clear_api_providers() {
    let mut registry = api_registry().write().expect("api registry lock poisoned");
    registry.clear();
}
