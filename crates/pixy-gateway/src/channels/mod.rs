use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use tokio::time::Instant;

pub mod feishu;
pub mod telegram;

pub type ChannelFuture<'a> = Pin<Box<dyn Future<Output = Result<(), String>> + 'a>>;
pub type DispatchFuture<'a> = Pin<Box<dyn Future<Output = Result<String, String>> + 'a>>;

pub trait SessionDispatcher {
    fn dispatch_text<'a>(
        &'a mut self,
        channel_name: &'a str,
        user_id: &'a str,
        text: &'a str,
    ) -> DispatchFuture<'a>;
}

pub trait Channel: Send {
    fn name(&self) -> &str;
    fn time_until_next_poll(&self, now: Instant) -> Duration;
    fn poll_if_due<'a>(
        &'a mut self,
        dispatcher: &'a mut dyn SessionDispatcher,
    ) -> ChannelFuture<'a>;
}
