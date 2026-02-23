use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::{Mutex as AsyncMutex, Notify, mpsc};

use crate::types::AssistantMessage;
use crate::types::AssistantMessageEvent;

type CompletionFn<T, R> = dyn Fn(&T) -> Option<R> + Send + Sync;

struct EventStreamInner<T, R> {
    sender: mpsc::UnboundedSender<T>,
    receiver: AsyncMutex<mpsc::UnboundedReceiver<T>>,
    completion: Arc<CompletionFn<T, R>>,
    final_result: Mutex<Option<R>>,
    event_notify: Notify,
    final_notify: Notify,
    done: AtomicBool,
}

pub struct EventStream<T, R> {
    inner: Arc<EventStreamInner<T, R>>,
}

impl<T, R> Clone for EventStream<T, R> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T, R> EventStream<T, R>
where
    T: Clone + Send + 'static,
    R: Clone + Send + 'static,
{
    pub fn new<F>(completion: F) -> Self
    where
        F: Fn(&T) -> Option<R> + Send + Sync + 'static,
    {
        let (sender, receiver) = mpsc::unbounded_channel();
        Self {
            inner: Arc::new(EventStreamInner {
                sender,
                receiver: AsyncMutex::new(receiver),
                completion: Arc::new(completion),
                final_result: Mutex::new(None),
                event_notify: Notify::new(),
                final_notify: Notify::new(),
                done: AtomicBool::new(false),
            }),
        }
    }

    pub fn push(&self, event: T) {
        if self.inner.done.load(Ordering::SeqCst) {
            return;
        }

        let result = (self.inner.completion)(&event);
        if let Some(result) = result {
            let mut guard = self
                .inner
                .final_result
                .lock()
                .expect("final_result mutex poisoned");
            if guard.is_none() {
                *guard = Some(result);
                self.inner.done.store(true, Ordering::SeqCst);
            }
            drop(guard);
            self.inner.final_notify.notify_waiters();
        }

        let _ = self.inner.sender.send(event);
        self.inner.event_notify.notify_waiters();
    }

    pub fn end(&self, result: Option<R>) {
        if self.inner.done.load(Ordering::SeqCst) {
            return;
        }

        if let Some(result) = result {
            let mut guard = self
                .inner
                .final_result
                .lock()
                .expect("final_result mutex poisoned");
            if guard.is_none() {
                *guard = Some(result);
            }
        }

        self.inner.done.store(true, Ordering::SeqCst);
        self.inner.event_notify.notify_waiters();
        self.inner.final_notify.notify_waiters();
    }

    pub async fn next(&self) -> Option<T> {
        loop {
            {
                let mut receiver = self.inner.receiver.lock().await;
                match receiver.try_recv() {
                    Ok(event) => return Some(event),
                    Err(mpsc::error::TryRecvError::Disconnected) => return None,
                    Err(mpsc::error::TryRecvError::Empty) => {
                        if self.inner.done.load(Ordering::SeqCst) {
                            return None;
                        }
                    }
                }
            }

            self.inner.event_notify.notified().await;
        }
    }

    pub async fn result(&self) -> Option<R> {
        loop {
            if let Some(result) = self
                .inner
                .final_result
                .lock()
                .expect("final_result mutex poisoned")
                .clone()
            {
                return Some(result);
            }

            if self.inner.done.load(Ordering::SeqCst) {
                return None;
            }

            self.inner.final_notify.notified().await;
        }
    }
}

pub struct AssistantMessageEventStream {
    inner: EventStream<AssistantMessageEvent, AssistantMessage>,
}

impl AssistantMessageEventStream {
    pub fn new() -> Self {
        let inner = EventStream::new(|event| match event {
            AssistantMessageEvent::Done { message, .. } => Some(message.clone()),
            AssistantMessageEvent::Error { error, .. } => Some(error.clone()),
            _ => None,
        });
        Self { inner }
    }

    pub fn push(&self, event: AssistantMessageEvent) {
        self.inner.push(event);
    }

    pub fn end(&self, result: Option<AssistantMessage>) {
        self.inner.end(result);
    }

    pub async fn next(&self) -> Option<AssistantMessageEvent> {
        self.inner.next().await
    }

    pub async fn result(&self) -> Option<AssistantMessage> {
        self.inner.result().await
    }
}

impl Default for AssistantMessageEventStream {
    fn default() -> Self {
        Self::new()
    }
}
