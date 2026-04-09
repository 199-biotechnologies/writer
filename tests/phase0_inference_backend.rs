use async_trait::async_trait;
use tokio_stream::StreamExt;
use writer_cli::backends::inference::capabilities::BackendCapabilities;
use writer_cli::backends::inference::request::GenerationRequest;
use writer_cli::backends::inference::response::{FinishReason, GenerationEvent, UsageStats};
use writer_cli::backends::inference::{BackendError, InferenceBackend, ModelListing};
use writer_cli::backends::types::{ModelHandle, ModelId};

struct FakeBackend;

#[async_trait]
impl InferenceBackend for FakeBackend {
    fn name(&self) -> &str {
        "fake"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_lora: true,
            supports_logit_bias: true,
            max_context: 8192,
            streaming: true,
            ..Default::default()
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelListing>, BackendError> {
        Ok(vec![ModelListing {
            id: "google/gemma-4-26b-a4b".parse().unwrap(),
            is_downloaded: true,
            size_bytes: Some(14_000_000_000),
        }])
    }

    async fn load_model(&self, id: &ModelId) -> Result<ModelHandle, BackendError> {
        Ok(ModelHandle(format!("fake:{id}")))
    }

    async fn generate(
        &self,
        _handle: &ModelHandle,
        request: GenerationRequest,
    ) -> Result<Box<dyn tokio_stream::Stream<Item = GenerationEvent> + Send + Unpin>, BackendError>
    {
        let n = request.params.n_candidates as usize;
        let events: Vec<GenerationEvent> = (0..n)
            .flat_map(|i| {
                vec![
                    GenerationEvent::Token {
                        candidate_index: i as u16,
                        text: format!("candidate-{i}"),
                        logprob: -0.5,
                    },
                    GenerationEvent::Done {
                        candidate_index: i as u16,
                        finish_reason: FinishReason::Stop,
                        usage: UsageStats {
                            prompt_tokens: 1,
                            generated_tokens: 1,
                            elapsed_ms: 1,
                        },
                        full_text: format!("candidate-{i}"),
                    },
                ]
            })
            .collect();
        Ok(Box::new(tokio_stream::iter(events)))
    }
}

#[tokio::test]
async fn fake_backend_round_trip() {
    let backend = FakeBackend;
    assert_eq!(backend.name(), "fake");
    assert!(backend.capabilities().supports_lora);

    let models = backend.list_models().await.unwrap();
    assert_eq!(models.len(), 1);

    let handle = backend.load_model(&models[0].id).await.unwrap();
    assert!(handle.0.starts_with("fake:"));

    let req = GenerationRequest::new(models[0].id.clone(), "hello".into()).with_n_candidates(3);
    let mut stream = backend.generate(&handle, req).await.unwrap();

    let mut done_count = 0;
    while let Some(ev) = stream.next().await {
        if matches!(ev, GenerationEvent::Done { .. }) {
            done_count += 1;
        }
    }
    assert_eq!(done_count, 3);
}
