use writer_cli::backends::types::{AdapterRef, ModelHandle, ModelId};

#[test]
fn model_id_parses_owner_repo_format() {
    let id: ModelId = "google/gemma-4-26b-a4b".parse().unwrap();
    assert_eq!(id.owner(), "google");
    assert_eq!(id.name(), "gemma-4-26b-a4b");
    assert_eq!(id.to_string(), "google/gemma-4-26b-a4b");
}

#[test]
fn model_id_rejects_missing_owner() {
    let result: Result<ModelId, _> = "gemma-4-26b-a4b".parse();
    assert!(result.is_err());
}

#[test]
fn adapter_ref_carries_path_and_profile_name() {
    let r = AdapterRef::new("default", std::path::PathBuf::from("/tmp/a.safetensors"));
    assert_eq!(r.profile, "default");
    assert_eq!(r.path, std::path::PathBuf::from("/tmp/a.safetensors"));
}

#[test]
fn model_handle_is_string_newtype() {
    let h = ModelHandle("handle-abc".into());
    assert_eq!(h.0, "handle-abc");
}

use writer_cli::backends::inference::capabilities::{
    BackendCapabilities, KvQuantKind, QuantSchemeKind,
};

#[test]
fn capabilities_default_is_the_narrowest_possible_backend() {
    let caps = BackendCapabilities::default();
    assert!(!caps.supports_lora);
    assert!(!caps.supports_logit_bias);
    assert!(!caps.supports_contrastive_decoding);
    assert!(!caps.supports_speculative_decoding);
    assert_eq!(caps.kv_quant, KvQuantKind::None);
    assert_eq!(caps.quant_schemes, vec![]);
    assert_eq!(caps.max_context, 2048);
}

#[test]
fn capabilities_builder_sets_flags() {
    let caps = BackendCapabilities {
        supports_lora: true,
        supports_logit_bias: true,
        kv_quant: KvQuantKind::TurboQuant,
        quant_schemes: vec![QuantSchemeKind::Q4KM, QuantSchemeKind::Q5KM],
        max_context: 128_000,
        ..Default::default()
    };
    assert!(caps.supports_lora);
    assert!(caps.supports_logit_bias);
    assert_eq!(caps.kv_quant, KvQuantKind::TurboQuant);
    assert_eq!(caps.quant_schemes.len(), 2);
    assert_eq!(caps.max_context, 128_000);
}

use std::collections::HashMap;
use writer_cli::backends::inference::request::{GenerationParams, GenerationRequest, LogitBiasMap};

#[test]
fn generation_request_builder_defaults_are_quality_oriented() {
    let req = GenerationRequest::new(
        "google/gemma-4-26b-a4b".parse::<ModelId>().unwrap(),
        "write an essay about ravens".to_string(),
    );
    assert_eq!(req.params.n_candidates, 8);
    assert_eq!(req.params.temperature, 0.7);
    assert_eq!(req.params.top_p, 0.92);
    assert_eq!(req.params.max_tokens, 2048);
    assert_eq!(req.params.kv_quant_preference, None);
    assert!(req.adapter.is_none());
    assert!(req.logit_bias.is_empty());
}

#[test]
fn generation_request_with_adapter_and_bias() {
    let mut bias: LogitBiasMap = HashMap::new();
    bias.insert("delve".into(), -5.0);
    bias.insert("leverage".into(), -3.0);

    let req = GenerationRequest::new(
        "google/gemma-4-26b-a4b".parse::<ModelId>().unwrap(),
        "draft a blog post".to_string(),
    )
    .with_adapter(AdapterRef::new("default", "/tmp/a.safetensors".into()))
    .with_logit_bias(bias)
    .with_n_candidates(16);

    assert!(req.adapter.is_some());
    assert_eq!(req.logit_bias.get("delve"), Some(&-5.0));
    assert_eq!(req.params.n_candidates, 16);
    // Silence unused-import warning when GenerationParams is not directly used.
    let _ = GenerationParams::default();
}

use writer_cli::backends::inference::response::{FinishReason, GenerationEvent, UsageStats};

#[test]
fn generation_event_token_carries_text_and_logprob() {
    let evt = GenerationEvent::Token {
        candidate_index: 0,
        text: "raven".into(),
        logprob: -1.2,
    };
    match evt {
        GenerationEvent::Token {
            candidate_index,
            text,
            logprob,
        } => {
            assert_eq!(candidate_index, 0);
            assert_eq!(text, "raven");
            assert!((logprob + 1.2).abs() < 1e-6);
        }
        _ => panic!("expected Token"),
    }
}

#[test]
fn generation_event_done_has_finish_reason_and_usage() {
    let evt = GenerationEvent::Done {
        candidate_index: 0,
        finish_reason: FinishReason::Stop,
        usage: UsageStats {
            prompt_tokens: 42,
            generated_tokens: 168,
            elapsed_ms: 1234,
        },
        full_text: "raven are clever birds".into(),
    };
    if let GenerationEvent::Done {
        usage,
        finish_reason,
        ..
    } = evt
    {
        assert_eq!(finish_reason, FinishReason::Stop);
        assert_eq!(usage.prompt_tokens, 42);
        assert_eq!(usage.generated_tokens, 168);
    } else {
        panic!("expected Done");
    }
}
