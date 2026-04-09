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
