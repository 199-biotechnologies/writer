use std::collections::HashSet;
use writer_cli::corpus::ingest;
use writer_cli::corpus::normalize;
use writer_cli::corpus::sample::{Sample, SampleMetadata, SampleSource};
use writer_cli::corpus::sources::markdown::MarkdownSource;
use writer_cli::corpus::sources::obsidian::{ObsidianSource, strip_wikilinks};
use writer_cli::corpus::sources::plain_text::PlainTextSource;
use writer_cli::corpus::sources::{Source, SourceRegistry};

#[test]
fn markdown_strips_front_matter_and_splits_headers() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("test.md");
    std::fs::write(
        &file,
        "---\ntitle: Test\n---\n\n# Section One\n\nHello world this is section one content.\n\n# Section Two\n\nThis is section two with enough words to pass the filter easily.\n",
    )
    .unwrap();

    let source = MarkdownSource;
    assert!(source.matches(&file));
    let samples = source.parse(&file, None).unwrap();
    assert_eq!(samples.len(), 2);
    assert_eq!(
        samples[0].metadata.context_tag.as_deref(),
        Some("Section One")
    );
    assert_eq!(
        samples[1].metadata.context_tag.as_deref(),
        Some("Section Two")
    );
    assert!(!samples[0].content.contains("title: Test"));
}

#[test]
fn plain_text_strips_bom_and_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("test.txt");
    std::fs::write(
        &file,
        "\u{FEFF}Hello world this is a test.\n\nAnother paragraph here with more content.",
    )
    .unwrap();

    let source = PlainTextSource;
    assert!(source.matches(&file));
    let samples = source.parse(&file, Some("essay")).unwrap();
    assert!(!samples.is_empty());
    assert!(!samples[0].content.contains('\u{FEFF}'));
    assert_eq!(samples[0].metadata.source, SampleSource::PlainText);
}

#[test]
fn obsidian_strips_wikilinks() {
    assert_eq!(
        strip_wikilinks("See [[my note]] for details"),
        "See my note for details"
    );
    assert_eq!(
        strip_wikilinks("See [[path/to/note|display text]]"),
        "See display text"
    );
    assert_eq!(strip_wikilinks("No links here"), "No links here");
}

#[test]
fn obsidian_parses_vault() {
    let dir = tempfile::tempdir().unwrap();
    let obsidian_dir = dir.path().join(".obsidian");
    std::fs::create_dir(&obsidian_dir).unwrap();
    std::fs::write(
        dir.path().join("note1.md"),
        "# My Note\n\nThis is a regular note with [[wikilinks]] and enough words to pass filters.\n",
    ).unwrap();
    std::fs::write(
        dir.path().join("note2.md"),
        "---\ntags: [daily]\n---\n\nToday I worked on the project and made good progress with several items.\n",
    ).unwrap();

    let source = ObsidianSource;
    assert!(source.matches(dir.path()));
    let samples = source.parse(dir.path(), None).unwrap();
    assert!(samples.len() >= 2);

    let journal = samples
        .iter()
        .find(|s| s.metadata.context_tag.as_deref() == Some("journal"));
    assert!(journal.is_some(), "should detect daily note");

    let notes = samples.iter().find(|s| {
        s.metadata.context_tag.as_deref() == Some("notes")
            || s.metadata.context_tag.as_deref() == Some("My Note")
    });
    assert!(notes.is_some(), "should have regular notes");
    assert!(!notes.unwrap().content.contains("[["));
}

#[test]
fn source_registry_detects_correct_source() {
    let registry = SourceRegistry::default_set();

    let md = std::path::Path::new("/tmp/test.md");
    assert_eq!(registry.detect(md).unwrap().name(), "markdown");

    let txt = std::path::Path::new("/tmp/test.txt");
    assert_eq!(registry.detect(txt).unwrap().name(), "plain_text");
}

#[test]
fn normalize_cleans_sample() {
    let sample = Sample::new(
        "Hello\u{200B} world\n\nBest regards\n-- \nBoris".into(),
        SampleMetadata::default(),
    );
    let cleaned = normalize::clean(sample);
    assert!(!cleaned.content.contains('\u{200B}'));
    assert!(!cleaned.content.contains("Boris"));
}

#[test]
fn ingest_pipeline_processes_directory() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("note.md"),
        "# Title\n\nSome content here that is long enough to be a valid writing sample.\n",
    )
    .unwrap();

    let (samples, report) = ingest::ingest(
        &[dir.path().to_path_buf()],
        Some("test"),
        4096,
        &HashSet::new(),
        true,
    )
    .unwrap();

    assert!(!samples.is_empty());
    assert!(report.samples_added > 0);
    assert!(report.total_words > 0);
}

#[test]
fn dedupe_skips_duplicates() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("note.md");
    std::fs::write(
        &file,
        "# Title\n\nThis is some content that should appear only once in the final corpus.\n",
    )
    .unwrap();

    let (samples1, _) = ingest::ingest(
        &[dir.path().to_path_buf()],
        None,
        4096,
        &HashSet::new(),
        true,
    )
    .unwrap();

    let existing: HashSet<String> = samples1.iter().map(|s| s.content_hash.clone()).collect();

    let (samples2, report2) =
        ingest::ingest(&[dir.path().to_path_buf()], None, 4096, &existing, true).unwrap();

    assert_eq!(samples2.len(), 0);
    assert!(report2.samples_skipped_dedupe > 0);
}
