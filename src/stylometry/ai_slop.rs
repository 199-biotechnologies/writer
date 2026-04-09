//! AI-slop word and phrase lists.
//!
//! Words and phrases characteristic of LLM-generated text that rarely
//! appear in authentic human writing.

pub const BANNED_WORDS: &[&str] = &[
    "delve", "tapestry", "landscape", "leverage", "nuance", "multifaceted",
    "holistic", "pivotal", "realm", "foster", "encompass", "underscore",
    "embark", "navigate", "robust", "vibrant", "comprehensive", "paramount",
    "intricate", "resonate", "furthermore", "moreover", "nevertheless",
    "harnessing", "synergy", "paradigm", "ecosystem", "streamline",
    "cutting-edge", "innovative", "game-changer", "groundbreaking",
    "transformative", "revolutionize", "spearhead", "catalyze", "galvanize",
    "facilitate", "bolster", "augment", "exacerbate", "juxtapose",
    "illuminate", "elucidate", "delineate", "unpack",
    "unravel", "dovetail", "interplay", "symbiosis", "confluence",
    "cornerstone", "linchpin", "bedrock", "underpinning", "scaffolding",
    "blueprint", "roadmap", "trajectory", "endeavor", "endeavour",
    "myriad", "plethora", "diverse", "proactive", "stakeholder",
    "optimize", "utilize", "implement", "framework",
];

pub const BANNED_PHRASES: &[&str] = &[
    "deep dive", "double-edged sword", "at the end of the day",
    "it's worth noting", "in today's world", "in the realm of",
    "it is important to note", "this is a testament to",
    "a testament to", "the landscape of", "the tapestry of",
    "in the ever-evolving", "it cannot be overstated",
    "serves as a reminder", "it is crucial to", "plays a pivotal role",
    "a multifaceted approach", "a holistic approach",
    "the intersection of", "when it comes to",
    "at its core", "by the same token", "in light of",
    "a nuanced understanding", "a comprehensive overview",
    "this underscores the", "it bears mentioning",
    "the broader implications", "a paradigm shift",
];
