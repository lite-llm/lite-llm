/// Standalone skeleton crate for Lite LLM.
///
/// This crate intentionally starts with a minimal public API.
pub struct LiteLlm;

impl LiteLlm {
    /// Creates a new Lite LLM skeleton instance.
    pub fn new() -> Self {
        Self
    }

    /// Returns the component name for smoke validation.
    pub fn name(&self) -> &'static str {
        "lite-llm"
    }
}

impl Default for LiteLlm {
    fn default() -> Self {
        Self::new()
    }
}
