//! Environment variable support for lite-llm configuration.
//!
//! This module provides utilities to load secrets and configuration from
//! environment variables, allowing tests and applications to avoid hardcoding
//! sensitive values.
//!
//! # Usage
//!
//! ```rust
//! use lite_llm::env::load_test_secrets;
//!
//! // Load secrets from environment variables
//! let secrets = load_test_secrets();
//! println!("Node ID: {:?}", secrets.node_id);
//! ```
//!
//! # Environment Variables
//!
//! - `LITE_LLM_NODE_ID` - Node identifier
//! - `LITE_LLM_SIGNER_ID` - Signer identifier for audit logging
//! - `LITE_LLM_SIGNING_SECRET` - Signing secret (minimum 8 characters)

use std::env;

/// Secrets loaded from environment variables for testing.
///
/// Each field is `None` if the corresponding environment variable is not set.
#[derive(Debug, Clone, Default)]
pub struct TestSecrets {
    /// Node identifier from `LITE_LLM_NODE_ID`
    pub node_id: Option<String>,
    /// Signer identifier from `LITE_LLM_SIGNER_ID`
    pub signer_id: Option<String>,
    /// Signing secret from `LITE_LLM_SIGNING_SECRET`
    pub signing_secret: Option<String>,
}

impl TestSecrets {
    /// Load secrets from environment variables.
    ///
    /// Returns a `TestSecrets` with fields populated from corresponding
    /// environment variables. Fields are `None` if not set.
    pub fn from_env() -> Self {
        Self {
            node_id: env::var("LITE_LLM_NODE_ID").ok(),
            signer_id: env::var("LITE_LLM_SIGNER_ID").ok(),
            signing_secret: env::var("LITE_LLM_SIGNING_SECRET").ok(),
        }
    }

    /// Get the signing secret, returning a default if not set.
    ///
    /// This is useful for tests that need a fallback value.
    pub fn signing_secret_or_default(&self, default: &str) -> String {
        self.signing_secret
            .clone()
            .unwrap_or_else(|| default.to_owned())
    }

    /// Get the node_id, returning a default if not set.
    pub fn node_id_or_default(&self, default: &str) -> String {
        self.node_id.clone().unwrap_or_else(|| default.to_owned())
    }

    /// Get the signer_id, returning a default if not set.
    pub fn signer_id_or_default(&self, default: &str) -> String {
        self.signer_id.clone().unwrap_or_else(|| default.to_owned())
    }
}

/// Load test secrets from environment variables.
///
/// This is a convenience function that wraps `TestSecrets::from_env()`.
pub fn load_test_secrets() -> TestSecrets {
    TestSecrets::from_env()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secrets_default_is_empty() {
        let secrets = TestSecrets::default();
        assert!(secrets.node_id.is_none());
        assert!(secrets.signer_id.is_none());
        assert!(secrets.signing_secret.is_none());
    }

    #[test]
    fn test_or_default_fallback() {
        let secrets = TestSecrets::default();
        assert_eq!(secrets.signing_secret_or_default("fallback"), "fallback");
        assert_eq!(secrets.node_id_or_default("default-node"), "default-node");
        assert_eq!(
            secrets.signer_id_or_default("default-signer"),
            "default-signer"
        );
    }

    #[test]
    fn test_or_default_uses_env_value() {
        // Set environment variables for this test
        std::env::set_var("LITE_LLM_NODE_ID", "env-node");
        std::env::set_var("LITE_LLM_SIGNER_ID", "env-signer");
        std::env::set_var("LITE_LLM_SIGNING_SECRET", "env-secret-123");

        let secrets = TestSecrets::from_env();
        assert_eq!(secrets.node_id_or_default("fallback"), "env-node");
        assert_eq!(secrets.signer_id_or_default("fallback"), "env-signer");
        assert_eq!(
            secrets.signing_secret_or_default("fallback"),
            "env-secret-123"
        );

        // Clean up
        std::env::remove_var("LITE_LLM_NODE_ID");
        std::env::remove_var("LITE_LLM_SIGNER_ID");
        std::env::remove_var("LITE_LLM_SIGNING_SECRET");
    }
}
