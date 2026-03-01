use std::error::Error;
use std::fmt;

pub type LiteLlmResult<T> = Result<T, LiteLlmError>;

#[derive(Debug)]
pub enum LiteLlmError {
    ContractDrift(&'static str),
    InvalidProfile(&'static str),
    InvalidMode(&'static str),
    Runtime(lite_llm_runtime::RuntimeError),
    Distributed(lite_llm_distributed::DistributedError),
    Storage(lite_llm_storage::StorageError),
    Training(lite_llm_training::TrainingError),
    Inference(lite_llm_inference::InferenceError),
    Security(lite_llm_security::SecurityError),
}

impl fmt::Display for LiteLlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContractDrift(msg) => write!(f, "contract drift detected: {msg}"),
            Self::InvalidProfile(msg) => write!(f, "invalid startup profile: {msg}"),
            Self::InvalidMode(msg) => write!(f, "invalid mode request: {msg}"),
            Self::Runtime(err) => write!(f, "runtime error: {err}"),
            Self::Distributed(err) => write!(f, "distributed error: {err}"),
            Self::Storage(err) => write!(f, "storage error: {err}"),
            Self::Training(err) => write!(f, "training error: {err}"),
            Self::Inference(err) => write!(f, "inference error: {err}"),
            Self::Security(err) => write!(f, "security error: {err}"),
        }
    }
}

impl Error for LiteLlmError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Runtime(err) => Some(err),
            Self::Distributed(err) => Some(err),
            Self::Storage(err) => Some(err),
            Self::Training(err) => Some(err),
            Self::Inference(err) => Some(err),
            Self::Security(err) => Some(err),
            Self::ContractDrift(_) | Self::InvalidProfile(_) | Self::InvalidMode(_) => None,
        }
    }
}

impl From<lite_llm_runtime::RuntimeError> for LiteLlmError {
    fn from(value: lite_llm_runtime::RuntimeError) -> Self {
        Self::Runtime(value)
    }
}

impl From<lite_llm_distributed::DistributedError> for LiteLlmError {
    fn from(value: lite_llm_distributed::DistributedError) -> Self {
        Self::Distributed(value)
    }
}

impl From<lite_llm_storage::StorageError> for LiteLlmError {
    fn from(value: lite_llm_storage::StorageError) -> Self {
        Self::Storage(value)
    }
}

impl From<lite_llm_training::TrainingError> for LiteLlmError {
    fn from(value: lite_llm_training::TrainingError) -> Self {
        Self::Training(value)
    }
}

impl From<lite_llm_inference::InferenceError> for LiteLlmError {
    fn from(value: lite_llm_inference::InferenceError) -> Self {
        Self::Inference(value)
    }
}

impl From<lite_llm_security::SecurityError> for LiteLlmError {
    fn from(value: lite_llm_security::SecurityError) -> Self {
        Self::Security(value)
    }
}
