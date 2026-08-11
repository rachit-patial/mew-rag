use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize)]
pub struct RawDocument {
    #[serde(default = "Uuid::new_v4")]
    pub id: Uuid,
    pub title: String,
    pub source: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Chunk {
    pub chunk_id: Uuid,
    pub doc_id: Uuid,
    pub title: String,
    pub text: String,
}
