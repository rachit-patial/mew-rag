use anyhow::Result;
use hex;
use sha2::{Digest, Sha256};
use sled::Db;

pub struct LlmCache {
    db: Db,
}

impl LlmCache {
    pub fn new(storage_path: &str) -> Result<Self> {
        let db = sled::open(storage_path)?;
        Ok(Self { db })
    }

    fn hash_key(augmented_prompt: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(augmented_prompt.as_bytes());
        hex::encode(hasher.finalize())
    }

    pub fn get(&self, augmented_prompt: &str) -> Result<Option<String>> {
        let key = Self::hash_key(augmented_prompt);
        if let Some(val) = self.db.get(key)? {
            Ok(Some(String::from_utf8(val.to_vec())?))
        } else {
            Ok(None)
        }
    }

    pub fn set(&self, augmented_prompt: &str, answer: &str) -> Result<()> {
        let key = Self::hash_key(augmented_prompt);
        self.db.insert(key, answer.as_bytes())?;
        self.db.flush()?;
        Ok(())
    }
}
