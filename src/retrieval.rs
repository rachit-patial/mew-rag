use crate::LlmCache;
use anyhow::Result;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::SearchPointsBuilder;
use serde_json::json;

pub struct RetrievalPipeline {
    qdrant: Qdrant,
    collection_name: String,
    embedding_model: TextEmbedding,
}

impl RetrievalPipeline {
    pub async fn new(qdrant_url: &str, collection_name: &str) -> Result<Self> {
        let qdrant = Qdrant::from_url(qdrant_url).build()?;

        let embedding_model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::AllMiniLML6V2Q).with_show_download_progress(true),
        )?;

        Ok(Self {
            qdrant,
            collection_name: collection_name.to_string(),
            embedding_model,
        })
    }

    pub async fn retrieve_context(&mut self, query: &str, top_k: u64) -> Result<Vec<String>> {
        let query_vector = self
            .embedding_model
            .embed(vec![query.to_string()], None)?
            .remove(0);

        let search_result = self
            .qdrant
            .search_points(
                SearchPointsBuilder::new(&self.collection_name, query_vector, top_k)
                    .with_payload(true),
            )
            .await?;

        let mut retrieved_docs = Vec::new();
        for point in search_result.result {
            if let Some(text_value) = point.payload.get("text") {
                if let Some(text_str) = text_value.as_str() {
                    retrieved_docs.push(text_str.to_string());
                }
            }
        }

        Ok(retrieved_docs)
    }

    pub fn build_augmented_prompt(&self, query: &str, context_chunks: &[String]) -> String {
        let context = context_chunks.join("\n----\n");

        format!(
            "You are a helpful assistant. Use ONLY the following context snippets to answer the user question.\n\n\
            ### CONTEXT:\n{}\n\n\
            ### QUESTION:\n{}\n\n\
            ### ANSWER:",
            context, query
        )
    }

    pub async fn generate_answer_groq(
        &self,
        prompt: &str,
        api_key: &str,
        cache: &LlmCache,
    ) -> Result<String> {
        if let Some(cached_answer) = cache.get(prompt)? {
            println!("[Cache Hit] Returning cached response!");
            return Ok(cached_answer);
        }

        println!("[Cache Miss] Calling the LLM...");

        let client = reqwest::Client::new();

        let response = client
            .post("https://api.groq.com/openai/v1/chat/completions")
            .bearer_auth(api_key)
            .json(&json!({
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.2
            }))
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        let answer = response["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("No response returned.")
            .to_string();

        cache.set(prompt, &answer)?;

        Ok(answer)
    }
}
