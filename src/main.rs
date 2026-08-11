mod cache;
mod ingestion;
mod model;
mod retrieval;

use anyhow::Result;
use cache::LlmCache;
use dotenvy::dotenv;
use ingestion::IngestionPipeline;
use model::RawDocument;
use retrieval::RetrievalPipeline;
use std::env;
use std::fs;

fn load_documents_from_json(
    file_path: &str,
) -> Result<Vec<RawDocument>, Box<dyn std::error::Error + Send + Sync>> {
    let data = fs::read_to_string(file_path)?;
    let docs: Vec<RawDocument> = serde_json::from_str(&data)?;
    Ok(docs)
}

#[tokio::main]
async fn main() -> Result<()> {
    let collection_name = "rust_rag_chunks";
    let qdrant_url = "http://localhost:6334";

    let pipeline = IngestionPipeline::new(qdrant_url, collection_name).await?;

    let sample_doc =
        load_documents_from_json("document.json").map_err(|e| anyhow::anyhow!("{}", e))?;

    println!("Ingesting {} document...", sample_doc.len());

    for doc in sample_doc {
        println!("Processing: {}", doc.title);
        pipeline.ingest_document(doc).await?;
    }
    println!("Document ingested successfully!");

    println!("\n🔎 Initializing Retrieval Pipeline...");
    let mut retrieval_pipeline = RetrievalPipeline::new(qdrant_url, collection_name).await?;

    let user_query = "How does Rust free memory without a garbage collector?";
    println!("? Query: \"{}\"", user_query);

    let context_chunks = retrieval_pipeline.retrieve_context(user_query, 2).await?;
    println!("Retrieved {} context chunks.", context_chunks.len());

    let augmented_prompt = retrieval_pipeline.build_augmented_prompt(user_query, &context_chunks);
    println!("\nPrompt Constructed:\n{}\n", augmented_prompt);

    dotenv().ok();

    let api_key = env::var("GROQ_API_KEY").expect("API Key not found");

    let cache = LlmCache::new("llm_cache")?;
    println!("Generating answer with local LLM...");
    let answer = retrieval_pipeline
        .generate_answer_groq(&augmented_prompt, &api_key, &cache)
        .await?;
    println!("🤖 LLM Answer:\n{}", answer);

    Ok(())
}
