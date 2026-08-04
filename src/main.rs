mod model;
mod ingestion;
mod retrieval;

use anyhow::Result;
use model::RawDocument;
use ingestion::IngestionPipeline;
use uuid::Uuid;
use retrieval::RetrievalPipeline;
use dotenvy::dotenv;
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    let collection_name = "rust_rag_chunks";
    let qdrant_url = "http://localhost:6334";

    let pipeline = IngestionPipeline::new(qdrant_url, collection_name).await?;

    let sample_doc = RawDocument {
        id: Uuid::new_v4(),
        title: "SD-WAN".to_string(),
        source: "docs/sd_wan_intro.md".to_string(),
        content: "SD-WAN (Software-Defined Wide Area Network) is a virtual technology that uses software to manage and smart-route network traffic across different locations. It replaces old, expensive private lines with a mix of regular internet, 5G, and secure links to boost speed and lower costs.".to_string(),
    };

    println!("Ingesting document...");
    pipeline.ingest_document(sample_doc).await?;
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

    println!("Generating answer with local LLM...");
    let answer = retrieval_pipeline.generate_answer_groq(&augmented_prompt, &api_key).await?;
    println!("🤖 LLM Answer:\n{}", answer);
    
    Ok(())
}