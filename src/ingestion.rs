use anyhow::Result;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use qdrant_client::qdrant::{CreateCollectionBuilder, Distance, PointStruct, UpsertPointsBuilder, VectorParamsBuilder};
use qdrant_client::{Payload, Qdrant};
use text_splitter::TextSplitter;
use uuid::Uuid;

use crate::model::{Chunk, RawDocument};

pub struct IngestionPipeline {
    qdrant: Qdrant,
    collection_name: String,
}

impl IngestionPipeline {
    /// Initialize DB client and collection
    pub async fn new(qdrant_url: &str, collection_name: &str) -> Result<Self> {
        let qdrant = Qdrant::from_url(qdrant_url).build()?;

        // Ensure collection exists
        if !qdrant.collection_exists(collection_name).await? {
            let vector_dim = 384; // Dimension size for MiniLM-L6-v2
            qdrant
                .create_collection(
                    CreateCollectionBuilder::new(collection_name).vectors_config(
                        VectorParamsBuilder::new(vector_dim, Distance::Cosine),
                    ),
                )
                .await?;
        }

        Ok(Self {
            qdrant,
            collection_name: collection_name.to_string(),
        })
    }

    /// Process a raw document through chunking, embedding, and indexing
    pub async fn ingest_document(&self, doc: RawDocument) -> Result<()> {
        // Step 1: Chunking
        let splitter = TextSplitter::new(120);
        let mut chunks: Vec<Chunk> = Vec::new();

        for chunk_text in splitter.chunks(&doc.content) {
            chunks.push(Chunk {
                chunk_id: Uuid::new_v4(),
                doc_id: doc.id,
                title: doc.title.clone(),
                text: chunk_text.to_string(),
            });
        }

        // Step 2: Local Embeddings
        let mut model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::AllMiniLML6V2Q).with_show_download_progress(true),
        )?;

        let chunk_texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let vectors = model.embed(chunk_texts, None)?;

        // Step 3: Build & Upsert Points
        let mut points: Vec<PointStruct> = Vec::new();
        for (chunk, vector) in chunks.into_iter().zip(vectors.into_iter()) {
            let payload: Payload = serde_json::json!({
                "doc_id": chunk.doc_id.to_string(),
                "title": chunk.title,
                "text": chunk.text,
            })
            .try_into()?;

            points.push(PointStruct::new(chunk.chunk_id.to_string(), vector, payload));
        }

        self.qdrant
            .upsert_points(UpsertPointsBuilder::new(&self.collection_name, points))
            .await?;

        Ok(())
    }
}