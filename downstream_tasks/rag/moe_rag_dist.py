import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.mistral.modeling_mistral import MistralModel
from typing import List, Dict, Optional
from core.distributed.distributed_pikv import DistributedPiKVMoE, DistributedPiKVManager
from core.distributed.distributed_config import distributed_config as dconfig

class DistributedMoERAG(nn.Module):
    """
    Distributed MoE-based RAG architecture that combines PiKV's KV cache mechanism with RAG's retrieval-augmented generation.
    """
    def __init__(self, 
                 base_model: str = "Salesforce/SFR-Embedding-Mistral",
                 num_experts: int = 4,
                 cache_size: int = 128,
                 use_lora: bool = False,
                 lora_rank: int = 4,
                 alpha: float = 1.0):
        super(DistributedMoERAG, self).__init__()
        
        # Initialize distributed environment
        self.world_size = dconfig['world_size']
        self.rank = dconfig['rank']
        
        # Initialize base model
        self.base_model = AutoModel.from_pretrained(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Get embedding dimension from base model
        self.embedding_dim = self.base_model.config.hidden_size
        
        # Initialize projection layer to match MoE input dimension
        self.projection = nn.Linear(self.embedding_dim, 256)  # 256 is the hidden_size in config
        
        # Initialize MoE components using distributed implementation
        self.moe = DistributedPiKVMoE()
        
        # Cache configuration
        self.cache_size = cache_size
        self.document_cache = {}
        self.query_cache = {}
        
        # Initialize distributed KV cache
        self._init_distributed_cache()
    
    def _init_distributed_cache(self):
        """Initialize distributed KV cache."""
        # Each rank maintains its own cache
        self.local_cache_size = self.cache_size // self.world_size
        self.cache_offset = self.rank * self.local_cache_size
        
    def get_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get embeddings from the base model."""
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.last_token_pool(outputs.last_hidden_state, attention_mask)
        return embeddings
    
    def last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool the last token's hidden state."""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def project_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Project embedding to MoE input dimension."""
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        # Ensure input dimension is correct
        if embedding.size(-1) != self.embedding_dim:
            embedding = embedding.view(-1, self.embedding_dim)
        return self.projection(embedding)
    
    def cache_document(self, doc_id: str, doc_embedding: torch.Tensor):
        """Cache document embeddings with importance scores."""
        if doc_id not in self.document_cache:
            # Project document embedding to match query embedding dimension
            projected_embedding = self.project_embedding(doc_embedding)
            self.document_cache[doc_id] = {
                'embedding': projected_embedding.detach(),
                'importance': torch.ones(1, device=doc_embedding.device)
            }
    
    def cache_query(self, query_id: str, query_embedding: torch.Tensor):
        """Cache query embeddings with importance scores."""
        if query_id not in self.query_cache:
            # Project query embedding to match document embedding dimension
            projected_embedding = self.project_embedding(query_embedding)
            self.query_cache[query_id] = {
                'embedding': projected_embedding.detach(),
                'importance': torch.ones(1, device=query_embedding.device)
            }
    
    def compute_similarity(self, query_embedding: torch.Tensor, doc_embedding: torch.Tensor) -> torch.Tensor:
        """Compute similarity between query and document embeddings."""
        # Ensure both embeddings are in the same dimension
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        if doc_embedding.dim() == 1:
            doc_embedding = doc_embedding.unsqueeze(0)
        
        # Normalize embeddings
        query_embedding = F.normalize(query_embedding, p=2, dim=1)
        doc_embedding = F.normalize(doc_embedding, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.mm(query_embedding, doc_embedding.t()).squeeze()
        return similarity
    
    def retrieve(self, query_embedding: torch.Tensor, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant documents using MoE-based retrieval."""
        # Project query embedding to match document embedding dimension
        projected_query = self.project_embedding(query_embedding)
        
        # Process through MoE
        moe_output = self.moe(projected_query)
        
        # Calculate similarity scores
        scores = []
        for doc_id, cache in self.document_cache.items():
            doc_embedding = cache['embedding']
            
            # Compute base similarity
            base_similarity = self.compute_similarity(projected_query, doc_embedding)
            
            # Compute MoE-enhanced similarity
            moe_similarity = self.compute_similarity(moe_output, doc_embedding)
            
            # Weighted combination of similarities
            final_score = 0.7 * base_similarity + 0.3 * moe_similarity
            scores.append((doc_id, final_score.item()))
        
        # Gather scores from all ranks
        all_scores = [None for _ in range(self.world_size)]
        dist.all_gather_object(all_scores, scores)
        
        # Combine scores from all ranks
        combined_scores = []
        for rank_scores in all_scores:
            if rank_scores is not None:  # Add check for None
                combined_scores.extend(rank_scores)
        
        # Sort by score and return top-k
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        return [{'doc_id': doc_id, 'score': score} for doc_id, score in combined_scores[:top_k]]
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                is_query: bool = True,
                doc_id: Optional[str] = None) -> torch.Tensor:
        """Forward pass through the model."""
        # Get base embeddings
        embeddings = self.get_embedding(input_ids, attention_mask)
        
        # Project embeddings to match MoE input dimension
        projected_embeddings = self.project_embedding(embeddings)
        
        # Process through MoE
        moe_output = self.moe(projected_embeddings)
        
        # Cache embeddings if doc_id is provided
        if doc_id is not None:
            if is_query:
                self.cache_query(doc_id, embeddings)
            else:
                self.cache_document(doc_id, embeddings)
        
        return moe_output

class DistributedMoERAGPipeline:
    """Pipeline for distributed MoE RAG system."""
    def __init__(self, model: DistributedMoERAG):
        self.model = model
        self.tokenizer = model.tokenizer
    
    def process_document(self, text: str, doc_id: str):
        """Process and cache a document."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        with torch.no_grad():
            self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                is_query=False,
                doc_id=doc_id
            )
    
    def process_query(self, query: str, query_id: str, top_k: int = 5):
        """Process a query and retrieve relevant documents."""
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        with torch.no_grad():
            query_embedding = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                is_query=True,
                doc_id=query_id
            )
        
        # Retrieve relevant documents
        results = self.model.retrieve(query_embedding, top_k=top_k)
        return results 