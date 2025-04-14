import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.mistral.modeling_mistral import MistralModel
from typing import List, Dict, Optional
from core.single.pikv_moe import PiKVMoE
from core.single.lora import LoRAPiKVMoE

class MoERAG(nn.Module):
    """
    MoE-based RAG architecture that combines PiKV's KV cache mechanism with RAG's retrieval-augmented generation.
    """
    def __init__(self, 
                 base_model: str = "Salesforce/SFR-Embedding-Mistral",
                 num_experts: int = 4,
                 cache_size: int = 128,
                 use_lora: bool = False,
                 rank: int = 4,
                 alpha: float = 1.0):
        super(MoERAG, self).__init__()
        
        # Initialize base model
        self.base_model = AutoModel.from_pretrained(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Initialize MoE components
        self.use_lora = use_lora
        if use_lora:
            self.moe = LoRAPiKVMoE(rank=rank, alpha=alpha)
        else:
            self.moe = PiKVMoE()
        
        # Cache configuration
        self.cache_size = cache_size
        self.document_cache = {}
        self.query_cache = {}
        
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
    
    def cache_document(self, doc_id: str, doc_embedding: torch.Tensor):
        """Cache document embeddings with importance scores."""
        if doc_id not in self.document_cache:
            self.document_cache[doc_id] = {
                'embedding': doc_embedding.detach(),
                'importance': torch.ones(1, device=doc_embedding.device)
            }
    
    def cache_query(self, query_id: str, query_embedding: torch.Tensor):
        """Cache query embeddings with importance scores."""
        if query_id not in self.query_cache:
            self.query_cache[query_id] = {
                'embedding': query_embedding.detach(),
                'importance': torch.ones(1, device=query_embedding.device)
            }
    
    def retrieve(self, query_embedding: torch.Tensor, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant documents using MoE-based retrieval."""
        # Normalize query embedding
        query_embedding = F.normalize(query_embedding, p=2, dim=1)
        
        # Process through MoE
        moe_output = self.moe(query_embedding)
        
        # Calculate similarity scores
        scores = []
        for doc_id, cache in self.document_cache.items():
            doc_embedding = cache['embedding']
            doc_embedding = F.normalize(doc_embedding, p=2, dim=1)
            
            # Combine base similarity with MoE-enhanced similarity
            base_similarity = (query_embedding @ doc_embedding.T).squeeze()
            moe_similarity = (moe_output @ doc_embedding.T).squeeze()
            
            # Weighted combination of similarities
            final_score = 0.7 * base_similarity + 0.3 * moe_similarity
            scores.append((doc_id, final_score.item()))
        
        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [{'doc_id': doc_id, 'score': score} for doc_id, score in scores[:top_k]]
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                is_query: bool = True,
                doc_id: Optional[str] = None) -> torch.Tensor:
        """Forward pass through the model."""
        # Get base embeddings
        embeddings = self.get_embedding(input_ids, attention_mask)
        
        # Process through MoE
        moe_output = self.moe(embeddings)
        
        # Cache embeddings if doc_id is provided
        if doc_id is not None:
            if is_query:
                self.cache_query(doc_id, embeddings)
            else:
                self.cache_document(doc_id, embeddings)
        
        return moe_output

class MoERAGPipeline:
    """Pipeline for MoE RAG system."""
    def __init__(self, model: MoERAG):
        self.model = model
        self.tokenizer = model.tokenizer
    
    def process_document(self, text: str, doc_id: str):
        """Process and cache a document."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                is_query=False,
                doc_id=doc_id
            )
    
    def process_query(self, query: str, query_id: str, top_k: int = 5):
        """Process a query and retrieve relevant documents."""
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
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