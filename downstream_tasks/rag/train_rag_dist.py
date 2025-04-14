import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from moe_rag_dist import DistributedMoERAG, DistributedMoERAGPipeline
from core.distributed.distributed_config import distributed_config as dconfig

def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = dconfig['dist_url'].split('://')[1].split(':')[0]
    os.environ['MASTER_PORT'] = dconfig['dist_url'].split(':')[-1]
    dist.init_process_group(
        backend=dconfig['dist_backend'],
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def train(rank, world_size):
    """Training function for each process."""
    setup(rank, world_size)
    
    # Initialize model
    model = DistributedMoERAG(
        base_model="Salesforce/SFR-Embedding-Mistral",
        num_experts=4,
        cache_size=128,
        use_lora=True,
        lora_rank=4,
        alpha=1.0
    )
    
    # Move model to current device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    
    # Initialize pipeline
    pipeline = DistributedMoERAGPipeline(model.module)  # Use model.module to access the actual model
    
    # Example documents
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question."
    ]
    
    # Process documents
    for i, doc in enumerate(documents):
        pipeline.process_document(doc, f"doc_{i}")
    
    # Example query
    query = "What is the meaning of life?"
    results = pipeline.process_query(query, "query_1", top_k=2)
    
    # Print results from rank 0
    if rank == 0:
        print("Query:", query)
        print("Top results:")
        for result in results:
            print(f"Document {result['doc_id']}: Score {result['score']:.4f}")
    
    cleanup()

if __name__ == "__main__":
    world_size = dconfig['world_size']
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True) 