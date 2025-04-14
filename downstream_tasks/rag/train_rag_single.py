import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

import torch
from downstream_tasks.rag.moe_rag_single import MoERAG, MoERAGPipeline

def main():
    # 初始化模型
    model = MoERAG(
        base_model="Salesforce/SFR-Embedding-Mistral",
        num_experts=4,
        cache_size=128,
        use_lora=True,
        rank=4,
        alpha=1.0
    )
    
    # 创建处理管道
    pipeline = MoERAGPipeline(model)
    
    # 示例文档
    documents = [
        ("doc1", "To bake a delicious chocolate cake, you'll need the following ingredients: all-purpose flour, sugar, cocoa powder, baking powder, baking soda, salt, eggs, milk, vegetable oil, and vanilla extract. Start by preheating your oven to 350°F (175°C). In a mixing bowl, combine the dry ingredients (flour, sugar, cocoa powder, baking powder, baking soda, and salt). In a separate bowl, whisk together the wet ingredients (eggs, milk, vegetable oil, and vanilla extract). Gradually add the wet mixture to the dry ingredients, stirring until well combined. Pour the batter into a greased cake pan and bake for 30-35 minutes. Let it cool before frosting with your favorite chocolate frosting. Enjoy your homemade chocolate cake!"),
        ("doc2", "The flu, or influenza, is an illness caused by influenza viruses. Common symptoms of the flu include a high fever, chills, cough, sore throat, runny or stuffy nose, body aches, headache, fatigue, and sometimes nausea and vomiting. These symptoms can come on suddenly and are usually more severe than the common cold. It's important to get plenty of rest, stay hydrated, and consult a healthcare professional if you suspect you have the flu. In some cases, antiviral medications can help alleviate symptoms and reduce the duration of the illness."),
        ("doc3", "Python is a high-level, interpreted programming language known for its simplicity and readability. It supports multiple programming paradigms, including procedural, object-oriented, and functional programming. Python has a comprehensive standard library and a vast ecosystem of third-party packages. It's widely used in web development, data science, artificial intelligence, and automation. The language's design philosophy emphasizes code readability with its notable use of significant whitespace."),
        ("doc4", "Machine learning is a subset of artificial intelligence that focuses on building systems that can learn from and make decisions based on data. It involves the use of algorithms to parse data, learn from it, and then make predictions or decisions. Common types of machine learning include supervised learning, unsupervised learning, and reinforcement learning. Applications range from image recognition and natural language processing to recommendation systems and predictive analytics.")
    ]
    
    # 处理文档
    print("Processing documents...")
    for doc_id, text in documents:
        pipeline.process_document(text, doc_id)
        print(f"Processed document: {doc_id}")
    
    # 示例查询
    queries = [
        ("query1", "How to make a chocolate cake?"),
        ("query2", "What are the symptoms of flu?"),
        ("query3", "What is Python programming?"),
        ("query4", "What is machine learning?")
    ]
    
    # 处理查询并获取结果
    print("\nProcessing queries and retrieving results...")
    for query_id, query in queries:
        results = pipeline.process_query(query, query_id, top_k=2)
        print(f"\nQuery: {query}")
        print("Top results:")
        for result in results:
            print(f"  Document ID: {result['doc_id']}, Score: {result['score']:.4f}")

if __name__ == "__main__":
    main() 