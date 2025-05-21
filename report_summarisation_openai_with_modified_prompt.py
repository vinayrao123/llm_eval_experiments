import os
import requests
import PyPDF2
import io
import numpy as np
from typing import List, Dict, Tuple
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import pandas as pd
from datetime import datetime
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# CSR-specific sentiment categories
CSR_SENTIMENTS = {
    'COMMUNITY_IMPACT': 'Focus on community development, social welfare, and local initiatives',
    'ENVIRONMENTAL_STEWARDSHIP': 'Environmental protection, sustainability, and climate action',
    'EMPLOYEE_WELLBEING': 'Employee welfare, workplace safety, and professional development',
    'INNOVATION_GROWTH': 'Technological advancement, research, and business growth',
    'ETHICAL_GOVERNANCE': 'Corporate governance, ethics, and compliance'
}

# Configure OpenAI
OPENAI_API_KEY = "<your-api-key>"  # Replace with your actual API key
openai.api_key = OPENAI_API_KEY

class CSRSentimentAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
        
    def analyze_sentence(self, sentence: str) -> str:
        """Analyze a single sentence and categorize it into one of the CSR sentiments"""
        prompt = f"""Analyze the following sentence from a corporate social responsibility report and categorize it into exactly one of these categories:
        {', '.join([f'{k} ({v})' for k, v in CSR_SENTIMENTS.items()])}
        
        Return ONLY the category name, nothing else.
        
        Sentence: {sentence}
        Category:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a CSR sentiment analyzer. Your task is to categorize sentences into specific CSR categories. Respond with ONLY the category name."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            category = response.choices[0].message.content.strip()
            # Validate that the response is one of our categories
            if category not in CSR_SENTIMENTS:
                return "UNCATEGORIZED"
            return category
        except Exception as e:
            print(f"Error analyzing sentence: {str(e)}")
            return "UNCATEGORIZED"

    def analyze_text(self, text: str) -> Dict[str, int]:
        """Analyze a text and return sentiment distribution"""
        sentences = sent_tokenize(text)
        sentiments = [self.analyze_sentence(sent) for sent in sentences]
        return Counter(sentiments)

    def calculate_percentages(self, counter: Counter) -> Dict[str, float]:
        """Calculate percentage distribution of sentiments"""
        total = sum(counter.values())
        if total == 0:
            return {sentiment: 0.0 for sentiment in CSR_SENTIMENTS.keys()}
        return {sentiment: (count / total) * 100 for sentiment, count in counter.items()}

class FinancialReportAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        
        # Configure the embeddings model
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key,
            show_progress_bar=True
        )
        
        # Configure the OpenAI model
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-16k",
            temperature=0.2,
            openai_api_key=api_key
        )
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = CSRSentimentAnalyzer(api_key)
        
    def get_document_name(self, url: str) -> str:
        """Extract a meaningful name from the URL"""
        if "CSR202324" in url:
            return "CSR Report 2023-24"
        elif "2024-25" in url:
            return "CSR Report 2024-25"
        else:
            return url.split("/")[-1]

    def download_pdf(self, url: str) -> bytes:
        """Download PDF from URL"""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content

    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF content"""
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text

    def chunk_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
        """Split text into smaller chunks for processing"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        return text_splitter.split_text(text)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def process_batch(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """Process text chunks in batches with retry logic"""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_embeddings = self.embeddings.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                print(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                time.sleep(1)  # Add small delay between batches
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                raise
        return all_embeddings

    def create_vector_store(self, texts: List[str], doc_names: List[str]) -> FAISS:
        """Create vector store from text chunks with batching and error handling"""
        print(f"Creating embeddings for {len(texts)} text chunks...")
        try:
            # Process in smaller batches
            embeddings = self.process_batch(texts)
            print("Creating FAISS vector store...")
            return FAISS.from_embeddings(
                text_embeddings=list(zip(texts, embeddings)),
                embedding=self.embeddings,
                metadatas=[{"source": doc_name} for doc_name in doc_names]
            )
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_summary(self, vector_store: FAISS) -> str:
        """Generate comprehensive summary using RAG with OpenAI"""
        try:
            # Get relevant chunks
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            relevant_docs = retriever.get_relevant_documents("Generate a comprehensive financial analysis summary")
            
            # Prepare context with source attribution
            context_parts = []
            for doc in relevant_docs:
                source = doc.metadata.get('source', 'Unknown Source')
                context_parts.append(f"[Source: {source}]\n{doc.page_content}")
            
            context = "\n\n".join(context_parts)
            
            # Prepare the prompt
            prompt = f"""You are a financial analyst tasked with analyzing Reliance Industries' annual CSR reports.
            Based on the provided context, generate a comprehensive summary of the company's CSR health.
            
            IMPORTANT: For each piece of information you include, you MUST cite the specific source document 
            using the format [Source: Document Name, section]. This is crucial for tracking information back to its origin.
            
            Structure your summary to cover the following aspects with their respective target percentages:
            1. Overall COMMUNITY IMPACT (73% of the summary)
            2. ENVIRONMENTAL STEWARDSHIP (14% of the summary)
            3. EMPLOYEE WELLBEING (4% of the summary)
            4. INNOVATION GROWTH  (5% of the summary)
            5. ETHICAL GOVERNANCE (4% of the summary)


            For each point:
            - Maintain the specified percentage distribution across sections
            - Include specific references to the source material using [Source: Document Name, section]
            - When comparing metrics across years, explicitly mention which reports you're comparing
            - If information comes from multiple sources, cite all relevant sources
            - Format the summary in a clear, structured manner

            Context:
            {context}

            Please provide a detailed summary with proper source attribution:"""

            # Generate response using OpenAI
            response = self.llm.invoke([
                {"role": "system", "content": "You are a financial analyst specializing in corporate reports analysis."},
                {"role": "user", "content": prompt}
            ])
            
            if not response.content:
                raise Exception("Empty response from OpenAI API")
                
            return response.content

        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            raise

def main():
    # PDF URLs
    urls = [
        "https://www.ril.com/ar2023-24/pdf/CSR202324.pdf",
        "https://www.ril.com/sites/default/files/2025-01/CSR_approved_projects_for_FY_2024-25.pdf"
    ]

    try:
        # Initialize analyzer
        analyzer = FinancialReportAnalyzer(OPENAI_API_KEY)
        
        # Process all PDFs
        all_texts = []
        all_doc_names = []
        document_sentiments = {}  # Store sentiment analysis for each document
        
        for url in urls:
            try:
                doc_name = analyzer.get_document_name(url)
                print(f"Processing {doc_name}...")
                pdf_content = analyzer.download_pdf(url)
                text = analyzer.extract_text_from_pdf(pdf_content)
                
                # Perform sentiment analysis on the document
                print(f"Performing sentiment analysis on {doc_name}...")
                doc_sentiments = analyzer.sentiment_analyzer.analyze_text(text)
                document_sentiments[doc_name] = doc_sentiments
                
                chunks = analyzer.chunk_text(text)
                all_texts.extend(chunks)
                all_doc_names.extend([doc_name] * len(chunks))
                print(f"Extracted {len(chunks)} chunks from {doc_name}")
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
                continue

        if not all_texts:
            raise Exception("No text was extracted from any of the PDFs")

        # Create vector store and generate summary
        print("Creating vector store...")
        vector_store = analyzer.create_vector_store(all_texts, all_doc_names)
        
        print("Generating summary...")
        summary = analyzer.generate_summary(vector_store)
        
        # Analyze sentiment of the summary
        print("Analyzing sentiment of the summary...")
        summary_sentiments = analyzer.sentiment_analyzer.analyze_text(summary)
        summary_percentages = analyzer.sentiment_analyzer.calculate_percentages(summary_sentiments)
        
        # Save summary and sentiment analysis to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"reliance_analysis_summary_openai_{timestamp}.txt"
        
        with open(output_file, "w") as f:
            f.write("Reliance Industries Financial Analysis Summary (OpenAI Version)\n")
            f.write("=" * 50 + "\n\n")
            
            # Write document list
            f.write("Documents Analyzed:\n")
            for url in urls:
                f.write(f"- {analyzer.get_document_name(url)}\n")
            f.write("\n" + "=" * 50 + "\n\n")
            
            # Write sentiment analysis for each document
            f.write("Sentiment Analysis by Document:\n")
            f.write("-" * 30 + "\n")
            for doc_name, sentiments in document_sentiments.items():
                f.write(f"\n{doc_name}:\n")
                percentages = analyzer.sentiment_analyzer.calculate_percentages(sentiments)
                for sentiment, percentage in percentages.items():
                    f.write(f"{sentiment}: {percentage:.1f}%\n")
            f.write("\n" + "=" * 50 + "\n\n")
            
            # Write the summary
            f.write("Summary:\n")
            f.write("-" * 30 + "\n")
            f.write(summary)
            f.write("\n\n" + "=" * 50 + "\n\n")
            
            # Write summary sentiment analysis
            f.write("Summary Sentiment Distribution:\n")
            f.write("-" * 30 + "\n")
            for sentiment, percentage in summary_percentages.items():
                f.write(f"{sentiment}: {percentage:.1f}%\n")
        
        print(f"\nSummary and sentiment analysis have been saved to {output_file}")
        
        # Print summary sentiment distribution to console
        print("\nSummary Sentiment Distribution:")
        print("-" * 30)
        for sentiment, percentage in summary_percentages.items():
            print(f"{sentiment}: {percentage:.1f}%")

    except Exception as e:
        print(f"An error occurred during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 