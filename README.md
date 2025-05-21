# Reliance Industries CSR Report Analyzer

This project uses RAG (Retrieval-Augmented Generation) with Google's Gemini API to analyze Reliance Industries' financial reports and generate comprehensive summaries.

## Features

- Downloads and processes PDF financial reports
- Extracts and chunks text for efficient processing
- Uses FAISS vector store for semantic search
- Generates comprehensive summaries using Gemini/OpenAI API
- Includes source references in the analysis
- Saves results to timestamped output files

## Setup Instructions

1. Clone this repository to your Google Colab environment
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Replace `YOUR_API_KEY` in the report_*.py with your actual Google Gemini API key or your OpenAI key
4. Run the script:
   ```bash
   python3.11 report_summarisation.py
   ```

## Output

The script will generate a summary file named `reliance_analysis_summary_YYYYMMDD_HHMMSS.txt` containing:
- Overall financial performance
- Key financial metrics and trends
- Major business developments
- Risk factors and challenges
- Future outlook

Each section includes references to the source material.

## Requirements

- Python 3.8+
- Google Gemini API key
- OpenAI API key
- Internet connection for downloading PDFs
- Sufficient disk space for processing PDFs

## Notes

- The script processes multiple years of annual reports
- Processing time may vary depending on the size of the reports
- Make sure you have sufficient memory in your Colab environment
- The script includes error handling for PDF downloads and processing 

## Sentiment Analysis Hypothesis and Results

### Hypothesis
The sentiment analysis of each line of each document around 5 identified sentiments (Innovation & Growth, Community Impact, Environmental Stewardship, Ethical Governance, and Employee Wellbeing) yields a percentage distribution of each sentiment in each document. The generated summary should reflect at least the average distribution of these sentiments present in the source documents.

### Source Document Sentiment Distribution

#### CSR Report 2023-24
- INNOVATION_GROWTH: 5.0%
- COMMUNITY_IMPACT: 80.6%
- ENVIRONMENTAL_STEWARDSHIP: 5.1%
- ETHICAL_GOVERNANCE: 2.9%
- EMPLOYEE_WELLBEING: 6.3%
- UNCATEGORIZED: 0.1%

#### CSR Report 2024-25
- ETHICAL_GOVERNANCE: 8.7%
- COMMUNITY_IMPACT: 67.4%
- ENVIRONMENTAL_STEWARDSHIP: 19.6%
- EMPLOYEE_WELLBEING: 2.2%
- INNOVATION_GROWTH: 2.2%

### Summary Generation Results

#### Without Sentiment Analysis Input
Summary Sentiment Distribution:
- COMMUNITY_IMPACT: 35.7%
- ENVIRONMENTAL_STEWARDSHIP: 14.3%
- EMPLOYEE_WELLBEING: 14.3%
- INNOVATION_GROWTH: 21.4%
- ETHICAL_GOVERNANCE: 14.3%

#### With Sentiment Analysis-Based Prompt Improvement
Summary Sentiment Distribution:
- COMMUNITY_IMPACT: 62.5%
- ENVIRONMENTAL_STEWARDSHIP: 18.8%
- EMPLOYEE_WELLBEING: 6.2%
- INNOVATION_GROWTH: 6.2%
- ETHICAL_GOVERNANCE: 6.2%

### Findings
The results demonstrate that incorporating sentiment analysis into the prompt engineering process significantly improved the alignment between the source document sentiment distribution and the generated summary. The improved version shows a much closer match to the source documents' emphasis on Community Impact, which was the dominant sentiment in both source documents. 