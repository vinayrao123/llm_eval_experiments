import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io

# Data from the analysis
sentiment_data = {
    'CSR_2023_24': {
        'INNOVATION_GROWTH': 5.0,
        'COMMUNITY_IMPACT': 80.6,
        'ENVIRONMENTAL_STEWARDSHIP': 5.1,
        'ETHICAL_GOVERNANCE': 2.9
        'EMPLOYEE_WELLBEING': 6.3,
        'UNCATEGORIZED': 0.1
    },
    'CSR_2024_25': {
        'ETHICAL_GOVERNANCE': 8.7,
        'COMMUNITY_IMPACT': 67.4,
        'ENVIRONMENTAL_STEWARDSHIP': 19.6,
        'EMPLOYEE_WELLBEING': 2.2,
        'INNOVATION_GROWTH': 2.2
    },
    'Summary_Without_SA': {
        'COMMUNITY_IMPACT': 35.7,
        'ENVIRONMENTAL_STEWARDSHIP': 14.3,
        'EMPLOYEE_WELLBEING': 14.3,
        'INNOVATION_GROWTH': 21.4,
        'ETHICAL_GOVERNANCE': 14.3
    },
    'Summary_With_SA': {
        'COMMUNITY_IMPACT': 62.5,
        'ENVIRONMENTAL_STEWARDSHIP': 18.8,
        'EMPLOYEE_WELLBEING': 6.2,
        'INNOVATION_GROWTH': 6.2,
        'ETHICAL_GOVERNANCE': 6.2
    }
}

def create_bar_chart(data, title, filename):
    plt.figure(figsize=(10, 6))
    sentiments = list(data.keys())
    percentages = list(data.values())
    
    # Create bar chart
    bars = plt.bar(sentiments, percentages)
    
    # Customize the chart
    plt.title(title, pad=20)
    plt.xlabel('Sentiment Categories')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf

def create_expected_distribution():
    # Calculate average distribution from source documents
    all_sentiments = set()
    for doc in ['CSR_2023_24', 'CSR_2024_25']:
        all_sentiments.update(sentiment_data[doc].keys())
    
    expected_dist = {}
    for sentiment in all_sentiments:
        values = []
        for doc in ['CSR_2023_24', 'CSR_2024_25']:
            if sentiment in sentiment_data[doc]:
                values.append(sentiment_data[doc][sentiment])
        expected_dist[sentiment] = sum(values) / len(values)
    
    return expected_dist

def generate_pdf():
    doc = SimpleDocTemplate("sentiment_analysis_results.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    story.append(Paragraph("Sentiment Analysis Results", title_style))
    story.append(Spacer(1, 20))
    
    # Source Document Distributions
    story.append(Paragraph("Source Document Sentiment Distributions", styles['Heading2']))
    story.append(Spacer(1, 12))
    
    # CSR 2023-24
    story.append(Paragraph("CSR Report 2023-24", styles['Heading3']))
    img = Image(create_bar_chart(sentiment_data['CSR_2023_24'], 
                                "CSR Report 2023-24 Sentiment Distribution",
                                "csr_2023_24.png"))
    img.drawHeight = 4*inch
    img.drawWidth = 6*inch
    story.append(img)
    story.append(Spacer(1, 20))
    
    # CSR 2024-25
    story.append(Paragraph("CSR Report 2024-25", styles['Heading3']))
    img = Image(create_bar_chart(sentiment_data['CSR_2024_25'],
                                "CSR Report 2024-25 Sentiment Distribution",
                                "csr_2024_25.png"))
    img.drawHeight = 4*inch
    img.drawWidth = 6*inch
    story.append(img)
    story.append(Spacer(1, 20))
    
    # Expected Distribution
    expected_dist = create_expected_distribution()
    story.append(Paragraph("Expected Average Distribution", styles['Heading2']))
    story.append(Spacer(1, 12))
    img = Image(create_bar_chart(expected_dist,
                                "Expected Average Sentiment Distribution",
                                "expected_dist.png"))
    img.drawHeight = 4*inch
    img.drawWidth = 6*inch
    story.append(img)
    story.append(Spacer(1, 20))
    
    # Summary Results
    story.append(Paragraph("Summary Generation Results", styles['Heading2']))
    story.append(Spacer(1, 12))
    
    # Without Sentiment Analysis
    story.append(Paragraph("Without Sentiment Analysis", styles['Heading3']))
    img = Image(create_bar_chart(sentiment_data['Summary_Without_SA'],
                                "Summary Without Sentiment Analysis",
                                "summary_without_sa.png"))
    img.drawHeight = 4*inch
    img.drawWidth = 6*inch
    story.append(img)
    story.append(Spacer(1, 20))
    
    # With Sentiment Analysis
    story.append(Paragraph("With Sentiment Analysis", styles['Heading3']))
    img = Image(create_bar_chart(sentiment_data['Summary_With_SA'],
                                "Summary With Sentiment Analysis",
                                "summary_with_sa.png"))
    img.drawHeight = 4*inch
    img.drawWidth = 6*inch
    story.append(img)
    
    # Build PDF
    doc.build(story)

if __name__ == "__main__":
    generate_pdf() 