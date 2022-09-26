# BlogSummarizer

### Aim: 

We are developing a web-based tool to summarize different medium blogs.


### Motivation:

We live in an era where we don't have time to read or go through the lengthy content, so we prefer short and explicit content. However, having quick content doesn't mean you ignore the essential points, and preserving all of these crucial points can be difficult when you summarize.
So, we aim to solve this by developing a summarization tool to generate content that will help folks to quickly understand any topic they wish to learn


### Dataset:

We will be using the medium articles from different blogs like Towards data science, Hackernoon to generate the dataset. We will scrape the website for the last 5 months. This gives us around 500+ articles.

#### Tech stack:
- Development: Python
- Web scraping: Beautifulsoup
- Models : Transformer, T5, T5 long
- Model Deployment: Streamlit


### Experimental Plan:

We will follow the complete life cycle of a data science project from gathering data through web scraping, cleaning, tokenizing and utilizing Huggingface transformers, T5, T5 Long models to generate the text summarization and compare the results.  The best performing model will be deployed. We will use streamlit to develop the web application to display the summarized text. 


### Anticipated challenges:

Web scraping will be a challenging task
Larger models might need more resources to run
