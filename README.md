# Reddit Comment Summarization and Sentiment Analysis

## Problem

The exponential growth of digital conversations on platforms like Reddit has created a demand for effective tools that distill the essence of discussions from voluminous comment threads. Users often struggle to navigate through extensive text data, hindering their ability to extract key insights and sentiment trends. The challenge is to develop a solution that can automatically summarize Reddit comments and provide an overview of the overall sentiment, facilitating rapid comprehension and analysis of discussions.

## Data Acquisition

To address the problem, the solution utilizes the PRAW API, a Python wrapper for the Reddit API, to scrape relevant comment data from Reddit posts. This raw textual data serves as the foundation for subsequent analysis and summarization.

## Model

The core of the solution lies in the choice of NLP models. The BART (Bidirectional and Auto-Regressive Transformers) model, known for its sequence-to-sequence capabilities, was selected for its prowess in text summarization tasks. BART combines the strengths of GPT (Generative Pre-trained Transformer) with elements of denoising autoencoders, making it an ideal candidate for generating concise and coherent summaries of lengthy comment threads. Additionally, users are empowered to choose between the BART model and OpenAI's Davinci model, accessible through the OpenAI API. This choice caters to users with diverse needs and resources, offering flexibility in selecting a summarization approach that aligns with their preferences.

## Solution Framework

The developed solution encompasses a user-friendly interface built using Streamlit, a Python library for creating interactive web applications. Users input the name of a subreddit, triggering the application to display the top posts of the subreddit. The user can then choose a model for the summarization task on one of the posts. The generated summary encapsulates the main points and key insights of the comment thread, allowing users to rapidly grasp the discussion's essence. Expanding beyond summarization, the solution incorporates sentiment analysis. By employing sentiment analysis techniques on the comment data, the application provides users with a holistic view of the discussion's emotional tone and sentiment trends, enhancing their understanding of the overall context.
