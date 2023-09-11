import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from functions import get_top_posts, get_hot_posts, get_controversial_posts,  get_comments, preprocessed, davinci_model, bart_model, sentiment_analysis

st.set_option('deprecation.showPyplotGlobalUse', False)

# Initialize session state variables
if 'name_subreddit' not in st.session_state:
    st.session_state.name_subreddit = None

if 'posts' not in st.session_state:
    st.session_state.posts = None

if 'post_number' not in st.session_state:
    st.session_state.post_number = None

if 'model_selected' not in st.session_state:
    st.session_state.model_selected = None

if 'api_key_entered' not in st.session_state:
    st.session_state.api_key_entered = False

summary = ""

# App title and description
st.write("""
# Sentiment Analysis and Summarization of Reddit Comments Web App
This app **summarizes comments and analyzes sentiments of a particular subreddit post**!
""")

# Add a sidebar for user actions
sidebar_selection = st.sidebar.radio("Select an option:", ("Top Posts", "Controversial Posts", "Hot Posts", "Input Post Link"))

# Top Posts section
if sidebar_selection == "Top Posts":
    name_subreddit = st.text_input("Enter the name of the subreddit")
    submit_button = st.button("Submit")

    # Check if the "Submit" button is clicked and save state
    if submit_button and name_subreddit:
        st.session_state.name_subreddit = name_subreddit
        # Assuming get_top_posts() returns top posts based on subreddit name
        st.session_state.posts = get_top_posts(name_subreddit)

elif sidebar_selection == "Controversial Posts":
    name_subreddit = st.text_input("Enter the name of the subreddit")
    submit_button = st.button("Submit")

    # Check if the "Submit" button is clicked and save state
    if submit_button and name_subreddit:
        st.session_state.name_subreddit = name_subreddit
        # Assuming get_top_posts() returns top posts based on subreddit name
        st.session_state.posts = get_controversial_posts(name_subreddit)

elif sidebar_selection == "Hot Posts":
    name_subreddit = st.text_input("Enter the name of the subreddit")
    submit_button = st.button("Submit")

    # Check if the "Submit" button is clicked and save state
    if submit_button and name_subreddit:
        st.session_state.name_subreddit = name_subreddit
        # Assuming get_top_posts() returns top posts based on subreddit name
        st.session_state.posts = get_hot_posts(name_subreddit)

else:
    post_link = st.text_input("Enter the post link")
    submit_button = st.button("Submit")

    # Check if the "Submit" button is clicked and save state
    #if submit_button and post_link:


# Display top ten posts
if st.session_state.posts is not None:
    st.write("Top Ten posts")
    for i, post_title in enumerate(st.session_state.posts['title'][:10], start=0):
        st.write(f"{i}. {post_title}")

# Check if posts are available and the user has submitted a post number
if st.session_state.posts is not None and st.session_state.name_subreddit:
    # Check if the post number has been submitted
    if st.session_state.post_number is None:
        # Get the post number from user input
        post_number = st.number_input("Which of the posts would you like to get more information on?", value=1)

        # Add a "Submit Post Number" button
        submit_post_number_button = st.button("Submit Post Number")

        # Check if the "Submit Post Number" button is clicked and save state
        if submit_post_number_button:
            st.session_state.post_number = post_number
            st.session_state.model_selected = None  # Reset model selection
            st.session_state.api_key_entered = False  # Reset API key entered flag

    else:
        # Get comments based on chosen post number
        title, comments_list = get_comments(st.session_state.posts, st.session_state.post_number)

        chunks, prompt_template, doc_store = preprocessed(comments_list)

        # Check if the model has been selected
        if st.session_state.model_selected is None:
            model = st.radio(
                "Select which model to use, if you have an openAI Key, you can select the Davinci model "
                "else select the open source model",
                ('Davinci', 'Bart'))

            # Add a "Submit Model" button
            submit_model_button = st.button("Submit Model")

            # Check if the "Submit Model" button is clicked and save state
            if submit_model_button:
                st.session_state.model_selected = model
                st.session_state.api_key_entered = False  # Reset API key entered flag

        else:
            model = st.session_state.model_selected

            if model == 'Davinci':
                if not st.session_state.api_key_entered:
                    OPENAI_API_KEY = st.text_input("Enter your OpenAI API key", type='password')
                    submit_key_button = st.button("Submit Key")

                    if submit_key_button:
                        st.session_state.api_key_entered = True

                if st.session_state.api_key_entered:
                    summary = davinci_model(chunks, prompt_template, doc_store, OPENAI_API_KEY)
                    st.write(summary)
            elif model == 'Bart':
                summary = bart_model(chunks, prompt_template, doc_store)
                st.write(summary)

    # Add a "Pick Another Post" button
    if st.session_state.post_number is not None:
        pick_another_button = st.button("Pick Another Post")

        # Check if the "Pick Another Post" button is clicked and reset post_number
        if pick_another_button:
            st.session_state.post_number = None
            st.session_state.model_selected = None
            st.session_state.api_key_entered = False

# Button to trigger sentiment analysis
if st.session_state.model_selected:
    sentiment_button = st.button("Analyze Sentiment")
    if sentiment_button:
        df = sentiment_analysis(comments_list)

        # Display the sentiment distribution chart using Streamlit
        st.write("Sentiment Distribution:")
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x='sentiment', color='blue')

        # Display values on top of the bars
        for p in plt.gca().patches:
            plt.gca().annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center',
                               va='bottom')

        # Set plot labels and title
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.title('Sentiment Distribution')

        st.pyplot(plt)





