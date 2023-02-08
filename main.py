# import necessary libraries
from transformers import pipeline, AutoTokenizer, TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, AdamWeightDecay
from bs4 import BeautifulSoup
import requests
import streamlit as st
from datasets import load_dataset, load_metric
import nltk
import numpy as np



def main():
    
    output = ""
    article = ""
    
    # api key
    api_key = "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"
    # caluclating the rouge score for tine fune model
    metric = load_metric("rouge")
    #ROUGE = Rouge()
    
    # fine tune the model. Here we choose the T5-base model
    model_checkpoint = "t5-base"
    
    # set the heading
    st.title("TLDR.ai: Summarize Blogs in 1 Click")

    # add input text field for url
    input_url = st.text_input("Enter URL of a Blog Post", key = "in_url")

    # add input text field for text
    summary_txt = st.text_input("Enter text to summarize", key = "summ_txt")
    
    # add a drop down to select the summarizer model
    summary_options = st.selectbox("Choose Summarizer", ("Select a model", "Distilbart-CNN-12-6", "Finetuned T5-Base"))
    
    
    # addd the prefix to the input text
    if model_checkpoint in ["t5-base"]:
        prefix = "summarize: "
    else:
        prefix = ""
    
    # fine tune the T5 base model
    res = fine_tune_custom_model( prefix, model_checkpoint)
    # left, right = st.columns(2)
    # with left:
    #     summ_clicked = st.button('Summarize')
    # with right:
    #     clear_clicked = st.button('Clear')
    

    # submit button
    with st.form(key='my_form'):
        submit_button = st.form_submit_button(label='Summarize')
        #clear = st.form_submit_button(label="Clear", on_click = clear_text)
    
    
    # if clear:
    #     st.write('Cleared')
        
    # if submit button is clicked   
    if submit_button:
        
        # loading spinner for the summarization process
        with st.spinner(
                text = f"Creating extractive summary using {summary_options}. This might take a few seconds ..."
            ):
            
            if not input_url and not summary_txt:
                new_title = '<p style="font-family:sans-serif; color:Red; font-size: 36px;">Please enter Input</p>'
                st.markdown(new_title, unsafe_allow_html=True)
                #st.write("Please enter a valid URL")
                
            elif input_url :
                   
                # apply data preprocessing once we have the url
                chunks, web_title, article = data_preprocess_from_url(requests.get(input_url))
        
                # execute the summarize model
                output = run_model(chunks, summary_options)
            
                # set the title of the blog post
                st.write("Title: ", web_title)
                

            
            # if the input is text and not URL
            elif not input_url and summary_txt:
                # apply data preprocessing once we have the url
                chunks, article = data_preprocess_text_pipeline(summary_txt)
                
                 # execute the summarize model
                output = run_model(chunks, summary_options)
            
            
        
        # display the summary in the streamlit app
        col1, col2 = st.columns(2)

        with col1:
            st.header("Full Article")
            if article:
                st.write(article)
            else:
                st.write("Error pringting article")

        with col2:
            st.header("Summary")
            if output:
                st.write(output)
            else:
                st.write("Output empty. Please try again")
        
        pred_lst = list(article.split())
        lable_lst = list(output.split())
        print(metric.compute(predictions=pred_lst, references=lable_lst))
        
        #print(ROUGE.get_scores(str(article), str(output)))
            
                    
    
    
    # if submit_button and summary_options == "Finetuned T5-Base" :
    #      with st.spinner(
    #             text = f"Creating extractive summary using {summary_options}. This might take a few seconds ..."
    #         ):      
             
    #          if not input_url:
    #             new_title = '<p style="font-family:sans-serif; color:Red; font-size: 36px;">Input text invalid. Please enter text</p>'
    #             st.markdown(new_title, unsafe_allow_html=True)
    #             #st.write("Please enter a valid URL")
                
    #          elif input_url:
                   
    #             # apply data preprocessing once we have the url
    #             chunks, web_title = data_preprocess_from_url(requests.get(input_url))
        
    #             # execute the HuggingFace model
    #             output = run_model(chunks, summary_options)
            
    #             # set the title of the blog post
    #             st.write(web_title)
            
    #          # if the input is text and not URL
    #          else:
                 
    #              # apply data preprocessing once we have the url
    #              chunks = data_preprocess_text_pipeline(summary_txt)
                
    #              # execute the HuggingFace model
    #              output = run_model(chunks, summary_options)
                
                
             

def data_preprocess_from_url(data):
    """This method is used to preprocess the text from the URL link before passing it to the summarizer model

    Args:
        data (str): url link

    Returns:
       chunks, article str, str_: returns the llist of chunks and the entire article
    """
    
    soup = BeautifulSoup(data.text, 'html.parser')
    # extract h1 and p HTML tags
    results = soup.find_all(['h1', 'p'])
    # get the title of the blog post
    web_title = soup.title.get_text()
    # remove the h1, p tags and get the text
    text = [result.text for result in results]
    # convert to string
    ARTICLE = ' '.join(text)
    
    # create chunks to pass to summarizer nmodels
    chunks = generate_chunks(ARTICLE)
    
    return (chunks, web_title, ARTICLE)


def data_preprocess_text_pipeline(data):
    """This method is used to preprocess the text before passing it to the summarizer model

    Args:
        text (str): input text

    Returns:
       chunks, article str, str_: returns the llist of chunks and the entire article
    """
    article = ''.join(data)
    chunks = generate_chunks(data)
    
    return (chunks, article)

    
def run_model(chunks, summarizer_name):
    """This method is used to run the summarizer model

    Args:
        chunks (str): list of data
        summarizer_name (string): name of the summarzation model to execute

    Returns:
       output string: summary of the article
    """
    # run the huggingface summarizer
    if summarizer_name == "Distilbart-CNN-12-6":
        summarizer = pipeline("summarization")
        res = summarizer(chunks, max_length=120, min_length=30, do_sample=False)
        output = ' '.join([summ['summary_text'] for summ in res])
        return output
    
    # run fine tuned model
    if summarizer_name == "Finetuned T5-Base":
        summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")
        res = summarizer(chunks, max_length=120, min_length=30, do_sample=False)
        output = ' '.join([summ['summary_text'] for summ in res])
        return output

def fine_tune_custom_model(prefix, model_checkpoint):
    pass

def generate_chunks(ARTICLE):
    """this method is used to generate chunks of text to pass to summarizer model

    Args:
        ARTICLE (_type_): _description_

    Returns:
       chunks str: list of sentences
    """
    # set the chunk size limit
    max_chunk = 500
    
    ARTICLE = ARTICLE.replace('.', '.<eos>')
    ARTICLE = ARTICLE.replace('?', '?<eos>')
    ARTICLE = ARTICLE.replace('!', '!<eos>')
    
    sentences = ARTICLE.split('<eos>')
    current_chunk = 0 
    chunks = []
    for sentence in sentences:
        if len(chunks) == current_chunk + 1: 
            if len(chunks[current_chunk]) + len(sentence.split(' ')) <= max_chunk:
                chunks[current_chunk].extend(sentence.split(' '))
            else:
                current_chunk += 1
                chunks.append(sentence.split(' '))
        else:
            print(current_chunk)
            chunks.append(sentence.split(' '))

    for chunk_id in range(len(chunks)):
        chunks[chunk_id] = ' '.join(chunks[chunk_id])
    
    # return th list of chunks
    return chunks


def metric_fn(metric, tokenizer, eval_predictions):
    """This function i used to evaluate the summary

    Args:
        tokenizer (_type_): token
        eval_predictions (_type_): _description_

    Returns:
        _type_: _description_
    """
    predictions, labels = eval_predictions
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    for label in labels:
        label[label < 0] = tokenizer.pad_token_id  # Replace masked label tokens
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Rouge expects a newline after each sentence
    decoded_predictions = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_predictions
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
    ]
    result = metric.compute(
        predictions=decoded_predictions, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    # Add mean generated length
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)

    return result



def preprocess_function(prefix, tokenizer, examples, max_input_length, max_target_length):
    """This funtion is used to convert the text to tokenized format

    Args:
        prefix (_type_): _description_
        tokenizer (_type_): _description_
        examples (_type_): _description_
        max_input_length (_type_): _description_
        max_target_length (_type_): _description_

    Returns:
        _type_: _description_
    """
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["summary"], max_length=max_target_length, truncation=True
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def clear_text():
    st.session_state["in_url"] = ""
    st.session_state["summ_txt"] = ""
    

def fine_tune_custom_model1(model_checkpoint):
    """This function is used to fine tune the summarizer model
    """
    batch_size = 8
    learning_rate = 2e-5
    weight_decay = 0.01
    num_train_epochs = 1
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    # load the custom dataset
    raw_datasets = load_dataset("csv", data_files="medium-articles.csv")
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")
    generation_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf", pad_to_multiple_of=128)
    # preprocess the data  
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    # add the model check point
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    # create train test split
    train_dataset = model.prepare_tf_dataset(
    tokenized_datasets["train"],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_collator,
    )

    validation_dataset = model.prepare_tf_dataset(
        tokenized_datasets["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,)

    generation_dataset = model.prepare_tf_dataset(
        tokenized_datasets["validation"],
        batch_size=8,
        shuffle=False,
        collate_fn=generation_data_collator
    )
    # adam optimizer
    optimizer = AdamWeightDecay(learning_rate=learning_rate, weight_decay_rate=weight_decay)
    # add the optimizer to the model
    model.compile(optimizer=optimizer) 
    
    # train the custom model
    model.fit(train_dataset, validation_data=validation_dataset, epochs=1)
    
    # save the model name
    model = TFAutoModelForSeq2SeqLM.from_pretrained("rohit/medium_articler_t5_summarizer")

if __name__ == "__main__":
    main()







URL = "https://medium.com/nlp-bits/what-is-natural-language-processing-1b0905feaeb3"

text = """In This Article we will explore and understand what is natural language processing and how is NLP helpful in the real world. NLP is used in the world all around us. 
A good way to gain a deeper understanding is by exploring the problems that NLP solves and then categorizing their task in NLP. Let’s start by understanding some of the fundamentals of NLP.  
Natural language processing is an area of computer science and artificial intelligence. That involves the study of human language through algorithms. 
It’s a combination of computational linguistic, machine learning, and deep learning models.  The goal of NLP is to ‘understand’ language’s whole meaning along with the complete intent and sentiment. 
Some popular applications are Amazon Alexa, Apple voice assistant, Siri, and tasks such as machine translation you have seen in Google Translate.  
Let’s explore some of the problems solved by NLP. NLP has helped us solve some real-world problem. 
Let’s go over some of them. First, let’s start by language modeling and classification. Language modeling involves the task of predicting the sequence of words to be used in a sentence based on how it was used in the past.  This is used for tasks such as speech recognition, machine translation, etc. On the other hand, classification in NLP is used to bucket similar objects like sentence, paragraphs or documents together in tasks such as classifying spam emails or identifying different sentiments etc.  Next is information extraction and retrieval. Information extraction and retrieval involves the task of finding and retrieving relevant information. This can be broken down into two components. One is to extract information such as to the people mentioned in an email, and the second is to retrieve information based on query from collection of data.  This is done by search engines such as Google and Bing. The next is conversational agents, commonly referred to as chatbots. Based on our understanding of language, dialogue systems can be built to interact with users to understand their intent and provide them with the relevant information.  Virtual assistants such as Alexa and Google Assistant are some great examples of conversational agents. The next is machine translation. Machine translation includes the translation of text from one language to another. This is done by applications such as Google Translate for both voice as well as text, where, for instance, English can be translated to Spanish. The last is topic modeling.  When a large collection of information is present, topic modeling involves a task of figuring out the underlining topical structure where a user can understand the important topic being discussed. Google News is one great example where algorithm understands the topic of each article and clusters similar article based on similar topic. Next we will look at some of the application in NLP bucketed by difficulty.  A good way to bucket NLP tasks can be categorize them from easy to hard. 
Let’s explore the reasoning behind the categorization of what makes NLP so interesting to use in the real world. 
Starting with easy, tasks such as spell check, simple keyword-based information retrieval system, and topic modeling, are the tasks that are more rule-based and hence not ambiguous.  
They are relatively easy tasks in the NLP world. The next is medium where tasks such as text classification, information extraction, and closed domain chatbots are intermediately difficult, as this involves multiple rules that needs to be conditionally modified based on the scenario.
For instance, in sentiment analysis, some words mean entirely different when considered as two words."""