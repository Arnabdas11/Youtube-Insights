from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import streamlit_scrollable_textbox as stx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable import RunnableLambda, RunnableParallel, RunnablePassthrough

class TubeInsights:
    def __init__(self, model, str_parser):
        self.model = model
        self.str_parser = str_parser
    
    def main(self, tube_url):
        
        if tube_url:
            st.video(tube_url)
            tube_id = tube_url.split('?')[1][2:]
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(tube_id, languages=['en'])
                transcript = " ".join([i['text'] for i in transcript_list])

            except TranscriptsDisabled:
                return "No Transcript"
            except Exception as e:
                print(f"Exception occurred: {e}")
                return ""
            else:
                return transcript

    
    def summarize(self, transcript):
        prompt = PromptTemplate(
            template="Summarize the given youtube video based on this transcript: {transcript}",
            input_variables=["transcript"]
        )

        summary_chain = prompt | self.model | self.str_parser
        summary = summary_chain.invoke({'transcript':transcript})

        return summary


    def key_takeaways(self, transcript):
        prompt = PromptTemplate(
            template="Extract the key takeways from the video {transcript} and return as points with heading",
            input_variables=["transcript"]
        )

        key_takeaways_chain = prompt | self.model | self.str_parser
        key_takeaways = key_takeaways_chain.invoke({'transcript':transcript})

        return key_takeaways

    def summary(self, transcript):
        prompt = PromptTemplate(
            template="Provide a summary of the entire video {transcript}",
            input_variables=['transcript']
        )

        summary_chain = prompt | self.model | self.str_parser
        summary = summary_chain.invoke({'transcript':transcript})

        return summary
    
    def format_docs(self, retrieved_docs):
        context_txt = '\n\n'.join(doc.page_content for doc in retrieved_docs)
        return context_txt

    def chat_box(self, transcript, query):
        # indexing
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.from_documents(chunks, embedding=embeddings)
        retriever = vector_store.as_retriever(search_type='mmr', search_kwargs={"k":3,"lambda_mult":0.5})

        retrieval_chain = RunnableParallel({
            'context' : retriever | RunnableLambda(self.format_docs),
            'query':RunnablePassthrough()
        })
        prompt = PromptTemplate(
            template=""" You are a helpful assistant. Answer the user's query:{query} only and only from the given context: {context}. Don't explicitly mention that you are answering from the transcribe. If the query is not related to context then mention that the query is not related to the content and politely refuse""",
            input_variables=['query','context']
        )
        chat_chain = retrieval_chain | prompt | self.model | self.str_parser
        llm_output = chat_chain.invoke(query)
        return llm_output
    
def toggle_button_state():
    st.session_state.button_clicked = not st.session_state.button_clicked


if __name__=='__main__':
    load_dotenv()
    model = ChatOpenAI(model="gpt-4o-mini")
    str_parser = StrOutputParser()
    tubeinsights = TubeInsights(model, str_parser)
    st.set_page_config(page_title="YouTube Insights - v1")
    st.header("YouTube Insights - v1")
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False
    

    tube_url = st.text_input("Enter youtube url here:")
    
    if tube_url:
        transcript = tubeinsights.main(tube_url)
        
        c1, c2, c3, c4= st.columns(4)
        with c1:
            transcript_btn = st.button("Transcript")
        
        with c2:
            key_takeaways_btn = st.button("Key Takeaways")
        
        with c3:
            summary_btn = st.button("Summary")
        
        with c4:
            chat_btn = st.button("Chat",on_click=toggle_button_state)
        

        if transcript_btn:
            stx.scrollableTextbox(transcript,height = 400)
        elif key_takeaways_btn:
            key_takeaways = tubeinsights.key_takeaways(transcript)
            st.markdown(key_takeaways)
        elif summary_btn:
            summary = tubeinsights.summary(transcript)
            st.markdown(summary)
        elif chat_btn or st.session_state.button_clicked:
            query = st.text_input("Enter your query about the video here:")
            if query:
                result = tubeinsights.chat_box(transcript, query)
                st.markdown(result)
            else:
                print("no query")
            
        




    
