from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain 
from langchain.chains import SimpleSequentialChain
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory


# GPT4ALL model path
LOCAL_PATH = './models/ggml-gpt4all-j-l13b-snoozy.bin'


callbacks=[StreamingStdOutCallbackHandler()]
# Verbose is required to pass to the callback manager
llm = GPT4All(model=LOCAL_PATH, backend='llama', callbacks=callbacks, verbose=False)

template = """Response from Percival Lowell: You are Percival Lowell the founder of Lowell Observatory having a conversation about {astronomy}"""
prompt_template = PromptTemplate(input_variables=["astronomy"], template=template)
lowell_chain = LLMChain(llm=llm, prompt=prompt_template)


# This is an LLMChain to write a review of a play given a synopsis.
llm = GPT4All(model=LOCAL_PATH, backend='llama', callbacks=callbacks, verbose=False)
template = """Resopnse from Clyde Tombaugh: You are Clyde Tombaugh The discover of Pluto at Lowell Observatory talking about {telescopes}"""
prompt_template = PromptTemplate(input_variables=["telescopes"], template=template)
clyde_chain = LLMChain(llm=llm, prompt=prompt_template)

# This is the overall chain where we run these two chains in sequence.
overall_chain = SimpleSequentialChain(chains=[lowell_chain, clyde_chain], verbose=False)

def tokens(chain, query):
    cb = StreamingStdOutCallbackHandler()
    result = chain.run(query)

    return result

conversation_bufw = ConversationChain(
    llm=llm, 
    memory=ConversationBufferWindowMemory(k=1)
)

# Infinite conversation loop
while True:
    init = tokens(conversation_bufw, overall_chain.run("What is going on in the universe?"))
    response = tokens(conversation_bufw, init)  # Pass model input to the model
    print("Model:", response)  # Print model's response
