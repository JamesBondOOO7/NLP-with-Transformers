import pandas as pd
from transformers import pipeline

def text_classification(text):
    """
        Predicts sentiment as POSITIVE or NEGATIVE
    Args:
        text (string): input string
    """
    # In HF Transformers, we instantiate a pipeline by calling the pipeline() function and providing the name of the task we are interested in
    classifier = pipeline("text-classification")
    
    # Each pipeline takes a string of text (or a list of strings) as i/p and returns a list of predictions
    outputs = classifier(text)
    print(pd.DataFrame(outputs))
    
    # ***** OUTPUT *****
    #       label       score
    # 0     NEGATIVE    0.901546
    
    
def namedEntityRecognition(text):
    """
        In NLP, real-world objects like products, places, and people are called named entities, and extracting them from text is called named entity recognition (NER).
    
    `ORG` : Organization
    `LOC` : Location
    `PER` : Person
    `MISC`: Miscellaneous
    Args:
        text (string): input string
    """
    
    ner_tagger = pipeline("ner", aggregation_strategy="simple")
    outputs = ner_tagger(text)
    print(pd.DataFrame(outputs))
    
    # ***** OUTPUT *****
    #   entity_group     score           word  start  end
    # 0          ORG  0.879010         Amazon      5   11
    # 1         MISC  0.990859  Optimus Prime     36   49
    # 2          LOC  0.999755        Germany     90   97
    # 3         MISC  0.556570           Mega    208  212
    # 4          PER  0.590256         ##tron    212  216
    # 5          ORG  0.669692         Decept    253  259
    # 6         MISC  0.498349        ##icons    259  264
    # 7         MISC  0.775362       Megatron    350  358
    # 8         MISC  0.987854  Optimus Prime    367  380
    # 9          PER  0.812096      Bumblebee    502  511


def questionAnswering(question, text):
    """
        In question answering, we provide the model with a passage of text called the context, along with a question whose answer we’d like to extract. The model then returns the span of text corresponding to the answer.

    Args:
        question (string):  question to be asked
        text (string):   the text from which the answer is extracted
    """
    
    reader = pipeline("question-answering")
    outputs = reader(question=question, context=text)
    print(pd.DataFrame([outputs]))
    
    # ***** OUTPUT *****
    #       score    start  end                   answer
    # 0  0.642406      335  358  an exchange of Megatron
    

def summarization(text):
    """
        take a long text as input and generate a short version with all
        the relevant facts

    Args:
        text (string): the i/p text
    """
    
    summarizer = pipeline("summarization")
    outputs = summarizer(text, max_length=45, clean_up_tokenization_spaces=True)
    
    print(outputs[0]['summary_text'])
    
    
    #  Bumblebee ordered an Optimus Prime action figure from your online store in Germany. Unfortunately, when I opened the package, I discovered to my horror that I had been sent an action figure of Megatron instead.
    
    
def translation(text):
    """
        translation is a task where the output consists of generated text.
        `English` to `German` translator

    Args:
        text (string): the i/p text
    """
    
    translator = pipeline("translation_en_to_de", 
                          model="Helsinki-NLP/opus-mt-en-de")
    outputs = translator(text, clean_up_tokenization_spaces=True, min_length=100)
    
    print(outputs[0]['translation_text'])
    
    # Sehr geehrter Amazon, letzte Woche habe ich eine Optimus Prime Action Figur aus Ihrem Online-Shop in Deutschland bestellt. Leider, als ich das Paket öffnete, entdeckte ich zu meinem Entsetzen, dass ich stattdessen eine Action Figur von Megatron geschickt worden war! Als lebenslanger Feind der Decepticons, Ich hoffe, Sie können mein Dilemma verstehen. Um das Problem zu lösen, Ich fordere einen Austausch von Megatron für die Optimus Prime Figur habe ich bestellt. Anbei sind Kopien meiner Aufzeichnungen über diesen Kauf. Ich erwarte, bald von Ihnen zu hören. Aufrichtig, Bumblebee.
    

def textGeneration(text):
    """
    provide faster replies to customer feedback by having access to an autocomplete function

    Args:
        text (string): the i/p string
    """
    
    generator = pipeline("text-generation")
    response = "Dear Bumblebee, I am sorry to hear your order was mixed up."
    
    prompt = text + "\n\nCustomer service response:\n" + response
    
    outputs = generator(prompt, max_length=200)
    
    print(outputs[0]['generated_text'])
    
    # Dear Amazon, last week I ordered an Optimus Prime action figure from your online store in Germany. Unfortunately, when I opened the package, I discovered to my horror that I had been sent an action figure of Megatron instead! As a lifelong enemy of the Decepticons, I hope you can understand my dilemma. To resolve the issue, I demand an exchange of Megatron for the Optimus Prime figure I ordered. Enclosed are copies of my records concerning this purchase. I expect to hear from you soon. Sincerely, Bumblebee.

    # Customer service response:
    # Dear Bumblebee, I am sorry to hear your order was mixed up. Please please make me realize that this isn't the Transformers I'm so excited about. It'll be fine that you did order a single action figure. And if you haven't been paying attention, I do appreciate that you chose the wrong brand of product. It is not your fault that my order went wrong, and you did not deliver at
    
    
if __name__ == '__main__':
    
    text = """Dear Amazon, last week I ordered an Optimus Prime action figure from your online store in Germany. Unfortunately, when I opened the package, I discovered to my horror that I had been sent an action figure of Megatron instead! As a lifelong enemy of the Decepticons, I hope you can understand my dilemma. To resolve the issue, I demand an exchange of Megatron for the Optimus Prime figure I ordered. Enclosed are copies of my records concerning this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""
    
    text_classification(text=text)
    
    namedEntityRecognition(text=text)
    
    question = "what does the customer want?"
    questionAnswering(question=question, text=text)
    
    summarization(text=text)
    
    translation(text=text)
    
    textGeneration(text=text)