def word_tkn(text, logs=False):
    """
        text to a tokenized list of words

    Args:
        text (list): sentence to be tokenized
    """
    
    tokenized_text = text.split()
    
    # Drawback : doesn't account for punctuation
    # Hence, vocab size can be really huge.
    print(tokenized_text)

if __name__ == '__main__':
    
    text = "Tokenizing text is a core task of NLP."
    word_tkn(text, logs=True)