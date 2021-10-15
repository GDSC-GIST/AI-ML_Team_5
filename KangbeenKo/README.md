# Kangbeen's Study Note

## <a href = "https://www.coursera.org/learn/natural-language-processing-tensorflow?specialization=tensorflow-in-practice">Coursera: Natural Language Processing in TensorFlow</a>

### Week 1: Sentiment in text (Last Update: 21.10.15 Fri)

The first step in understanding sentiment in text, and in particular when training a neural network to do so is the tokenization of that text. </br>

This is the process of converting the text into numeric values, with a number representing a word or a character. </br>

This week you'll learn about the Tokenizer and `pad_sequences` APIs in TensorFlow and how they can be used to prepare and encode text and sentences to get them ready for training neural networks!

- Weekly Meeting #1 Questions
    - Tokenizer에서 Hyper Parameter `num_words`가 의미하는 바가 무엇인가?

        <a href= "https://stackoverflow.com/questions/64158898/what-does-keras-tokenizer-num-words-specify">[What does Keras Tokenizer num_words specify?]</a>
    - 최신버전(v2)의 JSON 파일에서는 각 열이 comma로 분리되어 있지 않아 `json.load()`에서 에러가 발생한다. 이는 데이터셋의 문제인가?
    
        <a href="https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection/discussion/167722">[Parsing Error using Python]</a>

----

\* Additional Things
- You need to add your account to the organization repository to log the commit.
