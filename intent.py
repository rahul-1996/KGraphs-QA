import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import copy

module_url = '../intent_model'

def embed_useT(module):
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})

embed_fn = embed_useT(module_url)

messages = [
    'what are symptoms of Problem',
    'what is problem caused by disease?',
    'what are the side effects of treatment',
    'What medications are best for the treatment of problem?',
    'what diseases are cured by treatment',
    'what problem is a test conducted for',
    'what tests are conducted for Problem',
    'What are the signs that I have Problem?',
    'is problem a symptom of problem',
    'what happens when I take treatment',
    'I have problem, what treatment do I take?',
    'is treatment helpful for problem?',
    'what are the indications that I have problem?',
    'I have problem. Do I have problem?',
    'Can treatment cause problem?',
    'I am suffering from problem. does treatment cure it?',
    'Does treatment cure problem?',
    'My doctor told me I experience problem. Do I have problem?',
    'I have problem. Is medicine the correct treatment?',
    'Does treatment cure problem?',
    'What all problems does a test reveal?',
    'What problem causes problem?',
    'What is problem a symptom of?',
    'What are the treatments for problem?'
]

intents = [
    'disease_symptom',
    'symptom_disease',
    'treatment_side_effects',
    'disease_treatment',
    'treatment_disease',
    'test_problem',
    'problem_test',
    'disease_symptom',
    'symptom_disease',
    'treatment_side_effects',
    'disease_treatment',
    'treatment_disease',
    'disease_symptom',
    'symptom_disease',
    'treatment_side_effects',
    'disease_treatment',
    'treatment_disease',
    'symptom_disease',
    'disease_treatment',
    'treatment_disease',
    'test_problem',
    'disease_symptom',
    'symptom_disease',
    'disease_treatment'
]


def get_intent(message):
    mess = copy.copy(messages)
    mess.insert(0, message)
    encoding_matrix = embed_fn(mess)
    product = np.inner(encoding_matrix, encoding_matrix)
    index = np.argmax(product[0][1:], axis=0)
    return intents[index]

if __name__ =='__main__':
    message = "what are symptoms of Problem"
    print(get_intent(message))
