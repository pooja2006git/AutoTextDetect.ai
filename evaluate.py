# from transformers import pipeline

# # Load the model
# model = pipeline("text2text-generation", model="google/flan-t5-small")

# # Read the student's answer from a text file
# with open("student_answer.txt", "r", encoding="utf-8") as file:
#     student_answer = file.read().strip()

# # Debugging step: Print the student's answer
# print("DEBUG: Student's Answer Loaded:\n", student_answer, "\n")

# # Define the evaluation prompt
# prompt = f"""
# You are a strict but fair *English teacher* evaluating a student's answer. Assess it based on these criteria:

# 1 . *Word Limit Check (2 Marks)**  
#    - If *below 40 words*, deduct **1 mark**.  
#    - If *above 60 words*, deduct **1 mark**.  

# 2 .**Grammar & Language (5 Marks)**  
#    - Deduct **1 mark** for each major grammatical mistake (tense errors, incorrect sentence structure).  
#    - Deduct **0.5 marks** for minor spelling or punctuation errors.  

# 3 .**Relevance & Clarity (3 Marks)**  
#    - If the passage is *off-topic or unclear*, deduct up to **3 marks**.  

# ---
# 📄 *Student's Answer:*  
# "{student_answer}"

#  **Expected Output Format:**  
# Total Score: X/10  
# Word Count: XX (⚠ Note if below/above limit)  
# Mistakes Found:  
# - (List mistakes and suggestions)  

# Teacher's Feedback:  
# (A brief comment on strengths & weaknesses)

# Now, evaluate and provide the final result in this exact format.
# """

# # Get the model's response
# response = model(prompt, max_length=300)

# # Extract the generated text
# evaluation_result = response[0]['generated_text']

# # Print the evaluation result
# print("\n **Evaluation Result:**\n")
# print(evaluation_result)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build the model
model = Sequential([
    Flatten(input_shape=(28, 28)),      # Flatten 28x28 images into 784 features
    Dense(128, activation='relu'),      # Hidden layer
    Dense(10, activation='softmax')     # Output layer for 10 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
