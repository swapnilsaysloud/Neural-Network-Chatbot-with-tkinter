# Neural-Network-Chatbot-with-tkinter
This project is a chatbot built with Python, featuring a graphical user interface (GUI) developed using the tkinter library. The chatbot employs a neural network created with Keras to understand and respond to user input. It loads predefined intents and patterns from a JSON file, tokenizes user messages, and uses a bag-of-words approach to classify and generate responses based on those patterns. The project offers a user-friendly chat interface and can be further customized for a variety of conversational applications.

# Running the chatbot
To run the chatbot, we have two main files; train_chatbot.py and chatapp.py.

First, we train the model using the command in the terminal:

python train_chatbot.py
If we don’t see any error during training, we have successfully created the model. Then to run the app, we run the second file.

python chatgui.py
The program will open up a GUI window within a few seconds and you interact with your chatbot now!
