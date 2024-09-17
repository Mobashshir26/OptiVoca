# OptiVoca:

This project demonstrates a real-time AI assistant that integrates speech recognition, webcam feed processing, and AI-based response generation. The assistant listens to voice input, processes the prompt, and responds based on both the spoken prompt and the captured image from the webcam. It uses the Google Gemini model for inference and generates voice output for the response.

## Features

- **Real-time Speech Recognition**: Uses Google's Web Speech API for accurate speech-to-text conversion.
- **Webcam Integration**: Captures frames from the webcam and encodes them as base64 images to pass to the AI model.
- **AI-based Responses**: The Google Gemini model is used to generate responses based on the prompt and webcam feed.
- **Text-to-Speech Output**: The assistant speaks out the AI-generated responses using Google Text-to-Speech (gTTS).
- **Multithreading**: Separate threads for audio listening and webcam stream to ensure real-time processing.

## Dependencies

The project relies on the following Python packages:

- `opencv-python` for capturing webcam feed
- `base64` for encoding images
- `dotenv` for loading environment variables
- `langchain` for building an AI model inference chain
- `langchain-google-genai` for integrating Google Gemini model
- `speech_recognition` for speech-to-text conversion
- `gtts` for text-to-speech conversion
- `pygame` for playing audio responses
- `threading` for managing concurrent processes

## How It Works

- **Webcam Stream**: The application starts the webcam feed and captures frames in real-time using OpenCV. Each frame is encoded in base64 format to be passed to the AI model.

- **Speech Recognition**: Using the `speech_recognition` library, the application listens for voice input and converts the spoken words into text via the Google Web Speech API.

- **AI Processing**: The assistant combines the recognized speech and the webcam image, passing them to the Google Gemini model (through LangChain) to generate a response.

- **Text-to-Speech**: The AI response is converted into speech using Google Text-to-Speech (gTTS), and the audio is played back using the `pygame` library.

   
