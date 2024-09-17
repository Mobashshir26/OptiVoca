import base64
from threading import Lock, Thread
import cv2
from cv2 import VideoCapture, imencode
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
import speech_recognition as sr
from gtts import gTTS
import pygame
import io
from speech_recognition import Microphone, Recognizer, UnknownValueError

load_dotenv()

class WebcamStream:
    def __init__(self):
        self.stream = VideoCapture(index=0)
        _, self.frame = self.stream.read()
        self.running = False
        self.lock = Lock()

    def start(self):
        if self.running:
            return self

        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.running:
            _, frame = self.stream.read()
            self.lock.acquire()
            self.frame = frame
            self.lock.release()

    def read(self, encode=False):
        self.lock.acquire()
        frame = self.frame.copy()
        self.lock.release()

        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer)

        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stream.release()


class Assistant:
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)

    def answer(self, prompt, image):
        if not prompt:
            return

        print("Prompt:", prompt)

        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image.decode()},
            config={"configurable": {"session_id": "unused"}},
        ).strip()

        print("Response:", response)

        if response:
            self._tts(response)

    def _tts(self, response):
        try:
            # Generate speech from text
            tts = gTTS(text=response, lang='en')
            
            # Use BytesIO to store the audio in memory without saving it to disk
            audio_stream = io.BytesIO()
            tts.write_to_fp(audio_stream)
            audio_stream.seek(0)

            # Initialize pygame mixer
            pygame.mixer.init()

            # Stop any currently playing audio
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()

            # Load the audio stream from memory
            pygame.mixer.music.load(audio_stream, 'mp3')
            pygame.mixer.music.play()

            # Wait until the audio finishes playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

        except Exception as e:
            print(f"Error generating or playing audio: {e}")

    def _create_inference_chain(self, model):
        SYSTEM_PROMPT = """
        You are a witty assistant that will use the chat history and the image 
        provided by the user to answer its questions.

        Use few words on your answers. Go straight to the point. Do not use any
        emoticons or emojis. Do not ask the user any questions.

        Be friendly and helpful. Show some personality. Do not be too formal.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                ),
            ]
        )

        chain = prompt_template | model | StrOutputParser()

        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )


webcam_stream = WebcamStream().start()


model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
assistant = Assistant(model)


recognizer = Recognizer()
microphone = Microphone()

def process_audio(audio):
    try:
        # Recognize speech using Google Web Speech API
        print("Recognizing speech...")
        prompt = recognizer.recognize_google(audio)
        print(f"Recognized text: {prompt}")
        
        # Send the recognized speech and webcam image to the assistant
        assistant.answer(prompt, webcam_stream.read(encode=True))
    except UnknownValueError:
        print("Could not understand the audio")
    except sr.RequestError as e:
        print(f"Error with Google Web Speech API: {e}")

def listen_for_audio():
    with microphone as source:
        print("Adjusting for ambient noise... Please wait!")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening for speech...")

        while True:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
            process_audio(audio)

# Start a thread to continuously listen to audio in the background
audio_thread = Thread(target=listen_for_audio)
audio_thread.start()

# Main loop to display the webcam feed
while True:
    cv2.imshow("Webcam", webcam_stream.read())
    if cv2.waitKey(1) in [27, ord("q")]:
        break

webcam_stream.stop()
cv2.destroyAllWindows()
audio_thread.join()
