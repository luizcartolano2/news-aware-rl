""" Sentiment Classifier using Llama for Brazilian Economic Headlines
This module provides a sentiment classifier that uses the Llama model to classify
the sentiment of Brazilian economic headlines.

"""
import os
from llama_cpp import Llama

from constants import MODELS_PATH


class LlamaSentimentClassifier:
    """
    A sentiment classifier that uses the Llama model to classify Brazilian economic headlines.
    This class initializes the Llama model and provides a method to classify the sentiment
    of a given headline as POSITIVO, NEGATIVO, or NEUTRO.
    Attributes:
        model_path (str): Path to the Llama model file.
        llm (Llama): Instance of the Llama model.
    """
    def __init__(
            self,
            model_path: str = f"{MODELS_PATH}/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            n_ctx: int = 1024,
            n_threads: int = os.cpu_count(),
            n_batch: int = 16
    ):
        self.model_path = model_path
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=n_batch,
            use_mlock=True,
            verbose=False
        )

    @staticmethod
    def __build_prompt(headline: str) -> str:
        """
        Builds the prompt for the Llama model to classify the sentiment of a given headline.
        :param headline: The headline to classify.
        :return: a formatted prompt string for the Llama model.
        """
        return (
            f"Classifique o sentimento da seguinte manchete de economia brasileira em relacao a cotacao do dolar "
            f"como uma das opções: POSITIVO, NEGATIVO ou NEUTRO.\n\n"
            f"Manchete: \"{headline}\"\nSentimento:"
        )

    def classify(self, headline: str) -> str:
        """
        Classifies the sentiment of a given headline using the Llama model.
        This method sends a prompt to the Llama model and interprets the response
        to determine the sentiment as POSITIVO, NEGATIVO, or NEUTRO.
        :param headline: The headline to classify.
        :return: a string indicating the sentiment: "POSITIVO", "NEGATIVO", or "NEUTRO".
        """
        prompt = self.__build_prompt(headline)
        response = self.llm(prompt, max_tokens=10, stop=["\n", "."], echo=False)
        raw = response["choices"][0]["text"].strip().upper()

        if "POSI" in raw:
            return "POSITIVO"
        elif "NEGA" in raw:
            return "NEGATIVO"
        elif "NEUT" in raw:
            return "NEUTRO"
        else:
            return "NEUTRO"
