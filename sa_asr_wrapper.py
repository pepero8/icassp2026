class SAASRWrapper:
    """
    Wrapper class for SA-ASR model to perform diarization and transcription.
    """

    def __init__(self, sa_asr):
        self.sa_asr = sa_asr
        self.tokenizer = None  # todo: Initialize tokenizer(from transformer encoder)

    def diarize(self, audio):
        """
        Perform diarization and transciption on the given audio using the SA-ASR model.

        Args:
                audio (float32 tensor): 1sec audio chunk, shape (1, 22050).

        Returns:
                (1, D, T), spk embd and content embds concatenated
        """
        spkembd, text = self.sa_asr.transcribe(audio)

        # todo: Convert text to token ids using the tokenizer

        # todo: Concat speaker embeddings and content embeddings

        pass
