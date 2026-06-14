import os

from .postprocess_utils import rich_transcription_postprocess
from .sensevoice_bin import SenseVoiceSmall


cur_dir = os.path.dirname(os.path.abspath(__file__))
asr_model_path = os.path.join(cur_dir, "models")


class AsrModel:
    def __init__(
        self,
        model_dir=asr_model_path,
        prefer_optimized_model=True,
        batch_size=10,
        language="zh",
        use_itn=True,
        intra_op_num_threads=2,
    ):
        self.language = language
        self.use_itn = use_itn
        self._model = SenseVoiceSmall(
            model_dir,
            prefer_optimized_model=prefer_optimized_model,
            batch_size=batch_size,
            intra_op_num_threads=intra_op_num_threads,
        )

    def __call__(self, wav_file_path):
        res = self._model(
            wav_file_path,
            language=self.language,
            use_itn=self.use_itn,
        )
        res = res[0][0].tolist()
        text = rich_transcription_postprocess(res[0])
        return text

    def get_runtime_status(self):
        return self._model.get_runtime_status()
