import numpy as np
import onnxruntime as ort


class Tokenizer:
    def __init__(
        self,
        ortext_path,
        decode_model_path
    ):
        self.sess_options = ort.SessionOptions()
        try:
            self.sess_options.register_custom_ops_library(ortext_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to register ONNX Runtime extensions library: {ortext_path}"
            ) from exc
        self.sess_options.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(
            decode_model_path,
            self.sess_options
        )

    def decode(self, input_ids):
        decoder_inputs = dict(
            ids=np.array(input_ids, dtype=np.int64),
            fairseq=np.array([False], dtype=np.bool_)
        )
        return self.sess.run(None, decoder_inputs)
