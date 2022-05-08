from data_provider.models import Sample

from tensorflow import cast, float32, reshape


class Normalizer:

    @staticmethod
    def normalize_sample(sample: Sample) -> Sample:
        cast(sample.value, float32)
        sample.value /= 255
        sample.value = reshape(sample.value, (32, 32, 3))

        return sample
