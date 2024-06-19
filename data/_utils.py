from models import Pipeline


PIPELINE = Pipeline(skip_load_model=True)


def wav2feature(path_input):
    return PIPELINE.wav2feature(path_input)
