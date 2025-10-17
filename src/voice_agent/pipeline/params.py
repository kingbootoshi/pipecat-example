from pipecat.pipeline.task import PipelineParams


def make_params(allow_interruptions: bool = False) -> PipelineParams:
    return PipelineParams(allow_interruptions=allow_interruptions)


__all__ = ["make_params"]

