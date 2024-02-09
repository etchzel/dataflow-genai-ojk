from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from vertexai.preview.generative_models import GenerativeModel

from google.api_core.exceptions import ServerError
from google.api_core.exceptions import TooManyRequests

from apache_beam.io.components.adaptive_throttler import AdaptiveThrottler
from apache_beam.metrics.metric import Metrics
from apache_beam.ml.inference.base import ModelHandler
from apache_beam.utils import retry


class GeminiProHandler(ModelHandler):
  """
  Handler for Gemini API.
  """
  def __init__(
    self, 
    min_batch_size: Optional[int] = None,
    max_batch_size: Optional[int] = None,
    max_batch_duration_secs: Optional[int] = None,
    **kwargs):
    self._batching_kwargs = {}
    self._env_vars = kwargs.get('env_vars', {})
    if min_batch_size is not None:
      self._batching_kwargs["min_batch_size"] = min_batch_size
    if max_batch_size is not None:
      self._batching_kwargs["max_batch_size"] = max_batch_size
    if max_batch_duration_secs is not None:
      self._batching_kwargs["max_batch_duration_secs"] = max_batch_duration_secs

    self.throttled_secs = Metrics.counter(GeminiProHandler, "cumulativeThrottlingSeconds")
    self.throttler = AdaptiveThrottler(window_ms=1, bucket_ms=1, overload_ratio=2)

  def load_model(self) -> Any:
    """
    Load gemini-pro model
    """
    from vertexai.preview.generative_models import GenerativeModel
    model = GenerativeModel("gemini-pro")
    return model
  
  @retry.with_exponential_backoff(
    num_retries=5, 
    retry_filter=(lambda exception: isinstance(exception, (TooManyRequests, ServerError)))
  )
  def get_request(
      self,
      batch: Sequence[Any],
      model: GenerativeModel,
      throttle_delay_secs: int,
      inference_args: Optional[Dict[str, Any]]):
    import time
    import logging
    from google.api_core.exceptions import TooManyRequests
    from vertexai.preview.generative_models import FunctionDeclaration, Tool

    parser_func = FunctionDeclaration(
      name="get_score_sentiment",
      description="get the scoring and sentiment",
      parameters={
        "type": "object",
        "properties": {
          "sentiment": {"type": "string", "description": "sentiment"},
          "score": {"type": "integer", "description": "score of its sentiment"}
        }
      }
    )

    parser = Tool(function_declarations=[parser_func])

    result = []
    for data in batch:
      while self.throttler.throttle_request(time.time() * 1000):
        logging.info(
            "Delaying request for %d seconds due to previous failures",
            throttle_delay_secs)
        time.sleep(throttle_delay_secs)
        self.throttled_secs.inc(throttle_delay_secs)

      try:
        req_time = time.time()
        scoring = f"""
        Klasifikasikan sentiment text berikut terhadap entitas "{data.get('entity')}" menjadi positif, negatif, atau netral dan berikan probabilitas score nya.
        Jika netral maka berikan probabilitas score -0.2 hingga 0.2
        {data.get('text')}
        """

        response_sentiment = model.generate_content(
          scoring,
          generation_config={"temperature": 0.2}
        )

        parsing_result = f"""
        Ektraks sentiment dan score dari text berikut
        {response_sentiment.text}
        """

        response_parsing = model.generate_content(
          parsing_result,
          generation_config={"temperature": 0.2},
          tools=[parser]
        )

        sentiment = response_parsing.candidates[0].content.parts[0].function_call.args.get("sentiment", 'inference_error')
        score = response_parsing.candidates[0].content.parts[0].function_call.args.get("score", '')

        output = {
          "post_id": data.get("post_id"),
          "text": data.get("text"),
          "entity": data.get("entity"),
          "keyword": data.get("keyword"),
          "search_term": data.get("search_term"),
          "author_id": data.get("author_id"),
          "sentiment": sentiment,
          "score": score
        }

        self.throttler.successful_request(req_time * 1000)
        result.append(output)
      except TooManyRequests as e:
        logging.warning("request was limited by the service with code %i", e.code)
        raise
      except Exception as e:
        logging.error("unexpected exception raised as part of request, got %s", e)
        raise

    return result
  
  def run_inference(
    self, 
    batch: Sequence[Any], 
    model: GenerativeModel, 
    inference_args: Optional[Dict[str, Any]] = None
  ) -> Iterable[Any]:
    """
    Run inference
    """
    prediction = self.get_request(batch, model, throttle_delay_secs=5, inference_args=inference_args)
    return prediction
  
  def validate_inference_args(self, inference_args: Optional[Dict[str, Any]]):
    pass

  def batch_elements_kwargs(self) -> Mapping[str, Any]:
    return self._batching_kwargs