from typing import Any

from google.api_core.exceptions import ServerError
from google.api_core.exceptions import TooManyRequests

from apache_beam.io.components.adaptive_throttler import AdaptiveThrottler
from apache_beam.metrics.metric import Metrics
from apache_beam import DoFn
from apache_beam.utils import retry

class GeminiProHandler(DoFn):
  """
  Handler for Gemini API.
  """
  def __init__(self, throttle_delay_secs):
    self.throttle_delay_secs = throttle_delay_secs
    self.throttled_secs = Metrics.counter(GeminiProHandler, "cumulativeThrottlingSeconds")
    self.throttler = AdaptiveThrottler(window_ms=1, bucket_ms=1, overload_ratio=2)

  def setup(self) -> Any:
    """
    Load gemini-pro model
    """
    import vertexai
    from vertexai.preview.generative_models import GenerativeModel
    vertexai.init(project="datalabs-int-bigdata", location="us-central1")
    self.model = GenerativeModel("gemini-pro")
  
  @retry.with_exponential_backoff(
    num_retries=5, 
    retry_filter=(lambda exception: isinstance(exception, (TooManyRequests, ServerError)))
  )
  def get_request(self, data, throttle_delay_secs):
    import time
    import logging
    from google.api_core.exceptions import TooManyRequests
    from vertexai.preview.generative_models import (
      FunctionDeclaration, 
      Tool, 
      HarmCategory,
      HarmBlockThreshold
    )

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
    safety_config = {
      HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
      HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
      HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
      HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
      HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE
    }

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

      response_sentiment = self.model.generate_content(
        scoring,
        generation_config={"temperature": 0.2},
        safety_settings=safety_config
      )

      self.throttler.successful_request(req_time * 1000)

      try:
        parsing_result = f"""
        Ektraks sentiment dan score dari text berikut
        {response_sentiment.text}
        """
        response_parsing = self.model.generate_content(
          parsing_result,
          generation_config={"temperature": 0.2},
          tools=[parser],
          safety_settings=safety_config
        )
        sentiment = response_parsing.candidates[0].content.parts[0].function_call.args.get("sentiment", 'inference_error')
        score = response_parsing.candidates[0].content.parts[0].function_call.args.get("score", '')
      except:
        sentiment = "inference_error"
        score = ""

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
      return output
    except TooManyRequests as e:
      logging.warning("request was limited by the service with code %i", e.code)
      raise
    except Exception as e:
      logging.error("unexpected exception raised as part of request, got %s", e)
      raise
  
  def process(self, element):
    """
    Run inference
    """
    prediction = self.get_request(data=element, throttle_delay_secs=self.throttle_delay_secs)
    yield prediction