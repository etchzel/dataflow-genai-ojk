import argparse
import logging
import sys
import apache_beam as beam
# from modules.gemini_handler import GeminiProHandler
from modules.gemini_pardo import GeminiProHandler
from apache_beam.io.filesystem import CompressionTypes
from apache_beam.options.pipeline_options import PipelineOptions
# from apache_beam.ml.inference.base import RunInference
from apache_beam.io.gcp.bigquery import BigQuerySource

class JSONLWriter(beam.PTransform):
  def __init__(self,
                file_path_prefix, 
                coder=beam.coders.StrUtf8Coder(), 
                file_name_suffix='', 
                num_shards=0, 
                shard_name_template=None, 
                mime_type='application/json', 
                compression_type=CompressionTypes.AUTO, 
                *, 
                max_records_per_shard=None, 
                max_bytes_per_shard=None, 
                skip_if_empty=False):
    from modules.JSONLSink import _JSONLSink
    self._sink = _JSONLSink(
        file_path_prefix=file_path_prefix,
        coder=coder,
        file_name_suffix=file_name_suffix,
        num_shards=num_shards,
        shard_name_template=shard_name_template,
        mime_type=mime_type,
        compression_type=compression_type,
        max_records_per_shard=max_records_per_shard,
        max_bytes_per_shard=max_bytes_per_shard,
        skip_if_empty=skip_if_empty
    )
  
  def expand(self, pcoll):
    return (pcoll | beam.io.Write(self._sink))
  
def main(known_args, pipeline_args):
  runner = known_args.runner
  pipeline_options = PipelineOptions(pipeline_args, streaming=False, runner=runner)

  query_fb = """
  SELECT DISTINCT 
    post_id, 
    text, 
    posts.entity, 
    keyword, 
    search_term, 
    author.id as author_id 
  FROM `datalabs-int-bigdata.ojk_poc_new.facebook_posts` posts
  LEFT JOIN `datalabs-int-bigdata.ojk_poc_new.periode` periods 
  ON 1=1
    AND posts.entity = periods.entity
    AND DATE(TIMESTAMP_SECONDS(post_timestamp)) BETWEEN periods.from AND periods.to
  WHERE text is not null
  """

  query_twitter = """
  SELECT DISTINCT 
    post_id, 
    text, 
    posts.entity, 
    keyword, 
    search_term, 
    author.id as author_id 
  FROM `datalabs-int-bigdata.ojk_poc_new.twitter_posts` posts
  LEFT JOIN `datalabs-int-bigdata.ojk_poc_new.periode` periods 
  ON 1=1
    AND posts.entity = periods.entity
    AND DATE(TIMESTAMP_SECONDS(post_timestamp)) BETWEEN periods.from AND periods.to
  WHERE text is not null
  """

  query_tiktok = """
  SELECT DISTINCT 
    post_id, 
    text, 
    posts.entity, 
    keyword, 
    search_term,
    author.id as author_id 
  FROM `datalabs-int-bigdata.ojk_poc_new.tiktok_posts` posts
  LEFT JOIN `datalabs-int-bigdata.ojk_poc_new.periode` periods 
  ON 1=1
    AND posts.entity = periods.entity
    AND DATE(TIMESTAMP_SECONDS(post_timestamp)) BETWEEN periods.from AND periods.to
  WHERE text is not null AND REGEXP_CONTAINS(text, CONCAT("(?i:", posts.entity, ")"))
  """

  query_insta = """
  SELECT DISTINCT 
    post_id, 
    text, 
    posts.entity, 
    keyword, 
    search_term, 
    author.id as author_id 
  FROM `datalabs-int-bigdata.ojk_poc_new.instagram_posts` posts
  LEFT JOIN `datalabs-int-bigdata.ojk_poc_new.periode` periods 
  ON 1=1
    AND posts.entity = periods.entity
    AND DATE(TIMESTAMP_SECONDS(post_timestamp)) BETWEEN periods.from AND periods.to
  WHERE text is not null
  """

  if known_args.query_socmed == "facebook":
    query = query_fb
  elif known_args.query_socmed == "twitter":
    query = query_twitter
  elif known_args.query_socmed == "tiktok":
    query = query_tiktok
  elif known_args.query_socmed == "instagram":
    query = query_insta
  else:
    query = query_twitter

  with beam.Pipeline(options=pipeline_options) as pipeline:
    # predict = (
    #   pipeline
    #   | "Read Data" >> BigQuerySource(
    #       project="datalabs-int-bigdata", 
    #       query=query, 
    #       use_standard_sql=True
    #   )
    #   | "Inference" >> RunInference(model_handler=GeminiProHandler(min_batch_size=50, max_batch_size=1000))
    #   | "Flatten Batch" >> beam.FlatMap(lambda elements: elements)
    #   | "Write to GCS" >> JSONLWriter("gs://ojk-poc-scraping-564223160817/dataflow/facebook/inference", file_name_suffix=".json")
    # )

    predict = (
      pipeline
      | "Read Data" >> BigQuerySource(
          project="datalabs-int-bigdata", 
          query=query, 
          use_standard_sql=True
      )
      | "Inference" >> beam.ParDo(GeminiProHandler(throttle_delay_secs=int(known_args.throttle_delay)))
      | "Write to GCS" >> JSONLWriter(f"gs://ojk-poc-scraping-564223160817/dataflow/{known_args.query_socmed}/inference", file_name_suffix=".json")
    )

if __name__ == "__main__":
  # Configure logging
  log = logging.getLogger()
  log.setLevel(logging.INFO)
  stream_handler = logging.StreamHandler(sys.stdout)
  stream_handler.setLevel(logging.INFO)
  log.addHandler(stream_handler)

  # Use arguments
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--query_socmed", default="twitter"
  )
  parser.add_argument(
    "--throttle_delay", default=5
  )
  parser.add_argument(
    "--runner", default="DataflowRunner"
  )
  parser.add_argument(
    "--staging_location", default='gs://ojk-poc-scraping-564223160817/dataflow/staging'
  )
  parser.add_argument(
    "--temp_location", default='gs://ojk-poc-scraping-564223160817/dataflow/temp'
  )
  # parser.add_argument(
  #   "--template_location", default="gs://andi-ahr-bucket/dataflow/templates/batch-online-predict"
  # )
  parser.add_argument(
    "--requirements_file", default='requirements.txt'
  )
  parser.add_argument(
    "--setup_file", default="./setup.py"
  )
  parser.add_argument(
    "--region", default="asia-southeast2"
  )
  parser.add_argument(
    "--project", default='datalabs-int-bigdata'
  )
  known_args, pipeline_args = parser.parse_known_args()
  if known_args.runner=="DataflowRunner":
    pipeline_args.extend(
      ["--staging_location="+known_args.staging_location]
    )
    pipeline_args.extend(
      ["--temp_location="+known_args.temp_location]
    )
    # pipeline_args.extend(
    #   ["--template_location="+known_args.template_location]
    # )
    pipeline_args.extend(
      ["--requirements_file="+known_args.requirements_file]
    )
    pipeline_args.extend(
      ["--setup_file="+known_args.setup_file]
    )
    pipeline_args.extend(
      ["--region="+known_args.region]
    )
    pipeline_args.extend(
      ["--project="+known_args.project]
    )
    pipeline_args.extend(
      ["--machine_type=n2-standard-2"]
    )
    pipeline_args.extend(
      ["--max_num_workers=1"]
    )
  print(known_args, pipeline_args)
  main(known_args, pipeline_args)