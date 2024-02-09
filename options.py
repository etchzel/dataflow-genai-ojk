from apache_beam.options.pipeline_options import PipelineOptions

class UserOptions(PipelineOptions):
  @classmethod
  def _add_argparse_args(cls, parser):
    parser.add_value_provider_argument('--bucket_name', help='Name of the bucket to get the file from')
    parser.add_value_provider_argument('--prefix', help='Name of the folder prefix for the files to search')